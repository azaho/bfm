import torch
import torch.nn as nn

from bfm.training.setup_registry import setups
from bfm.training.training_setup import TrainingSetup
from bfm.core.logger import log

from bfm.model.base import BFModule
from bfm.model.modules.transformer_implementation import Transformer
from bfm.model.preprocessing.spectrogram import SpectrogramPreprocessor    
from bfm.model.preprocessing.laplacian_rereferencing import laplacian_rereference_batch
from bfm.model.encoders.electrode_embedding import (
    ElectrodeEmbedding_Learned, 
    ElectrodeEmbedding_NoisyCoordinate, 
    ElectrodeEmbedding_Learned_CoordinateInit, 
    ElectrodeEmbedding_Zero
)


# This file first defines the model components, then the training setup.

### 
# Flow of data in this model:
# The data starts out as (batch_size, n_electrodes, n_timesamples)
# 1. (batch_size, n_electrodes, n_timesamples) -> STFT -> (batch_size, n_electrodes, n_timebins, max_frequency_bin)
# 2. (batch_size, n_electrodes, n_timebins, max_frequency_bin) -> transformer -> (batch_size, n_electrodes, n_timebins, max_frequency_bin)
# loss function: compare the output of the transformer to the real future data, using a MSE loss
###

### DEFINING THE MODEL COMPONENTS ###

from .mse_rm import SimpleTransformerModel


### DEFINING THE TRAINING SETUP ###
@setups.register("mse_ar")
class mse_ar(TrainingSetup):
    def __init__(self, all_subjects, config, verbose=True):
        super().__init__(all_subjects, config, verbose)

    def initialize_model(self):
        """
            This function initializes the model.

            It must set the self.model_components dictionary to a dictionary of the model components, like
            {"model": model, "electrode_embeddings": electrode_embeddings}, where model and electrode_embeddings are PyTorch modules (those classes must inherit from bfm.model.base)
        """
        config = self.config
        device = config['device']
        assert config['model']['signal_preprocessing']['spectrogram'] == True, "For the moment, we only support spectrogram"

        ### LOAD MODEL ###

        self.fft_preprocessor = SpectrogramPreprocessor(config['model']['signal_preprocessing']['spectrogram_parameters'], output_dim=-1)

        d_input = self.fft_preprocessor.n_freqs
        self.model = SimpleTransformerModel(
            spectrogram_parameters=config['model']['signal_preprocessing']['spectrogram_parameters'],
            d_model=config['model']['transformer']['d_model'],
            d_input=d_input,
            n_layers=config['model']['transformer']['n_layers'],
            n_heads=config['model']['transformer']['n_heads'],
            dropout=config['training']['dropout']
        ).to(device, dtype=config['model']['dtype'])
        config['model']['name'] = "SimpleTransformerModel"

        ### LOAD ELECTRODE EMBEDDINGS ###

        electrode_embeddings_class = { # Select the right class based on the config
            'learned': ElectrodeEmbedding_Learned,
            'zero': ElectrodeEmbedding_Zero,
            'coordinate_init': ElectrodeEmbedding_Learned_CoordinateInit,
            'noisy_coordinate': ElectrodeEmbedding_NoisyCoordinate,
        }[config['model']['electrode_embedding']['type']]

        self.electrode_embeddings = electrode_embeddings_class( # Initialize the electrode embeddings
            config['model']['transformer']['d_model'], 
            embedding_dim=config['model']['electrode_embedding']['dim'],
            coordinate_noise_std=config['model']['electrode_embedding']['coordinate_noise_std'],
        ).to(device, dtype=config['model']['dtype'])

        for subject in self.all_subjects.values(): # we need to add every subject one by one to create the embeddings map (every electrode of every subject gets its own embedding)
            if self.verbose:
                log(f"Adding subject {subject.subject_identifier} to electrode embeddings...", priority=0)
            self.electrode_embeddings.add_subject(subject)
        self.electrode_embeddings = self.electrode_embeddings.to(device, dtype=config['model']['dtype']) # moving to device again to ensure the new parameters are on the correct device

        ### ADDING MODEL COMPONENTS TO THE DICTIONARY ###

        self.model_components['fft_preprocessor'] = self.fft_preprocessor
        self.model_components['model'] = self.model
        self.model_components['electrode_embeddings'] = self.electrode_embeddings

    def _preprocess_add_electrode_indices(self, batch):
        electrode_indices = []
        subject_identifier = batch['metadata']['subject_identifier']
        for electrode_label in batch['electrode_labels'][0]:
            key = (subject_identifier, electrode_label)
            electrode_indices.append(self.electrode_embeddings.embeddings_map[key])
        batch['electrode_index'] = torch.tensor(electrode_indices, dtype=torch.long).unsqueeze(0).expand(batch['data'].shape[0], -1) # shape: (batch_size, n_electrodes)
        return batch
    def _preprocess_subset_electrodes(self, batch):
        batch, selected_idx = super()._preprocess_subset_electrodes(batch, output_selected_idx=True)
        batch_size = batch['data'].shape[0]
        if 'electrode_index' in batch:
            batch['electrode_index'] = batch['electrode_index'][:, selected_idx]
        return batch
    
    # All of these will be applied to the batch before it is passed to the model
    def get_preprocess_functions(self, pretraining=False):
        preprocess_functions = []
        if self.config['model']['signal_preprocessing']['laplacian_rereference']:
            preprocess_functions.append(self._preprocess_laplacian_rereference)
        if self.config['model']['signal_preprocessing']['normalize_voltage']:
            preprocess_functions.append(self._preprocess_normalize_voltage) 
        preprocess_functions.append(self._preprocess_add_electrode_indices)
        if pretraining:
            preprocess_functions.append(self._preprocess_subset_electrodes)
        return preprocess_functions

    def calculate_pretrain_loss(self, batch, output_accuracy=True):
        # INPUT:
        #   batch['data'] shape: (batch_size, n_electrodes, n_timesamples)
        #   batch['electrode_index'] shape: (batch_size, n_electrodes)
        #   batch['metadata']: dictionary containing metadata like the subject identifier and trial id, sampling rate, etc.
        # OUTPUT:
        #   This function will output a dictionary of losses, with the keys being the loss names and the values being the loss values.
        #   The final loss is the mean of all the losses. Accuracies are exempt and are just used for logging.
        batch['data'] = batch['data'].to(self.model.device, dtype=self.model.dtype, non_blocking=True)
        batch['electrode_index'] = batch['electrode_index'].to(self.model.device, non_blocking=True)
        
        embeddings = self.electrode_embeddings(batch)
        preprocessed_data = self.fft_preprocessor(batch) # shape: (batch_size, n_electrodes, n_timebins, d_input)
        transformed_data = self.model(preprocessed_data, embeddings) # shape: (batch_size, n_electrodes, n_timebins, d_output)

        losses = {
            "mse_a": torch.nn.functional.mse_loss(transformed_data[:, :, :-self.config['training']['future_bin_idx'], :], preprocessed_data[:, :, self.config['training']['future_bin_idx']:, :])
        }
        return losses

    def generate_frozen_features(self, batch, stop_at_block=-2):
        # INPUT:
        #   batch['data'] shape: (batch_size, n_electrodes, n_timesamples)
        #   batch['electrode_labels'] shape: list of length 1 (since it's the same across the batch), each element is a list of electrode labels
        #   batch['metadata']: dictionary containing metadata like the subject identifier and trial id, sampling rate, etc.
        # OUTPUT:
        #   features shape: (batch_size, n_electrodes or n_electrodes+1, n_timebins, *) where * can be arbitrary
        #   if n_electrodes+1, then the first dimension is the cls token
        batch['data'] = batch['data'].to(self.model.device, dtype=self.model.dtype, non_blocking=True)
        batch['electrode_index'] = batch['electrode_index'].to(self.model.device, non_blocking=True)

        embeddings = self.electrode_embeddings(batch)
        preprocessed_data = self.fft_preprocessor(batch) # shape: (batch_size, n_electrodes, n_timebins, d_input)
        features = self.model(preprocessed_data, embeddings, stop_at_block=stop_at_block) # shape: (batch_size, n_electrodes, n_timebins, d_model)
        return features