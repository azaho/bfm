from bfm.model.encoders.electrode_embedding import ElectrodeEmbedding_Learned, ElectrodeEmbedding_NoisyCoordinate, ElectrodeEmbedding_Learned_CoordinateInit, ElectrodeEmbedding_Zero
from bfm.model.preprocessing.laplacian_rereferencing import laplacian_rereference_batch
from bfm.core.logger import log
import torch
from bfm.training.training_setup import TrainingSetup
from bfm.model.base import BFModule
from bfm.model.modules.transformer_implementation import Transformer
import torch.nn as nn
import json
import random

# This file first defines the model components, then the training setup.

### 
# Flow of data in this model:
# The data starts out as (batch_size, n_electrodes, n_timesamples)
# 1. (batch_size, n_electrodes, n_timesamples) -> STFT -> (batch_size, n_electrodes, n_timebins, max_frequency_bin)
# 2. (batch_size, n_electrodes, n_timebins, max_frequency_bin) -> transformer -> (batch_size, n_electrodes, n_timebins, max_frequency_bin)
# loss function: compare the output of the transformer to the real future data, using a MSE loss
###

### DEFINING THE MODEL COMPONENTS ###

from bfm.model.preprocessing.spectrogram import SpectrogramPreprocessor    

def mask_random_electrodes_and_timebins(batch, p_electrodes=0.5, p_timebins=0.5, key='data'):
    """
    input: 
        dictionary batch, with
            batch[key] shape: (batch_size, n_electrodes, n_timebins, d_input)
            batch['electrode_labels'] shape: list of length 1 (since it's the same across the batch), each element is a list of electrode labels
            and batch['metadata'] containing the subject identifier and trial id
    output:
        the same dictionary batch, with the following changes:
            batch[key], a masked version of the input data, with the same shape as the input data
            batch['mask_electrodes'], a mask of shape (n_electrodes,), with 1s where the data is masked and 0s where it is not
            batch['mask_timebins'], a mask of shape (n_timebins,), with 1s where the data is masked and 0s where it is not
    """
    batch_size, n_electrodes, n_timebins, d_input = batch[key].shape

    mask_electrodes = torch.rand(n_electrodes) < p_electrodes
    mask_timebins = torch.rand(n_timebins) < p_timebins

    mask_electrodes = mask_electrodes.to(batch[key].device, dtype=batch[key].dtype)
    mask_timebins = mask_timebins.to(batch[key].device, dtype=batch[key].dtype)

    batch[key] = batch[key] * (1-mask_electrodes.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)) * (1-mask_timebins.unsqueeze(0).unsqueeze(0).unsqueeze(-1))
    batch['mask_electrodes'] = mask_electrodes
    batch['mask_timebins'] = mask_timebins
    return batch

def get_dk_to_electrodes(batch, electrode_locations):
    # electrode_locations = electrode_locations[batch['metadata']['subject_identifier']]

    # Create a dictionary with DK locations as keys and lists of electrodes as values
    dk_to_electrodes = {}
    for electrode, dk_location in electrode_locations.items():
        if dk_location not in dk_to_electrodes:
            dk_to_electrodes[dk_location] = []
        dk_to_electrodes[dk_location].append(electrode)
    return dk_to_electrodes

def mask_inter_region(batch, electrode_locations, key='data'):
    electrode_locations = electrode_locations[batch['metadata']['subject_identifier']]
    dk_to_electrodes = get_dk_to_electrodes(batch, electrode_locations)

    batch_size, n_electrodes, n_timebins, d_input = batch[key].shape
    electrode_labels = batch['electrode_labels'][0]

    min_n_electrodes = 4
    min_electrode_regions = [dk for dk, electrodes in dk_to_electrodes.items() if len(electrodes) >= min_n_electrodes]
    selected_region = random.choice(min_electrode_regions)
    selected_electrode_labels = dk_to_electrodes[selected_region]
    selected_electrode_indices = torch.tensor([electrode_labels.index(label) for label in selected_electrode_labels], device=batch[key].device, dtype=torch.long)

    mask_electrodes = torch.zeros(n_electrodes, device=batch[key].device, dtype=batch[key].dtype)
    mask_electrodes[selected_electrode_indices] = 1

    allowed_remove_indices = torch.arange(n_electrodes, device=batch[key].device, dtype=torch.long)
    allowed_remove_indices = allowed_remove_indices[~torch.isin(allowed_remove_indices, selected_electrode_indices)]

    batch[key] = batch[key] * (1-mask_electrodes.unsqueeze(-1).unsqueeze(-1).unsqueeze(0))
    batch['mask_electrodes'] = mask_electrodes
    return batch, allowed_remove_indices

    
def mask_intra_region(batch, electrode_locations, p_electrodes=0.5, key='data'):
    electrode_locations = electrode_locations[batch['metadata']['subject_identifier']]
    dk_to_electrodes = get_dk_to_electrodes(batch, electrode_locations)

    batch_size, n_electrodes, n_timebins, d_input = batch[key].shape
    electrode_labels = batch['electrode_labels'][0]

    min_n_electrodes = 4
    min_electrode_regions = [dk for dk, electrodes in dk_to_electrodes.items() if len(electrodes) >= min_n_electrodes]
    selected_region = random.choice(min_electrode_regions)
    selected_region_electrode_labels = dk_to_electrodes[selected_region]
    selected_region_electrode_indices = torch.tensor([electrode_labels.index(label) for label in selected_region_electrode_labels], device=batch[key].device, dtype=torch.long)

    # delete all data from electrodes not in the selected region
    non_selected_region_electrode_indices = torch.tensor([i for i in range(n_electrodes) if i not in selected_region_electrode_indices], device=batch[key].device, dtype=torch.long)
    batch[key][:, non_selected_region_electrode_indices, :, :] *= 0
    # allow deleting those electrodes in the subsetting function later
    allowed_remove_indices = torch.arange(n_electrodes, device=batch[key].device, dtype=torch.long)
    allowed_remove_indices = allowed_remove_indices[~torch.isin(allowed_remove_indices, selected_region_electrode_indices)]

    n_electrode_indices_to_select = int(len(selected_region_electrode_indices) * p_electrodes)
    selected_electrode_indices = selected_region_electrode_indices[torch.randperm(len(selected_region_electrode_indices))[:n_electrode_indices_to_select]]

    mask_electrodes = torch.zeros(n_electrodes, device=batch[key].device, dtype=batch[key].dtype)
    mask_electrodes[selected_electrode_indices] = 1

    batch[key] = batch[key] * (1-mask_electrodes.unsqueeze(-1).unsqueeze(-1).unsqueeze(0))
    batch['mask_electrodes'] = mask_electrodes
    return batch, allowed_remove_indices

class MTMModel(BFModule):
    def __init__(self, d_model, spectrogram_parameters, d_input, n_layers=5, n_heads=12, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        
        self.transformer = Transformer(d_input=d_input, d_model=d_model, 
                                        d_output=d_input, 
                                        n_layer=n_layers, n_head=n_heads, causal=True, 
                                        rope=True, rope_base=128, dropout=dropout)

        # the four tasks in MtM
        self.prompt_cosmoothing = torch.nn.Parameter(torch.zeros(d_model))
        self.prompt_causal = torch.nn.Parameter(torch.zeros(d_model))
        self.prompt_interregion = torch.nn.Parameter(torch.zeros(d_model))
        self.prompt_intraregion = torch.nn.Parameter(torch.zeros(d_model))

    def forward(self, electrode_data, embeddings=None, electrode_transformer_only=False, stop_at_block=None):
        # electrode_data is of shape (batch_size, n_electrodes, n_timebins, d_input)
        # embeddings is of shape (batch_size, n_electrodes, d_model)
        batch_size, n_electrodes, n_timebins, d_input = electrode_data.shape

        positions = torch.arange(n_timebins, device=electrode_data.device, dtype=torch.long)
        positions = positions.unsqueeze(0).unsqueeze(0).expand(batch_size, n_electrodes, -1) # shape: (batch_size, n_electrodes, n_timebins)

        positions = positions.reshape(batch_size, n_electrodes * n_timebins)
        electrode_data = electrode_data.reshape(batch_size, n_electrodes * n_timebins, d_input)        
        if embeddings is not None:
            embeddings = embeddings.unsqueeze(2).expand(batch_size, n_electrodes, n_timebins, -1) # shape: (batch_size, n_electrodes, n_timebins, d_model)
            embeddings = embeddings.reshape(batch_size, n_electrodes * n_timebins, -1) # shape: (batch_size, n_electrodes * n_timebins, d_model)
        
        transformed_data = self.transformer(electrode_data, embeddings=embeddings, positions=positions, stop_at_block=stop_at_block) # shape: (batch_size, n_electrodes * n_timebins, d_output)
        
        d_output = transformed_data.shape[-1]
        transformed_data = transformed_data.reshape(batch_size, n_electrodes, n_timebins, d_output) # note: d_input = d_output = max_frequency_bin if stop_at_block is None, otherwise d_output = d_model
        
        return transformed_data


### DEFINING THE TRAINING SETUP ###

class mse_mtm(TrainingSetup):
    def __init__(self, all_subjects, config, verbose=True):
        super().__init__(all_subjects, config, verbose)

        # Load electrode locations
        # TODO: remove this hard coded path
        with open('analyses/andrii/25_08_21_extract_electrode_locations/btbank_electrode_locations.json', 'r') as f:
            self.electrode_locations = json.load(f)

    def initialize_model(self):
        """
            This function initializes the model.

            It must set the self.model_components dictionary to a dictionary of the model components, like
            {"model": model, "electrode_embeddings": electrode_embeddings}, where model and electrode_embeddings are PyTorch modules (those classes must inherit from bfm.model.base)
        """
        config = self.config
        device = config['device']
        assert config['model']['signal_preprocessing']['spectrogram'] == True, "For the moment, we only support spectrogram"

        self.p_mask_electrodes = 0.5
        self.p_mask_timebins = 0.5
        if 'other' in config:
            if 'p_mask_electrodes' in config['other']:
                self.p_mask_electrodes = float(config['other']['p_mask_electrodes'])
            if 'p_mask_timebins' in config['other']:
                self.p_mask_timebins = float(config['other']['p_mask_timebins'])

        ### LOAD MODEL ###

        self.fft_preprocessor = SpectrogramPreprocessor(config['model']['signal_preprocessing']['spectrogram_parameters'], output_dim=-1)

        self.model = MTMModel(
            spectrogram_parameters=config['model']['signal_preprocessing']['spectrogram_parameters'],
            d_model=config['model']['transformer']['d_model'],
            d_input=self.fft_preprocessor.max_frequency_bin,
            n_layers=config['model']['transformer']['n_layers'],
            n_heads=config['model']['transformer']['n_heads'],
            dropout=config['training']['dropout']
        ).to(device, dtype=config['model']['dtype'])
        config['model']['name'] = "SimpleMSEAutoregressiveModel"

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
        batch['electrode_index'] = torch.tensor(electrode_indices, dtype=torch.long).unsqueeze(0).repeat(batch['data'].shape[0], 1) # shape: (batch_size, n_electrodes)
        return batch
    
    def _preprocess_subset_electrodes(self, batch, output_selected_idx=False, allowed_remove_idx=None, force_remove_idx=None, keys=['data']):
        batch, selected_idx = super()._preprocess_subset_electrodes(batch, output_selected_idx=True, allowed_remove_idx=allowed_remove_idx, force_remove_idx=force_remove_idx, keys=keys)
        batch_size = batch['data'].shape[0]
        if 'electrode_index' in batch:
            batch['electrode_index'] = batch['electrode_index'][:, selected_idx].clone()
        if output_selected_idx:
            return batch, selected_idx
        return batch
    
    # All of these will be applied to the batch before it is passed to the model
    def get_preprocess_functions(self, pretraining=False):
        preprocess_functions = []
        if self.config['model']['signal_preprocessing']['laplacian_rereference']:
            preprocess_functions.append(self._preprocess_laplacian_rereference)
        if self.config['model']['signal_preprocessing']['normalize_voltage']:
            preprocess_functions.append(self._preprocess_normalize_voltage) 
        preprocess_functions.append(self._preprocess_add_electrode_indices)
        # if pretraining:
        #     preprocess_functions.append(self._preprocess_subset_electrodes)
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

        loss_type = random.choice(['cosmoothing', 'causal', 'inter_region', 'intra_region'])
        
        batch['preprocessed_data_clean'] = preprocessed_data.clone()
        batch['preprocessed_data'] = preprocessed_data.clone()
        allowed_remove_indices = None
        force_remove_indices = None
        if loss_type == 'cosmoothing':
            batch = mask_random_electrodes_and_timebins(batch, p_electrodes=self.p_mask_electrodes, p_timebins=self.p_mask_timebins, key='preprocessed_data')
        elif loss_type == 'causal':
            pass
        elif loss_type == 'inter_region':
            batch, allowed_remove_indices = mask_inter_region(batch, self.electrode_locations, key='preprocessed_data')
        elif loss_type == 'intra_region':
            batch, force_remove_indices = mask_intra_region(batch, self.electrode_locations, p_electrodes=self.p_mask_electrodes, key='preprocessed_data')
            # force_remove_indices = allowed_remove_indices
        
        batch, selected_idx = self._preprocess_subset_electrodes(batch, allowed_remove_idx=allowed_remove_indices, force_remove_idx=force_remove_indices, output_selected_idx=True, keys=['preprocessed_data_clean', 'preprocessed_data'])
        if 'mask_electrodes' in batch:
            batch['mask_electrodes'] = batch['mask_electrodes'][selected_idx]
        embeddings = embeddings[:, selected_idx]

        transformed_data = self.model(batch['preprocessed_data'], embeddings) # shape: (batch_size, n_electrodes, n_timebins, d_output)

        if loss_type == 'causal':
            n_timebins = preprocessed_data.shape[2]
            mse = torch.nn.functional.mse_loss(transformed_data[:, :, :n_timebins-self.config['training']['future_bin_idx'], :], batch['preprocessed_data_clean'][:, :, self.config['training']['future_bin_idx']:, :])
        else:
            mse = torch.nn.functional.mse_loss(transformed_data[:, batch['mask_electrodes'].bool(), :, :], batch['preprocessed_data_clean'][:, batch['mask_electrodes'].bool(), :, :])

        losses = {
            'mse_'+loss_type: mse,
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