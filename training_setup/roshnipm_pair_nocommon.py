from model.electrode_embedding import ElectrodeEmbedding_Learned, ElectrodeEmbedding_NoisyCoordinate, ElectrodeEmbedding_Learned_CoordinateInit, ElectrodeEmbedding_Zero
from model.preprocessing.laplacian_rereferencing import laplacian_rereference_batch, laplacian_rereference_neural_data
from training_setup.training_config import log
import torch
from training_setup.training_setup import TrainingSetup
# from subject.dataset import SubjectTrialDataset, PreprocessCollator, SubjectBatchSampler
from subject.dataset_pair import SubjectTrialPairDataset, PreprocessCollatorPair, SubjectBatchPairSampler, load_subjects
from model.BFModule import BFModule
from model.transformer_implementation import Transformer
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import os

import numpy as np
from evaluation.neuroprobe.config import ROOT_DIR, SAMPLING_RATE, BRAINTREEBANK_SUBJECT_TRIAL_MOVIE_NAME_MAPPING
import pandas as pd
from training_setup.training_setup import TrainingSetup

# for main function
from evaluation.neuroprobe.config import NEUROPROBE_FULL_SUBJECT_TRIALS


# This file first defines the model components, then the training setup.

### 
# Flow of data in this model:
# The data starts out as (batch_size, n_electrodes, n_timesamples)
# 1. (batch_size, n_electrodes, n_timesamples) -> FFT -> (batch_size, n_electrodes, n_timebins, max_frequency_bin)
# 2. (batch_size, n_electrodes, n_timebins, max_frequency_bin) -> electrode transformer -> (batch_size, n_timebins, d_model)
# 3. (batch_size, n_timebins, d_model) -> time transformer -> (batch_size, 1, n_timebins, d_model)
# loss function: compare the output of the time transformer on half of electrodes 
#   to the output of the electrode transformer on the other half on the next timestep, using a contrastive loss
#
# NOTE: For paired training (comparing two brains), may need to reduce batch_size by ~50% 
# since you're processing twice the data per batch compared to single-brain training.
###

### Running command for initial benchmarking: 
# python pretrain.py --training.setup_name andrii0 --cluster.cache_subjects 1 --cluster.eval_at_beginning 1 --training.train_subject_trials btbank3_0,btbank7_0,btbank10_0,btbank4_1,btbank7_1 --training.eval_subject_trials btbank3_1,btbank3_2,btbank4_0,btbank4_2,btbank10_1  --cluster.eval_model_every_n_epochs 5 --training.eval_tasks speech,gpt2_surprisal
# python pretrain.py --training.setup_name roshnipm_pair_nocommon --cluster.cache_subjects 1 --cluster.eval_at_beginning 1 --training.train_subject_trials btbank3_0,btbank7_0,btbank10_0,btbank4_1,btbank7_1 --training.eval_subject_trials btbank3_1,btbank3_2,btbank4_0,btbank4_2,btbank10_1 --training.max_n_electrodes 64 --cluster.eval_model_every_n_epochs 5 --training.eval_tasks speech,gpt2_surprisal

### DEFINING THE MODEL COMPONENTS ###

class SpectrogramPreprocessor(BFModule):
    def __init__(self, output_dim=-1, max_frequency=200):
        super(SpectrogramPreprocessor, self).__init__()
        self.max_frequency = max_frequency
        self.output_dim = output_dim

        assert self.max_frequency == 200, "Max frequency must be 200"
        self.max_frequency_bin = 40 # XXX hardcoded max frequency bin
        
        # Transform FFT output to match expected output dimension
        self.output_transform = nn.Identity() if self.output_dim == -1 else nn.Linear(self.max_frequency_bin, self.output_dim)
    
    # edited to process both subjects
    def forward(self, batch):
        # batch['data'] is of shape (batch_size, n_electrodes, n_samples)
        # batch['metadata'] is a dictionary containing metadata like the subject identifier and trial id, sampling rate, etc.
        batch_size, n_electrodes = batch['data'].shape[:2]
        
        # Reshape for STFT
        x = batch['data'].reshape(batch_size * n_electrodes, -1)
        x = x.to(dtype=torch.float32)  # Convert to float32 for STFT
        
        # STFT parameters
        nperseg = 400
        noverlap = 350
        window = torch.hann_window(nperseg, device=x.device)
        hop_length = nperseg - noverlap
        
        # Compute STFT
        x = torch.stft(x,
                      n_fft=nperseg, 
                      hop_length=hop_length,
                      win_length=nperseg,
                      window=window,
                      return_complex=True,
                      normalized=False,
                      center=True)
        # Take magnitude
        x = torch.abs(x)

        # Pad or trim to max_frequency dimension
        if x.shape[1] < self.max_frequency_bin:
            x = torch.nn.functional.pad(x, (0, 0, 0, self.max_frequency_bin - x.shape[1]))
        else:
            x = x[:, :self.max_frequency_bin]
            
        # Reshape back
        _, n_freqs, n_times = x.shape
        x = x.reshape(batch_size, n_electrodes, n_freqs, n_times)
        x = x.transpose(2, 3) # (batch_size, n_electrodes, n_timebins, n_freqs)

        # Z-score normalization
        x = x - x.mean(dim=[0, 2], keepdim=True)
        x = x / (x.std(dim=[0, 2], keepdim=True) + 1e-5)

        # Transform to match expected output dimension
        x = self.output_transform(x)  # shape: (batch_size, n_electrodes, n_timebins, output_dim)
        
        return x.to(dtype=batch['data'].dtype)

class ElectrodeTransformer(BFModule):
    def __init__(self, d_model, n_layers=5, n_heads=12, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.transformer = Transformer(d_input=d_model, d_model=d_model, d_output=d_model, 
                                            n_layer=n_layers, n_head=n_heads, causal=False, 
                                            rope=False, dropout=dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, d_model))

    def forward(self, electrode_data, embeddings=None, only_cls_token=False):
        # electrode_data is of shape (batch_size, n_electrodes, n_timebins, d_model)
        # embeddings is of shape (batch_size, n_electrodes, d_model)
        batch_size, n_electrodes, n_timebins, d_model = electrode_data.shape
        
        if embeddings is not None:
            electrode_data = electrode_data + embeddings.unsqueeze(2) # shape: (batch_size, n_electrodes, n_timebins, d_model)

        electrode_data = electrode_data.transpose(1, 2) # (batch_size, n_timebins, n_electrodes, d_model)
        electrode_data = electrode_data.reshape(batch_size * n_timebins, n_electrodes, d_model)

        electrode_data = torch.cat([self.cls_token.unsqueeze(0).repeat(batch_size * n_timebins, 1, 1), electrode_data], dim=1) # shape: (batch_size * n_timebins, n_electrodes + 1, d_input)

        electrode_data = self.transformer(electrode_data) # shape: (batch_size * n_timebins, n_electrodes + 1, d_model)

        if only_cls_token:
            electrode_data = electrode_data[:, 0, :] # shape: (batch_size * n_timebins, d_model)
            electrode_data = electrode_data.reshape(batch_size, n_timebins, self.d_model)
        else:
            electrode_data = electrode_data.reshape(batch_size, n_timebins, n_electrodes + 1, self.d_model).transpose(1, 2) # shape: (batch_size, n_electrodes + 1, n_timebins, d_model)

        return electrode_data

class TimeTransformer(BFModule):
    def __init__(self, d_model, n_layers=5, n_heads=12, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.transformer = Transformer(d_input=d_model, d_model=d_model, d_output=d_model, 
                                            n_layer=n_layers, n_head=n_heads, causal=True, 
                                            rope=True, rope_base=128, dropout=dropout)
    
    def forward(self, electrode_transformed_data):
        # electrode_transformed_data is of shape (batch_size, n_timebins, d_model)
        batch_size, n_timebins, d_model = electrode_transformed_data.shape
        electrode_transformed_data = self.transformer(electrode_transformed_data) # shape: (batch_size, n_timebins, d_model)
        return electrode_transformed_data

class OriginalModel(BFModule):
    def __init__(self, d_model, n_layers_electrode=5, n_layers_time=5, n_heads=12, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_layers_electrode = n_layers_electrode
        self.n_layers_time = n_layers_time
        
        self.fft_preprocessor = SpectrogramPreprocessor(output_dim=d_model, max_frequency=200)
        self.electrode_transformer = ElectrodeTransformer(d_model, n_layers_electrode, n_heads, dropout)
        self.time_transformer = TimeTransformer(d_model, n_layers_time, n_heads, dropout)

        self.temperature_param = nn.Parameter(torch.tensor(0.0))

    def forward(self, batch, embeddings=None, electrode_transformer_only=False):
        # batch['data'] is of shape (batch_size, n_electrodes, n_timesamples)
        # batch['electrode_index'] is of shape (batch_size, n_electrodes)
        # batch['metadata'] is a dictionary containing metadata like the subject identifier and trial id, sampling rate, etc.        
        
        electrode_data = self.fft_preprocessor(batch) # shape: (batch_size, n_electrodes, n_timebins, d_model)

        electrode_transformed_data = self.electrode_transformer(electrode_data, embeddings) # shape: (batch_size, n_electrodes + 1, n_timebins, d_model)
        if electrode_transformer_only:
            return electrode_transformed_data
        
        time_transformed_data = self.time_transformer(electrode_transformed_data[:, 0, :, :]) # shape: (batch_size, n_timebins, d_model) 
        return electrode_transformed_data, time_transformed_data.unsqueeze(1) # shape: (batch_size, n_electrodes + 1, n_timebins, d_model), (batch_size, 1, n_timebins, d_model)


### DEFINING THE TRAINING SETUP ###

class roshnipm_pair_nocommon(TrainingSetup):
    def __init__(self, all_subjects, config, verbose=True):
        super().__init__(all_subjects, config, verbose)

    def initialize_model(self):
        """
            This function initializes the model.

            It must set the self.model_components dictionary to a dictionary of the model components, like
            {"model": model, "electrode_embeddings": electrode_embeddings}, where model and electrode_embeddings are PyTorch modules (those classes must inherit from model.BFModule)
        """
        config = self.config
        device = config['device']
        assert config['model']['signal_preprocessing']['spectrogram'] == True, "For the moment, we only support spectrogram"

        ### LOAD MODEL ###

        self.model = OriginalModel(
            d_model=config['model']['transformer']['d_model'],
            n_layers_electrode=config['model']['transformer']['n_layers'],
            n_layers_time=config['model']['transformer']['n_layers'],
            n_heads=config['model']['transformer']['n_heads'],
            dropout=config['training']['dropout']
        ).to(device, dtype=config['model']['dtype'])
        config['model']['name'] = "RoshniPM_PairModel"

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

        self.model_components['model'] = self.model
        self.model_components['electrode_embeddings'] = self.electrode_embeddings

    # for Laplacian rereferencing, we need the labels of the electrodes to determine the immediate neighbors of each electrode.
    # For example, for the right amygdala electrode RAMY2, the neighbors are RAMY1 and RAMY3
    def _preprocess_laplacian_rereference(self, batch):
        laplacian_rereference_batch(batch, remove_non_laplacian=False, inplace=True)

        if 'data_b' in batch:
            electrode_data = batch['data_b']
            electrode_labels = batch['electrode_labels_b']

            rereferenced_data, rereferenced_labels, original_electrode_indices = laplacian_rereference_neural_data(electrode_data, electrode_labels[0], remove_non_laplacian=False)
            
            batch['data_b'] = rereferenced_data
            batch['electrode_labels_b'] = [rereferenced_labels] * batch['data_b'].shape[0]

            if 'electrode_index_b' in batch:
                batch['electrode_index_b'] = batch['electrode_index_b'][:, original_electrode_indices]
        
        return batch
    
    def _preprocess_normalize_voltage(self, batch):
        batch['data'] = batch['data'] - torch.mean(batch['data'], dim=[0, 2], keepdim=True)
        batch['data'] = batch['data'] / (torch.std(batch['data'], dim=[0, 2], keepdim=True) + 1)
        
        if 'data_b' in batch:
            batch['data_b'] = batch['data_b'] - torch.mean(batch['data_b'], dim=[0, 2], keepdim=True)
            batch['data_b'] = batch['data_b'] / (torch.std(batch['data_b'], dim=[0, 2], keepdim=True) + 1)
        
        return batch

    def _preprocess_subset_electrodes(self, batch):
        batch_size = batch['data'].shape[0]
        n_electrodes = batch['data'].shape[1]
        subset_n_electrodes = min(n_electrodes, self.config['training']['max_n_electrodes']) if self.config['training']['max_n_electrodes']>0 else n_electrodes
        # Randomly subselect / permute electrodes
        selected_idx = torch.randperm(n_electrodes)[:subset_n_electrodes]
        batch['data'] = batch['data'][:, selected_idx]
        if 'electrode_labels' in batch:
            batch['electrode_labels'] = [[batch['electrode_labels'][0][i] for i in selected_idx]] * batch_size
        if 'electrode_index' in batch:
            batch['electrode_index'] = batch['electrode_index'][:, selected_idx]

        if 'data_b' in batch:
            batch_size = batch['data_b'].shape[0]
            n_electrodes = batch['data_b'].shape[1]
            subset_n_electrodes = min(n_electrodes, self.config['training']['max_n_electrodes']) if self.config['training']['max_n_electrodes']>0 else n_electrodes
            # Randomly subselect / permute electrodes
            selected_idx = torch.randperm(n_electrodes)[:subset_n_electrodes]
            batch['data_b'] = batch['data_b'][:, selected_idx]
            if 'electrode_labels_b' in batch:
                batch['electrode_labels_b'] = [[batch['electrode_labels_b'][0][i] for i in selected_idx]] * batch_size
            if 'electrode_index_b' in batch:
                batch['electrode_index_b'] = batch['electrode_index_b'][:, selected_idx]

        return batch

    def _preprocess_add_electrode_indices(self, batch):
        electrode_indices = []
        subject_identifier = batch['metadata']['subject_identifier']
        for electrode_label in batch['electrode_labels'][0]:
            key = (subject_identifier, electrode_label)
            electrode_indices.append(self.electrode_embeddings.embeddings_map[key])
        batch['electrode_index'] = torch.tensor(electrode_indices, dtype=torch.long).unsqueeze(0).expand(batch['data'].shape[0], -1) # shape: (batch_size, n_electrodes)
        
        # edited to process both subjects
        if 'data_b' in batch:
            electrode_indices = []
            subject_identifier = batch['metadata_b']['subject_identifier']
            for electrode_label in batch['electrode_labels_b'][0]:
                key = (subject_identifier, electrode_label)
                electrode_indices.append(self.electrode_embeddings.embeddings_map[key])
            batch['electrode_index_b'] = torch.tensor(electrode_indices, dtype=torch.long).unsqueeze(0).expand(batch['data_b'].shape[0], -1) # shape: (batch_size, n_electrodes)
            
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

    # TODO: change to be the contrastive loss in the two subjects
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
        if 'data_b' in batch:
            batch['data_b'] = batch['data_b'].to(self.model.device, dtype=self.model.dtype, non_blocking=True)
            batch['electrode_index_b'] = batch['electrode_index_b'].to(self.model.device, non_blocking=True)
        
        losses = {}
        config = self.config
        def _add_to_loss_contrastive(losses, output, target, loss_suffix):
            # output and target shape: (batch_size, n_electrodes, n_timebins-future_bin_idx, d_model)
            if config['training']['normalize_features']:
                output_ = output / (torch.norm(output, dim=-1, keepdim=True) + 0.001)
                target_ = target / (torch.norm(target, dim=-1, keepdim=True) + 0.001)
            similarity = output_.permute(1, 2, 0, 3) @ target_.permute(1, 2, 3, 0) # shape: (n_electrodes, n_timebins-future_bin_idx, batch_size, batch_size)
            if config['training']['use_temperature_param']:
                similarity = similarity * torch.minimum(torch.exp(self.model.temperature_param), torch.tensor(config['training']['max_temperature_param'], device=self.model.device, dtype=self.model.dtype))
            expanded_arange = torch.arange(batch_size).unsqueeze(0).unsqueeze(0).repeat(output.shape[1], output.shape[2], 1).to(self.model.device, dtype=torch.long).reshape(-1)

            loss = torch.nn.functional.cross_entropy(similarity.view(-1, batch_size), expanded_arange)
            losses[f'contrastive_{loss_suffix}'] = loss
            if output_accuracy:
                accuracy_bin = (similarity.view(-1, batch_size).argmax(dim=-1) == expanded_arange).float().mean()
                losses[f'accuracy_{loss_suffix}'] = accuracy_bin
            return losses
        future_bin_idx = config['training']['future_bin_idx']

        # Note that due to the RandomELectrodeCollator in the dataset class, the electrodes are already shuffled and cut to max_n_electrodes
        batch_size, n_electrodes, n_samples = batch['data'].shape
        
        # Split the batch into two halves, so that we can compute the contrastive loss between the two halves
        batch_a = {
            'data': batch['data'][:, :, :],
            'electrode_index': batch['electrode_index'],
            'metadata': batch['metadata'],
        }
        batch_b = {
            'data': batch['data_b'][:, :, :],
            'electrode_index': batch['electrode_index_b'],
            'metadata': batch['metadata_b'],
        }

        embeddings_a = self.electrode_embeddings(batch_a)
        embeddings_b = self.electrode_embeddings(batch_b)
        electrode_transformed_data_a, time_transformed_data_a = self.model(batch_a, embeddings_a) # shape: (batch_size, n_electrodes + 1, n_timebins, d_model), (batch_size, 1, n_timebins, d_model)
        electrode_transformed_data_b, time_transformed_data_b = self.model(batch_b, embeddings_b) # shape: (batch_size, n_electrodes + 1, n_timebins, d_model), (batch_size, 1, n_timebins, d_model)

        # add two symmetric loss components (for the electrode)
        losses = _add_to_loss_contrastive(losses, time_transformed_data_a[:, :, :-future_bin_idx], electrode_transformed_data_b[:, :1, future_bin_idx:], 'a')
        losses = _add_to_loss_contrastive(losses, time_transformed_data_b[:, :, :-future_bin_idx], electrode_transformed_data_a[:, :1, future_bin_idx:], 'b')

        return losses


    def generate_frozen_features(self, batch):
        # INPUT:
        #   batch['data'] shape: (batch_size, n_electrodes, n_timesamples)
        #   batch['electrode_labels'] shape: list of length 1 (since it's the same across the batch), each element is a list of electrode labels
        #   batch['metadata']: dictionary containing metadata like the subject identifier and trial id, sampling rate, etc.
        # OUTPUT:
        #   features shape: (batch_size, *) where * can be arbitrary (and will be concatenated for regression)
        batch['data'] = batch['data'].to(self.model.device, dtype=self.model.dtype, non_blocking=True)
        batch['electrode_index'] = batch['electrode_index'].to(self.model.device, non_blocking=True)

        embeddings = self.electrode_embeddings(batch)
        features = self.model(batch, embeddings, electrode_transformer_only=True) # shape: (batch_size, n_electrodes + 1, n_timebins, d_model)
        features = features[:, 0:1, :, :] # shape: (batch_size, 1, n_timebins, d_model) -- take just the cls token

        if self.config['cluster']['eval_aggregation_method'] == 'mean':
            features = features.mean(dim=[1, 2])
        elif self.config['cluster']['eval_aggregation_method'] == 'concat':
            features = features.reshape(batch['data'].shape[0], -1)
        return features

    # TODO: change to be the contrastive loss in the two subjects
    # everything should work without this extra function overriding the original training_setup.py load_dataloaders
    # Task 1: make dataset_pair_scuffed.py work with the original training_setup.py load_dataloaders
    # Task 2: make the contrastive loss work with the new dataset_pair_scuffed.py
    def load_dataloaders(self):
        """
            This function loads the dataloaders for the training and test sets.

            It must set the self.train_dataloader and self.test_dataloader attributes to the dataloaders (they are used in the pretraining code in pretrain.py)
        """
        config = self.config

        # Step 1: Load datasets
        # Group subjects by movie to create pairs
        movie_to_subject_trials = {}
        # for now, bypassing movie list
        for subject_identifier, trial_id in config['training']['train_subject_trials']:
            movie_key = f"{subject_identifier}_{trial_id}"
            if movie_key in BRAINTREEBANK_SUBJECT_TRIAL_MOVIE_NAME_MAPPING:
                movie_name = BRAINTREEBANK_SUBJECT_TRIAL_MOVIE_NAME_MAPPING[movie_key]
                # if movie_name not already in movie_to_subject_trials, create an empty list
                if movie_name not in movie_to_subject_trials:
                    movie_to_subject_trials[movie_name] = []
                # add the subject_identifier and trial_id to the list
                movie_to_subject_trials[movie_name].append((subject_identifier, trial_id))

        print(movie_to_subject_trials)
        
        # Create pairs of subjects watching the same movie
        paired_datasets = []
        for movie_name, subject_trials in movie_to_subject_trials.items():
            # if there are at least 2 subjects watching the same movie, create pairs
            if len(subject_trials) >= 2:
                # create pairs from all combinations
                for i in range(len(subject_trials)):
                    for j in range(len(subject_trials)):
                        if i == j:
                            continue
                        subject_a_id, trial_a_id = subject_trials[i]
                        subject_b_id, trial_b_id = subject_trials[j]
                        
                        if self.verbose: 
                            log(f"Creating paired dataset: {subject_a_id}_{trial_a_id} + {subject_b_id}_{trial_b_id} (movie: {movie_name})", indent=1, priority=1)
                        
                        # Calculate window size in samples
                        window_size = int(config['model']['context_length'] * SAMPLING_RATE)
                        
                        # Calculate actual movie duration from trigger times file
                        # Use the shorter of the two trials to ensure both have data
                        print(self.all_subjects)

                        subject_a = self.all_subjects[subject_a_id]
                        subject_b = self.all_subjects[subject_b_id]
                        
                        # Get movie duration from trigger times
                        trigger_times_file_a = os.path.join(ROOT_DIR, "subject_timings", f'sub_{int(subject_a_id.replace("btbank", ""))}_trial{int(trial_a_id):03}_timings.csv')
                        trigger_times_file_b = os.path.join(ROOT_DIR, "subject_timings", f'sub_{int(subject_b_id.replace("btbank", ""))}_trial{int(trial_b_id):03}_timings.csv')
                        
                        trigs_df_a = pd.read_csv(trigger_times_file_a)
                        trigs_df_b = pd.read_csv(trigger_times_file_b)
                        
                        # Get the end time from the last row (should be the 'end' type row)
                        movie_duration_a = trigs_df_a[trigs_df_a['type'] == 'end']['movie_time'].iloc[-1] if 'end' in trigs_df_a['type'].values else trigs_df_a['movie_time'].max()
                        movie_duration_b = trigs_df_b[trigs_df_b['type'] == 'end']['movie_time'].iloc[-1] if 'end' in trigs_df_b['type'].values else trigs_df_b['movie_time'].max()
                        
                        # Debug: Print movie durations
                        if self.verbose:
                            log(f"  Movie durations: {subject_a_id}_{trial_a_id}={movie_duration_a:.2f}s, {subject_b_id}_{trial_b_id}={movie_duration_b:.2f}s", indent=2, priority=1)
                        
                        # Use the shorter duration to ensure both subjects have data
                        total_time_seconds = min(movie_duration_a, movie_duration_b)
                        
                        window_time_seconds = window_size / SAMPLING_RATE
                        n_windows = int(total_time_seconds / window_time_seconds)
                        
                        # Create movie times for consecutive windows
                        movie_times = np.linspace(0, total_time_seconds, n_windows)
                        
                        # Create the paired dataset
                        dataset = SubjectTrialPairDataset(
                            subject_a, trial_a_id, window_size,
                            dtype=config['training']['data_dtype'],
                            output_metadata=True,
                            output_electrode_labels=True,
                            subject_b=subject_b, 
                            trial_id_b=trial_b_id,
                            movie_times=movie_times,
                            trigger_times_dir=os.path.join(ROOT_DIR, "subject_timings"),
                            sampling_rate=SAMPLING_RATE
                        )
                        
                        paired_datasets.append(dataset)
                        if self.verbose: 
                            log(f"Finished creating paired dataset: {len(dataset)} windows", indent=1, priority=1)

        if not paired_datasets:
            raise ValueError("No valid paired datasets found. Make sure subjects in train_subject_trials watch the same movies.")

        # Step 2: Split into train and test
        train_datasets = []
        test_datasets = []
        for dataset in paired_datasets:
            train_size = int(len(dataset) * (1 - config['training']['p_test']))
            train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
            train_datasets.append(train_dataset)
            test_datasets.append(test_dataset)
        train_dataset = ConcatDataset(train_datasets)
        test_dataset = ConcatDataset(test_datasets)

        # Step 3: Create dataloaders with custom sampler
        num_workers_dataloader_test = max(int(config['cluster']['num_workers_dataloaders'] * 0.15), 1)
        num_workers_dataloader_train = config['cluster']['num_workers_dataloaders'] - num_workers_dataloader_test
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=SubjectBatchPairSampler(
                [len(ds) for ds in train_datasets],
                batch_size=config['training']['batch_size'],
                shuffle=True
            ),
            num_workers=num_workers_dataloader_train,
            pin_memory=True,  # pin memory for faster GPU transfer
            persistent_workers=True,  # keep worker processes alive between iterations
            prefetch_factor=config['cluster']['prefetch_factor'],
            collate_fn=PreprocessCollatorPair(preprocess_functions=self.get_preprocess_functions(pretraining=True))
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_sampler=SubjectBatchPairSampler(
                [len(ds) for ds in test_datasets],
                batch_size=config['training']['batch_size'],
                shuffle=False
            ),
            num_workers=num_workers_dataloader_test,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=config['cluster']['prefetch_factor'],
            collate_fn=PreprocessCollatorPair(preprocess_functions=self.get_preprocess_functions(pretraining=True))
        )

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader