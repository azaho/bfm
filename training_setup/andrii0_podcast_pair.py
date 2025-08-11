from model.electrode_embedding import ElectrodeEmbedding_Learned, ElectrodeEmbedding_NoisyCoordinate, ElectrodeEmbedding_Learned_CoordinateInit, ElectrodeEmbedding_Zero
from model.preprocessing.laplacian_rereferencing import laplacian_rereference_batch
from training_setup.training_config import log
import torch
from training_setup.training_setup import TrainingSetup
from model.BFModule import BFModule
from model.transformer_implementation import Transformer
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from subject.podcast_pair import PodcastTrialPairDataset, PodcastBatchPairSampler, load_podcast_subjects
from subject.podcast_pair import PreprocessCollatorPair

# This file first defines the model components, then the training setup.

### 
# Flow of data in this model:
# The data starts out as (batch_size, n_electrodes, n_timesamples) for two subjects
# 1. (batch_size, n_electrodes, n_timesamples) -> FFT -> (batch_size, n_electrodes, n_timebins, max_frequency_bin)
# 2. (batch_size, n_electrodes, n_timebins, max_frequency_bin) -> electrode transformer -> (batch_size, n_timebins, d_model)
# 3. (batch_size, n_timebins, d_model) -> time transformer -> (batch_size, 1, n_timebins, d_model)
# loss function: compare the output of the time transformer on subject A 
#   to the output of the electrode transformer on subject B at the same time point, using a contrastive loss
###


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
    
    def forward(self, batch):
        # batch['data'] is of shape (batch_size, n_electrodes, n_samples)
        # batch['metadata'] is a dictionary containing metadata like the subject identifier and trial id, sampling rate, etc.
        batch_size, n_electrodes = batch['data'].shape[:2]
        
        # Reshape for STFT
        x = batch['data'].reshape(batch_size * n_electrodes, -1)
        x = x.to(dtype=torch.float32)  # Convert to float32 for STFT
        
        # STFT parameters
        # new parameters based on andrii's sweeps
        nperseg = 512
        noverlap = 384
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

class andrii0_podcast_pair(TrainingSetup):
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
        config['model']['name'] = "AndriiOriginalPodcastPairModel"

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

    def _preprocess_add_electrode_indices(self, batch):
        electrode_indices = []
        subject_identifier = batch['metadata']['subject_identifier']
        for electrode_label in batch['electrode_labels'][0]:
            key = (subject_identifier, electrode_label)
            electrode_indices.append(self.electrode_embeddings.embeddings_map[key])
        batch['electrode_index'] = torch.tensor(electrode_indices, dtype=torch.long).unsqueeze(0).expand(batch['data'].shape[0], -1) # shape: (batch_size, n_electrodes)
        
        # Handle second subject
        if 'data_b' in batch:
            electrode_indices = []
            subject_identifier = batch['metadata_b']['subject_identifier']
            for electrode_label in batch['electrode_labels_b'][0]:
                key = (subject_identifier, electrode_label)
                electrode_indices.append(self.electrode_embeddings.embeddings_map[key])
            batch['electrode_index_b'] = torch.tensor(electrode_indices, dtype=torch.long).unsqueeze(0).expand(batch['data_b'].shape[0], -1) # shape: (batch_size, n_electrodes)
            
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

        # Handle second subject
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
        #   batch['data'] shape: (batch_size, n_electrodes, n_timesamples) for subject A
        #   batch['data_b'] shape: (batch_size, n_electrodes, n_timesamples) for subject B
        #   batch['electrode_index'] shape: (batch_size, n_electrodes) for subject A
        #   batch['electrode_index_b'] shape: (batch_size, n_electrodes) for subject B
        #   batch['metadata']: dictionary containing metadata for subject A
        #   batch['metadata_b']: dictionary containing metadata for subject B
        # OUTPUT:
        #   This function will output a dictionary of losses, with the keys being the loss names and the values being the loss values.
        #   The final loss is the mean of all the losses. Accuracies are exempt and are just used for logging.
        batch['data'] = batch['data'].to(self.model.device, dtype=self.model.dtype, non_blocking=True)
        batch['electrode_index'] = batch['electrode_index'].to(self.model.device, non_blocking=True)
        batch['data_b'] = batch['data_b'].to(self.model.device, dtype=self.model.dtype, non_blocking=True)
        batch['electrode_index_b'] = batch['electrode_index_b'].to(self.model.device, non_blocking=True)
        
        losses = {}
        config = self.config
        def _add_to_loss_contrastive(losses, output, target, loss_suffix):
            # output and target shape: (batch_size, n_timebins-future_bin_idx, d_model) - now both are time transformer outputs
            if config['training']['normalize_features']:
                output_ = output / (torch.norm(output, dim=-1, keepdim=True) + 0.001)
                target_ = target / (torch.norm(target, dim=-1, keepdim=True) + 0.001)
            similarity = output_.permute(1, 0, 2) @ target_.permute(1, 2, 0) # shape: (n_timebins-future_bin_idx, batch_size, batch_size)
            if config['training']['use_temperature_param']:
                similarity = similarity * torch.minimum(torch.exp(self.model.temperature_param), torch.tensor(config['training']['max_temperature_param'], device=self.model.device, dtype=self.model.dtype))
            expanded_arange = torch.arange(batch_size).unsqueeze(0).repeat(output.shape[1], 1).to(self.model.device, dtype=torch.long).reshape(-1)

            loss = torch.nn.functional.cross_entropy(similarity.view(-1, batch_size), expanded_arange)
            losses[f'contrastive_{loss_suffix}'] = loss
            if output_accuracy:
                accuracy_bin = (similarity.view(-1, batch_size).argmax(dim=-1) == expanded_arange).float().mean()
                losses[f'accuracy_{loss_suffix}'] = accuracy_bin
            return losses
        future_bin_idx = config['training']['future_bin_idx']

        batch_size, n_electrodes, n_samples = batch['data'].shape
        
        # Create batches for both subjects
        batch_a = {
            'data': batch['data'],
            'electrode_index': batch['electrode_index'],
            'metadata': batch['metadata'],
        }
        batch_b = {
            'data': batch['data_b'],
            'electrode_index': batch['electrode_index_b'],
            'metadata': batch['metadata_b'],
        }

        embeddings_a = self.electrode_embeddings(batch_a)
        embeddings_b = self.electrode_embeddings(batch_b)
        electrode_transformed_data_a, time_transformed_data_a = self.model(batch_a, embeddings_a) # shape: (batch_size, n_electrodes + 1, n_timebins, d_model), (batch_size, 1, n_timebins, d_model)
        electrode_transformed_data_b, time_transformed_data_b = self.model(batch_b, embeddings_b) # shape: (batch_size, n_electrodes + 1, n_timebins, d_model), (batch_size, 1, n_timebins, d_model)

        # Extract time transformer outputs (remove the extra dimension)
        time_output_a = time_transformed_data_a.squeeze(1)  # shape: (batch_size, n_timebins, d_model)
        time_output_b = time_transformed_data_b.squeeze(1)  # shape: (batch_size, n_timebins, d_model)

        # NEW: Compare time transformer outputs instead of time vs electrode
        # Person A's time transformer predicts Person B's time transformer output
        if future_bin_idx == 0:
            # When predicting current time bin, use all time bins for both tensors
            losses = _add_to_loss_contrastive(losses, time_output_a, time_output_b, 'a')
            # Person B's time transformer predicts Person A's time transformer output  
            losses = _add_to_loss_contrastive(losses, time_output_b, time_output_a, 'b')
        else:
            # When predicting future time bins, use offset slicing
            losses = _add_to_loss_contrastive(losses, time_output_a[:, :-future_bin_idx], time_output_b[:, future_bin_idx:], 'a')
            # Person B's time transformer predicts Person A's time transformer output  
            losses = _add_to_loss_contrastive(losses, time_output_b[:, :-future_bin_idx], time_output_a[:, future_bin_idx:], 'b')

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

    def load_dataloaders(self):
        """
        Load dataloaders for paired podcast training.
        Uses temporal block splitting to avoid temporal contamination.
        """
        config = self.config
        
        # Step 1: Generate all possible subject pairs
        train_subject_trials = config['training']['train_subject_trials']
        eval_subject_trials = config['training']['eval_subject_trials']
        
        # Create all possible non-self pairs
        train_subject_pairs = []
        for i, (subject_a_id, trial_a_id) in enumerate(train_subject_trials):
            for j, (subject_b_id, trial_b_id) in enumerate(train_subject_trials):
                if i < j:  # Avoid duplicates and self-pairs
                    train_subject_pairs.append((subject_a_id, subject_b_id))
        
        eval_subject_pairs = []
        for i, (subject_a_id, trial_a_id) in enumerate(eval_subject_trials):
            for j, (subject_b_id, trial_b_id) in enumerate(eval_subject_trials):
                if i < j:  # Avoid duplicates and self-pairs
                    eval_subject_pairs.append((subject_a_id, subject_b_id))
        
        if self.verbose:
            log(f"Generated {len(train_subject_pairs)} training pairs and {len(eval_subject_pairs)} evaluation pairs")
        
        # Step 2: Create paired datasets
        train_datasets = []
        test_datasets = []
        
        for subject_a_id, subject_b_id in train_subject_pairs:
            if self.verbose: 
                log(f"Creating paired dataset: {subject_a_id} + {subject_b_id}", indent=1, priority=1)
            
            subject_a = self.all_subjects[subject_a_id]
            subject_b = self.all_subjects[subject_b_id]
            
            # Calculate window size
            window_size = int(config['model']['context_length'] * subject_a.get_sampling_rate())
            
            # Create paired dataset
            dataset = PodcastTrialPairDataset(
                subject_a, subject_b, window_size,
                dtype=config['training']['data_dtype'],
                output_metadata=True,
                output_electrode_labels=True
            )
            
            # TEMPORAL BLOCK SPLITTING (80/20 split)
            # Split into contiguous train/test blocks to prevent temporal leakage
            total_windows = len(dataset)
            train_size = int(total_windows * (1 - config['training']['p_test']))
            
            # Create train indices (first 80% of windows)
            train_indices = list(range(train_size))
            # Create test indices (last 20% of windows) 
            test_indices = list(range(train_size, total_windows))
            
            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            test_dataset = torch.utils.data.Subset(dataset, test_indices)
            
            train_datasets.append(train_dataset)
            test_datasets.append(test_dataset)
            
            if self.verbose:
                log(f"Finished creating paired dataset: {len(dataset)} windows (train: {len(train_indices)}, test: {len(test_indices)})", indent=1, priority=1)

        if not train_datasets:
            raise ValueError("No valid paired datasets found. Make sure train_subject_pairs is properly configured.")

        train_dataset = ConcatDataset(train_datasets)
        test_dataset = ConcatDataset(test_datasets)

        # Step 3: Create dataloaders with custom sampler
        num_workers_dataloader_test = max(int(config['cluster']['num_workers_dataloaders'] * 0.15), 1)
        num_workers_dataloader_train = config['cluster']['num_workers_dataloaders'] - num_workers_dataloader_test
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=PodcastBatchPairSampler(
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
            batch_sampler=PodcastBatchPairSampler(
                [len(ds) for ds in test_datasets],
                batch_size=config['training']['batch_size'],
                shuffle=False  # No shuffling for test set
            ),
            num_workers=num_workers_dataloader_test,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=config['cluster']['prefetch_factor'],
            collate_fn=PreprocessCollatorPair(preprocess_functions=self.get_preprocess_functions(pretraining=True))
        )
        
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        
        if self.verbose:
            log(f"Created dataloaders: train={len(train_dataset)} samples, test={len(test_dataset)} samples") 