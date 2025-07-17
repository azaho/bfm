from src.model.electrode_embedding import ElectrodeEmbedding_Learned, ElectrodeEmbedding_NoisyCoordinate, ElectrodeEmbedding_Learned_CoordinateInit, ElectrodeEmbedding_Zero
from src.model.preprocessing.laplacian_rereferencing import laplacian_rereference_batch
from src.training_setup.training_config import log
import torch
from src.training_setup.training_setup import TrainingSetup
from src.model.BFModule import BFModule
from src.model.transformer_implementation import Transformer
import torch.nn as nn
import numpy as np

# This file first defines the model components, then the training setup.

### 
# Flow of data in this model:
# The data starts out as (batch_size, n_electrodes, n_timesamples)
# 1. (batch_size, n_electrodes, n_timesamples) -> FFT -> (batch_size, n_electrodes, n_timebins, max_frequency_bin)
# 2. (batch_size, n_electrodes, n_timebins, max_frequency_bin) -> electrode transformer -> (batch_size, n_timebins, d_model)
# 3. (batch_size, n_timebins, d_model) -> time transformer -> (batch_size, 1, n_timebins, d_model)
# loss function: compare the output of the time transformer on half of electrodes 
#   to the output of the electrode transformer on the other half on the next timestep, using a contrastive loss
###

### DEFINING THE MODEL COMPONENTS ###

pretrain_config = {
    'mask': {
        'time': {
            'enabled': True,
            'consecutive_min': 1,
            'consecutive_max': 5, 
            'p': 0.10,
        },
        'freq': {
            'enabled': True,
            'consecutive_min': 1,
            'consecutive_max': 2,
            'p': 0.10,
        },

        'replacement_distribution': {
            'zero': 0.8, # replace the masked data with 0
            'none': 0.1, # leave the masked data as is
            'random_resample': 0.1, # replace the masked data with a random sample from the same electrode, different batch item
        }
    }
}


class Mask(BFModule):
    def __init__(self, mask_config):
        super().__init__()
        self.mask_config = mask_config
    
    def mask(self, x, axis, consecutive_min, consecutive_max, p):
        # x is of shape (batch_size, n_electrodes, n_timebins, n_freqs)

        x = x.transpose(axis, 0) # transpose so that the axis dimension is the first dimension
        # shape: (AXIS_DIM, batch_size, n_electrodes, OTHER_AXIS_DIM)

        n_bins = x.shape[0]
        mask_starts = torch.rand(n_bins) < p
        if not mask_starts.any(): mask_starts[torch.randint(0, n_bins, (1,))] = True # if got unlucky and nothing was masked, mask a random bin
        masked_indices = torch.zeros(n_bins, dtype=torch.bool)

        for i in range(n_bins):
            if mask_starts[i]:
                n_consecutive = np.random.randint(consecutive_min, consecutive_max + 1)
                masked_indices[i:i+n_consecutive] = True

                random_number = torch.rand(1)
                if random_number < self.mask_config['replacement_distribution']['zero']:
                    x[i:i+n_consecutive] = 0
                elif random_number < self.mask_config['replacement_distribution']['zero'] + self.mask_config['replacement_distribution']['none']:
                    pass
                elif random_number < self.mask_config['replacement_distribution']['zero'] + self.mask_config['replacement_distribution']['none'] + self.mask_config['replacement_distribution']['random_resample']:
                    batch_permutation = torch.randperm(x.shape[1])
                    x[i:i+n_consecutive] = x[i:i+n_consecutive, batch_permutation] # permute the batch dimension for those indices
        
        x = x.transpose(0, axis) # transpose back so that the axis dimension is back where it was

        return x, masked_indices

    def forward(self, x):
        # x is of shape (batch_size, n_electrodes, n_timebins, n_freqs)
        batch_size, n_electrodes, n_timebins, n_freqs = x.shape

        if self.mask_config['time']['enabled']:
            x, masked_indices_time = self.mask(x, 2, self.mask_config['time']['consecutive_min'], self.mask_config['time']['consecutive_max'], self.mask_config['time']['p'])
        
        if self.mask_config['freq']['enabled']:
            x, masked_indices_freq = self.mask(x, 3, self.mask_config['freq']['consecutive_min'], self.mask_config['freq']['consecutive_max'], self.mask_config['freq']['p'])

        return x, masked_indices_time, masked_indices_freq


class SpectrogramPreprocessor(BFModule):
    def __init__(self, spectrogram_parameters, output_dim=-1):
        super(SpectrogramPreprocessor, self).__init__()
        self.output_dim = output_dim
        self.spectrogram_parameters = spectrogram_parameters
        
        # from https://docs.pytorch.org/docs/stable/generated/torch.fft.rfftfreq.html
        # if n is nperseg, and d is 1/sampling_rate, then f = torch.arange((n + 1) // 2) / (d * n)
        # note: nperseg is always going to be even, so it simplifies to torch.arange(n/2) / n * sampling_rate
        # note: n = sampling_rate * tperseg, so it simplifies to torch.arange(sampling_rate * tperseg / 2) / tperseg
        #    which is a list that goes from 0 to sampling_rate / 2 in increments of sampling_rate / nperseg = 1 / tperseg
        # so max frequency bin is max_frequency * tperseg + 1 (adding one to make the endpoint inclusive)
        self.max_frequency_bin = round(self.spectrogram_parameters['max_frequency'] * self.spectrogram_parameters['tperseg'] + 1)

        # Transform FFT output to match expected output dimension
        self.output_transform = nn.Identity() if self.output_dim == -1 else nn.Linear(self.max_frequency_bin, self.output_dim)

        self.mask = Mask(pretrain_config['mask'])
    
    def forward(self, batch, mask=False):
        # batch['data'] is of shape (batch_size, n_electrodes, n_samples)
        # batch['metadata'] is a dictionary containing metadata like the subject identifier and trial id, sampling rate, etc.
        batch_size, n_electrodes = batch['data'].shape[:2]
        
        # Reshape for STFT
        x = batch['data'].reshape(batch_size * n_electrodes, -1)
        x = x.to(dtype=torch.float32)  # Convert to float32 for STFT
        
        
        # STFT parameters
        sampling_rate = batch['metadata']['sampling_rate']
        nperseg = round(self.spectrogram_parameters['tperseg'] * sampling_rate)
        noverlap = round(self.spectrogram_parameters['poverlap'] * nperseg)
        hop_length = nperseg - noverlap
        
        window = {
            'hann': torch.hann_window,
            'boxcar': torch.ones,
        }[self.spectrogram_parameters['window']](nperseg, device=x.device)
        
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

        # Trim to max frequency (using a pre-calculated max frequency bin)
        x = x[:, :self.max_frequency_bin, :]
            
        # Reshape back
        _, n_freqs, n_times = x.shape
        x = x.reshape(batch_size, n_electrodes, n_freqs, n_times)
        x = x.transpose(2, 3) # (batch_size, n_electrodes, n_timebins, n_freqs)
        
        # Z-score normalization
        x = x - x.mean(dim=[0, 2], keepdim=True)
        x = x / (x.std(dim=[0, 2], keepdim=True) + 1e-5)
        
        if mask:
            x, masked_indices_time, masked_indices_freq = self.mask(x) # shape: (batch_size, n_electrodes, n_timebins, n_freqs)

        # Transform to match expected output dimension
        x = self.output_transform(x)  # shape: (batch_size, n_electrodes, n_timebins, output_dim)
        x = x.to(dtype=batch['data'].dtype)

        output = {
            'data': x,
        }   
        if mask:
            output['masked_indices_time'] = masked_indices_time
            output['masked_indices_freq'] = masked_indices_freq
        return output

class BrainBERT(BFModule):
    def __init__(self, spectrogram_parameters, d_model=192, n_layers=4, n_heads=8, dropout=0.1, causal=False):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.spectrogram_preprocessor = SpectrogramPreprocessor(spectrogram_parameters, output_dim=-1)

        self.d_input = self.spectrogram_preprocessor.max_frequency_bin
        self.d_output = self.spectrogram_preprocessor.max_frequency_bin

        self.transformer = Transformer(d_input=self.d_input, d_model=d_model, d_output=self.d_output, 
                                            n_layer=n_layers, n_head=n_heads, causal=causal, 
                                            rope=True, rope_base=128, dropout=dropout)
    
    def forward(self, batch, mask=False, stop_at_block=None):
        # batch['data'] is of shape (batch_size, n_electrodes, n_timesamples)
        # batch['electrode_index'] is of shape (batch_size, n_electrodes)
        # batch['metadata'] is a dictionary containing metadata like the subject identifier and trial id, sampling rate, etc.        
        
        output = self.spectrogram_preprocessor(batch, mask=mask) 
        # data shape: (batch_size, n_electrodes, n_timebins, max_frequency_bin)
        # masked_indices_time shape: (n_timebins, )
        # masked_indices_freq shape: (n_freqs, )

        batch_size, n_electrodes, n_timebins, n_freqs = output['data'].shape

        output['data'] = output['data'].reshape(batch_size * n_electrodes, n_timebins, n_freqs)
        output['data'] = self.transformer(output['data'], stop_at_block=stop_at_block) # shape: (batch_size * n_electrodes, n_timebins, max_frequency_bin)
        output['data'] = output['data'].reshape(batch_size, n_electrodes, n_timebins, output['data'].shape[-1])

        return output

### DEFINING THE TRAINING SETUP ###

class andrii_brainbert(TrainingSetup):
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

        self.model = BrainBERT(
            spectrogram_parameters=config['model']['signal_preprocessing']['spectrogram_parameters'],
            d_model=config['model']['transformer']['d_model'],
            n_layers=config['model']['transformer']['n_layers'],
            n_heads=config['model']['transformer']['n_heads'],
            dropout=config['training']['dropout'],
            causal=True,
        ).to(device, dtype=config['model']['dtype'])
        config['model']['name'] = "AndriiBrainBERT"

        self.model_components['model'] = self.model


    # All of these will be applied to the batch before it is passed to the model
    def get_preprocess_functions(self, pretraining=False):
        preprocess_functions = []
        if self.config['model']['signal_preprocessing']['laplacian_rereference']:
            preprocess_functions.append(self._preprocess_laplacian_rereference)
        if self.config['model']['signal_preprocessing']['normalize_voltage']:
            preprocess_functions.append(self._preprocess_normalize_voltage) 
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

        losses = {}

        preprocessed_input = self.model.spectrogram_preprocessor(batch, mask=False)['data']

        output = self.model(batch, mask=True)
        # output is a dictionary with the following keys:
        # data shape: (batch_size, n_electrodes, n_timebins, max_frequency_bin)
        # masked_indices_time shape: (n_timebins, )
        # masked_indices_freq shape: (n_freqs, )

        masked_indices_time = output['masked_indices_time']
        masked_indices_freq = output['masked_indices_freq']
        losses['l2_time'] = torch.nn.functional.mse_loss(output['data'][:, :, masked_indices_time], preprocessed_input[:, :, masked_indices_time])
        losses['l2_freq'] = torch.nn.functional.mse_loss(output['data'][:, :, :, masked_indices_freq], preprocessed_input[:, :, :, masked_indices_freq])
        return losses


    def generate_frozen_features(self, batch, stop_at_block=3):        
        # INPUT:
        #   batch['data'] shape: (batch_size, n_electrodes, n_timesamples)
        #   batch['electrode_labels'] shape: list of length 1 (since it's the same across the batch), each element is a list of electrode labels
        #   batch['metadata']: dictionary containing metadata like the subject identifier and trial id, sampling rate, etc.
        # OUTPUT:
        #   features shape: (batch_size, n_electrodes or n_electrodes+1, n_timebins, *) where * can be arbitrary
        #   if n_electrodes+1, then the first dimension is the cls token
        batch['data'] = batch['data'].to(self.model.device, dtype=self.model.dtype, non_blocking=True)

        output = self.model(batch, mask=False, stop_at_block=stop_at_block)
        # output is a dictionary with the following keys:
        # data shape: (batch_size, n_electrodes, n_timebins, max_frequency_bin)

        features = output['data'][:, :, :, :] # shape: (batch_size, n_electrodes, n_timebins, d_model)
        return features