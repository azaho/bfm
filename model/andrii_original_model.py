import torch
import torch.nn as nn
from model.transformer_implementation import BFModule, Transformer
import numpy as np

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
        
        # Z-score normalization
        x = x - x.mean(dim=[0, 2], keepdim=True)
        x = x / (x.std(dim=[0, 2], keepdim=True) + 1e-5)

        
        # Transform to match expected output dimension
        x = x.transpose(2, 3) # (batch_size, n_electrodes, n_timebins, n_freqs)
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

    def forward(self, electrode_data, embeddings=None):
        # electrode_data is of shape (batch_size, n_electrodes, n_timebins, d_model)
        # embeddings is of shape (batch_size, n_electrodes, d_model)
        batch_size, n_electrodes, n_timebins, d_model = electrode_data.shape
        
        if embeddings is not None:
            electrode_data = electrode_data + embeddings.unsqueeze(2) # shape: (batch_size, n_electrodes, n_timebins, d_model)

        electrode_data = electrode_data.transpose(1, 2) # (batch_size, n_timebins, n_electrodes, d_model)
        electrode_data = electrode_data.reshape(batch_size * n_timebins, n_electrodes, d_model)

        electrode_data = torch.cat([self.cls_token.unsqueeze(0).repeat(batch_size * n_timebins, 1, 1), electrode_data], dim=1) # shape: (batch_size * n_timebins, n_electrodes + 1, d_input)

        electrode_data = self.transformer(electrode_data) # shape: (batch_size * n_timebins, n_electrodes + 1, d_model)

        electrode_data = electrode_data[:, 0, :] # shape: (batch_size * n_timebins, d_model)
        electrode_data = electrode_data.reshape(batch_size, n_timebins, self.d_model)
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
        
        self.spectrogram_preprocessor = SpectrogramPreprocessor(output_dim=d_model, max_frequency=200)
        self.electrode_transformer = ElectrodeTransformer(d_model, n_layers_electrode, n_heads, dropout)
        self.time_transformer = TimeTransformer(d_model, n_layers_time, n_heads, dropout)

        self.temperature_param = nn.Parameter(torch.tensor(0.0))

    def forward(self, batch, embeddings=None, evaluation_features=False):
        # batch['data'] is of shape (batch_size, n_electrodes, n_timesamples)
        # batch['electrode_index'] is of shape (batch_size, n_electrodes)
        # batch['metadata'] is a dictionary containing metadata like the subject identifier and trial id, sampling rate, etc.        
        
        electrode_data = self.spectrogram_preprocessor(batch) # shape: (batch_size, n_electrodes, n_timebins, d_model)

        electrode_transformed_data = self.electrode_transformer(electrode_data, embeddings) # shape: (batch_size, n_timebins, d_model)
        if evaluation_features:
            return electrode_transformed_data.unsqueeze(1)
        
        time_transformed_data = self.time_transformer(electrode_transformed_data) # shape: (batch_size, n_timebins, d_model)
        return electrode_transformed_data.unsqueeze(1), time_transformed_data.unsqueeze(1) # shape: (batch_size, 1, n_timebins, d_model)