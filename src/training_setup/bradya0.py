'''
This file defines the model components and the training setup 
for the bradya0 model, which is a simple linear model for the onboarding task.
'''
from typing import Dict, Any

import torch
import torch.nn as nn

from src.model.BFModule import BFModule
from src.training_setup.training_setup import TrainingSetup

"""
Flow of data in this model:
The data starts out as (batch_size, n_electrodes, n_timesamples)

1. (batch_size, n_electrodes, n_timesamples) 
   -> FFT 
   -> (batch_size, n_electrodes, n_timebins, max_frequency_bin)

2. (batch_size, n_electrodes, n_timebins, max_frequency_bin) 
   -> electrode transformer 
   -> (batch_size, n_timebins, d_model)

3. (batch_size, n_timebins, d_model) 
   -> time transformer 
   -> (batch_size, 1, n_timebins, d_model)

Loss function: compare the output of the time transformer on half of electrodes 
to the output of the electrode transformer on the other half on the next timestep, using a contrastive loss.
"""

### DEFINING THE MODEL COMPONENTS ###

class LinearModel(BFModule):
    '''Averages over electrodes, split into bins and applies linear regression.'''
    def __init__(self, bin_size: int = 10):
        super().__init__()
        self.bin_size = bin_size
        self.linear = nn.Linear(bin_size, bin_size)
        
    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        '''
        Args:
            batch (dict): Dictionary containing:
                - 'data' (Tensor): Shape (batch_size, n_electrodes, n_timesamples).
                - 'electrode_index' (Tensor): Shape (batch_size, n_electrodes).
                - 'metadata' (dict): Contains subject identifier, trial ID, sampling rate, etc.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - pred (Tensor): Predicted future bins, shape (batch_size, n_bins - 1, bin_size).
                - x_next (Tensor): Actual next bins, shape (batch_size, n_bins - 1, bin_size).
                
        Raises:
            ValueError: If the number of samples is not sufficient to form bins of the specified size
        '''
        # Average over electrodes: (batch_size, n_electrodes, n_timesamples) -> (batch_size, n_timesamples)
        x = batch['data'].mean(dim=1)
        
        # Ensure we have enough samples to form bins
        n_samples = x.shape[1]
        n_bins = n_samples // self.bin_size
        if n_bins < 2:
            raise ValueError(f"Not enough samples to form bins of size {self.bin_size}.")
        
        # Reshape into bins: (batch_size, n_bins, bin_size)
        x = x[:, :n_bins * self.bin_size].reshape(x.shape[0], n_bins, self.bin_size)
        
        # Predict future bin from previous bins
        x_prev = x[:, :-1, :]  # (batch_size, n_bins - 1, bin_size)
        x_next = x[:, 1:, :]   # (batch_size, n_bins - 1, bin_size)
        
        # Apply linear regression to each previous bin
        pred = self.linear(x_prev)  # (batch_size, n_bins - 1, bin_size)
        return pred, x_next
    

### DEFINING THE TRAINING SETUP ###

class bradya0(TrainingSetup):
    '''Simple Linear Model for Onboarding Task'''
    
    def __init__(self, all_subjects, config, verbose=True):
        super().__init__(all_subjects, config, verbose)

    def initialize_model(self):
        config = self.config
        device = config['device']
        model_config = config['model']
        
        if not model_config['signal_preprocessing']['spectrogram']:
            raise ValueError("For the moment, we only support spectrogram preprocessing. Please set 'spectrogram' to True in the config.")

        # Initialize the linear model
        self.model = LinearModel().to(device, dtype=model_config['dtype'])
        model_config['name'] = "LinearModel"

        self.model_components['model'] = self.model


    def calculate_pretrain_loss(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        '''
        Calculate the L2 loss between the predicted future bins and the actual next bins.
        
        Args:
            batch (dict): Dictionary containing:
                - 'data' (Tensor): Shape (batch_size, n_electrodes, n_timesamples).
                - 'electrode_index' (Tensor): Shape (batch_size, n_electrodes).
                - 'metadata' (dict): Contains subject identifier, trial ID, sampling rate, etc.
            
        Returns:
            dict: Dictionary containing containing losses and their values.
                The final loss is the mean of all the losses.
                Accuracies are exempt and are just used for logging.
        '''      
        batch['data'] = batch['data'].to(self.model.device, dtype=self.model.dtype, non_blocking=True)
  
        x_pred, x_next = self.model(batch)  # shape: (batch_size, n_bins - 1, bin_size), (batch_size, n_bins - 1, bin_size)
        
        # Calculate L2 loss
        loss = torch.nn.functional.mse_loss(x_pred, x_next)
                
        return { 'l2_loss': loss }

    def generate_frozen_features(self, batch):
        # INPUT:
        #   batch['data'] shape: (batch_size, n_electrodes, n_timesamples)
        #   batch['electrode_labels'] shape: list of length 1 (since it's the same across the batch), each element is a list of electrode labels
        #   batch['metadata']: dictionary containing metadata like the subject identifier and trial id, sampling rate, etc.
        # OUTPUT:
        #   features shape: (batch_size, *) where * can be arbitrary (and will be concatenated for regression)
        batch['data'] = batch['data'].to(self.model.device, dtype=self.model.dtype, non_blocking=True)

        features, _ = self.model(batch)  # shape: (batch_size, n_bins - 1, bin_size)
        features = features.mean(dim=1)  # shape: (batch_size, bin_size)
        
        return features