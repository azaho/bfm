'''
Linear Model Training Setup for tvpower

Instructions:
This implements a simple linear model that:
1. Takes the data and averages over all electrodes
2. Splits into bins of predetermined size
3. Uses linear regression to predict the future bin from the past bin using L2 loss

For evaluation (generate_frozen_features), uses the model's learned predictions as features
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from training_setup.training_setup import TrainingSetup
from model.BFModule import BFModule
from model.preprocessing.laplacian_rereferencing import laplacian_rereference_batch
from training_setup.training_config import log

###MODEL COMPONENTS ###

class LinearModel(BFModule):
    def __init__(self, bin_size):
        super().__init__()
        self.bin_size = bin_size
        # Linear layer: input all timestamps in current bin, output all timestamps in next bin
        self.linear = nn.Linear(bin_size, bin_size)

    def forward(self, x):
        # x: (batch_size, n_electrodes, n_timesamples)
        #avg over all electrodes
        x = x.mean(dim=1)  # (batch_size, n_timesamples)
        
        # Bin the time dimension
        n_time = x.shape[1]
        n_bins = n_time // self.bin_size
        x = x[:, :n_bins * self.bin_size]  # remove remainder to fit bins
        x = x.reshape(x.shape[0], n_bins, self.bin_size)  # (batch, n_bins, bin_size)
        
        # Predict next bin from previous bin
        #input: x[:, :-1, :] (all timestamps in current bins), Target: x[:, 1:, :] (all timestamps in next bins)
        pred = self.linear(x[:, :-1, :])  # (batch, n_bins-1, bin_size)
        return pred, x[:, 1:, :]  # returns (prediction, target)

###TRAINING SETUP ###

class tvpower_linear(TrainingSetup):
    def __init__(self, all_subjects, config, verbose=True):
        super().__init__(all_subjects, config, verbose)
        #use bin size from config, default to 10
        self.bin_size = config.get('bin_size', 10)

    def initialize_model(self):
        """
            This function initializes the model.

            It must set the self.model_components dictionary to a dictionary of the model components, like
            {"model": model}, where model is a PyTorch module (must inherit from model.BFModule)
        """
        config = self.config
        device = config['device']

        ### LOAD MODEL ###

        self.model = LinearModel(self.bin_size).to(device, dtype=config['model']['dtype'])
        config['model']['name'] = "TVPowerLinearModel"

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
        #   batch['metadata']: dictionary containing metadata like the subject identifier and trial id, sampling rate, etc.
        # OUTPUT:
        #   This function will output a dictionary of losses, with the keys being the loss names and the values being the loss values.
        #   The final loss is the mean of all the losses. Accuracies are exempt and are just used for logging.
        batch['data'] = batch['data'].to(self.model.device, dtype=self.model.dtype, non_blocking=True)
        
        # Apply preprocessing
        if self.config['model']['signal_preprocessing']['laplacian_rereference']:
            laplacian_rereference_batch(batch, remove_non_laplacian=False, inplace=True)
            
        if self.config['model']['signal_preprocessing']['normalize_voltage']:
            batch['data'] = batch['data'] - torch.mean(batch['data'], dim=[0, 2], keepdim=True)
            batch['data'] = batch['data'] / (torch.std(batch['data'], dim=[0, 2], keepdim=True) + 1)
        
        # Pass through model
        pred, target = self.model(batch['data'])
        
        # Compute MSE loss
        loss = F.mse_loss(pred, target)
        return {"mse_loss": loss}

    def generate_frozen_features(self, batch):
        # INPUT:
        #   batch['data'] shape: (batch_size, n_electrodes, n_timesamples)
        #   batch['electrode_labels'] shape: list of length 1 (since it's the same across the batch), each element is a list of electrode labels
        #   batch['metadata']: dictionary containing metadata like the subject identifier and trial id, sampling rate, etc.
        # OUTPUT:
        #   features shape: (batch_size, *) where * can be arbitrary (and will be concatenated for regression)
        
        # Convert data to model's expected dtype and device
        batch['data'] = batch['data'].to(self.model.device, dtype=self.model.dtype, non_blocking=True)
        
        # Apply same preprocessing as in training
        if self.config['model']['signal_preprocessing']['laplacian_rereference']:
            laplacian_rereference_batch(batch, remove_non_laplacian=False, inplace=True)
        
        if self.config['model']['signal_preprocessing']['normalize_voltage']:
            batch['data'] = batch['data'] - torch.mean(batch['data'], dim=[0, 2], keepdim=True)
            batch['data'] = batch['data'] / (torch.std(batch['data'], dim=[0, 2], keepdim=True) + 1)
        
        #get model predictions as features (the learned representations)
        pred, _ = self.model(batch['data'])  # (batch, n_bins-1, bin_size)
        
        #flatten to 2D for downstream evaluation: (batch, (n_bins-1) * bin_size)
        features = pred.reshape(pred.shape[0], -1)
        
        return features  #return the model's learned predictions as features