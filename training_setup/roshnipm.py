'''
Instructions:
Try to implement a simple linear model (you will only need to create a single file in the training_setup/ folder with the new 
training setup). The linear model will take the data, average over all of the electrodes, split into bins of your predetermined 
size, and then use the linear regression to predict the future bin from the past bin using the L2 loss.

For evaluation (generate_frozen_features), feel free to come up with your own scheme for what the "features" of the model will be
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from training_setup.training_setup import TrainingSetup
from model.preprocessing.laplacian_rereferencing import laplacian_rereference_batch

class LinearModel(nn.Module):
    def __init__(self, bin_size, d_bin):
        super().__init__()
        self.bin_size = bin_size
        # Linear layer: input all timestamps in current bin, output all timestamps in next bin
        self.linear = nn.Linear(bin_size, bin_size)

    def forward(self, x):
        # x: (batch_size, n_electrodes, n_timesamples)
        # Average over all electrodes
        x = x.mean(dim=1)  # (batch_size, n_timesamples)
        
        # Bin the time dimension
        n_time = x.shape[1]
        n_bins = n_time // self.bin_size
        x = x[:, :n_bins * self.bin_size]  # remove remainder to fit bins
        x = x.reshape(x.shape[0], n_bins, self.bin_size)  # (batch, n_bins, bin_size)
        
        # Predict next bin from previous bin
        # Input: x[:, :-1, :] (all timestamps in current bins), Target: x[:, 1:, :] (all timestamps in next bins)
        # Note: Each bin is processed independently by the linear layer - no cross-bin contamination
        # However, all bins are processed in the same forward pass, so there could be batch-level effects
        pred = self.linear(x[:, :-1, :])  # (batch, n_bins-1, bin_size)
        return pred, x[:, 1:, :]  # returns (prediction, target)

class roshnipm(TrainingSetup):
    def __init__(self, all_subjects, config, verbose=True):
        super().__init__(all_subjects, config, verbose)
        # 10 is the default bin size
        self.bin_size = config.get('bin_size', 10)
        self.d_bin = self.bin_size  # each bin contains bin_size timestamps
        self.model = LinearModel(self.bin_size, self.d_bin)
        self.model_components = {"model": self.model}
        self.electrode_embeddings = None  # simple linear model doesn't need electrode embeddings

    def initialize_model(self):
        self.model = LinearModel(self.bin_size, self.d_bin)
        # move model to device and set dtype, like andrii0 does
        self.model = self.model.to(self.config['device'], dtype=self.config['model']['dtype'])
        self.model_components = {"model": self.model}

    def get_preprocess_functions(self, pretraining=False):
        """
        Get the preprocess functions for the training setup.
        Default: only subset electrodes to the maximum number during pretraining (during eval, pass all electrodes).

        This must be an array of functions, each takes just batch as input and returns the (modified) batch. Modifying it in place is fine (but still return the batch).
        """
        preprocess_functions = []
        if self.config['model']['signal_preprocessing']['laplacian_rereference']:
            preprocess_functions.append(self._preprocess_laplacian_rereference)
        if self.config['model']['signal_preprocessing']['normalize_voltage']:
            preprocess_functions.append(self._preprocess_normalize_voltage)
        if pretraining:
            preprocess_functions.append(self._preprocess_subset_electrodes)
        return preprocess_functions

    def calculate_pretrain_loss(self, batch, output_accuracy=True):
        # batch['data']: (batch, n_electrodes, n_timesamples)
        # batch['electrode_index']: (batch, n_electrodes) 
        # batch['metadata']: dictionary containing metadata
        batch['data'] = batch['data'].to(self.config['device'], dtype=self.config['model']['dtype'], non_blocking=True)
        
        # Pass through model
        pred, target = self.model(batch['data'])
        
        # Compute MSE loss
        loss = F.mse_loss(pred, target)
        return {"mse_loss": loss}

    def generate_frozen_features(self, batch):
        # Convert data to model's expected dtype and device
        batch['data'] = batch['data'].to(self.config['device'], dtype=self.config['model']['dtype'], non_blocking=True)
        
        # Get model predictions as features (the learned representations)
        # difference between this and calculate_pretrain_loss is that this doesn't 
        # use the target for loss calculation
        pred, _ = self.model(batch['data'])  # (batch, n_bins-1, bin_size)
        
        # Flatten to 2D for downstream evaluation: (batch, (n_bins-1) * bin_size)
        features = pred.reshape(pred.shape[0], -1)
        
        return features  # Return the model's learned predictions as features

    # edited in pretrain.py to ignore electrode_index if unnecessary
    def load_dataloaders(self):
        super().load_dataloaders()

    def model_parameters(self, verbose=False):
        return self.model.parameters()

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()

    # edited in training_setup.py to ignore electrode_index if unnecessary
    def calculate_pretrain_test_loss(self):
        return super().calculate_pretrain_test_loss()

    def save_model(self, epoch, eval_results={}, save_in_dir="runs/data/", training_statistics_store=None):
        return super().save_model(epoch, eval_results, save_in_dir, training_statistics_store)