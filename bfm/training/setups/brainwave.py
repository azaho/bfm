'''
Implementation of BrainWave (https://arxiv.org/abs/2402.10251)
'''
from typing import Dict, Any

import torch

from training.setup_registry import setups
from training.training_setup import TrainingSetup
from model.encoders.convolution import ConvolutionPreprocessor

@setups.register("brainwave")
class BrainWave(TrainingSetup):
    """
    BrainWave model trainer. 
    
    Args:
        all_subjects: List of all subjects in the dataset.
        config: Configuration dictionary for the training process.
        verbose (bool): Boolean flag for verbose output.
    """
    def __init__(self, all_subjects, config, verbose=True):
        super().__init__(all_subjects, config, verbose)
    
    
    def initialize_model(self):
        self.conv = ConvolutionPreprocessor(
            patch_length=self.config['model']['patch_length'],
            n_freqbins=self.config['model']['n_freqbins'],
            output_dim=self.config['model']['signal_preprocessing']['convolution_output_dim'],
            out_channels=self.config['model']['signal_preprocessing']['convolution_out_channels']            
        )
        
    
    def forward(self, x: torch.Tensor):
        e = self.conv(x) # embeddings of shape [batch_size, n_electrodes, n_patches, output_dim]
        return e

    def calculate_pretrain_loss(self, batch: Dict[str, Any], output_accuracy: bool = False) -> Dict[str, torch.Tensor]:
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
        return {}
    
    
    def generate_frozen_features(self, batch: Dict):
        raise NotImplementedError("This function is not (yet) implemented for this training setup.")
