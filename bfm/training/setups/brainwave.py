'''Implementation of BrainWave (https://arxiv.org/abs/2402.10251)'''
from typing import Dict, Any
import torch

from bfm.training.setup_registry import setups
from bfm.training.training_setup import TrainingSetup
from bfm.model.factory import build_model
from bfm.model.registry import backbones


CONFIG = {
    'device': 'cuda',
    'model': {
        'dtype': 'float32',
    },
    'encoder': {
        'name': 'convolution',
        'kwargs': {
            'patch_length': 60,
            'n_freqbins': 64,  # TODO: Get this
            'output_dim': 768,
            'out_channels': 8,
            'kernel_size': 1 / 4,
            'stride': 1 / 8,
            'padding': 0,
        }
    },
    'backbone': {
        'name': 'brainwave',
        'kwargs': {
            'hidden_size': 768,
            'n_layer': 10,
            'n_heads': 16,
            'ffn_dim': 2048,
            'max_len': 61,  # 60 signal patches + [CLS]
        },
    }
}


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
        encoder, backbone = build_model(CONFIG)
        self.model_components["encoder"] = encoder
        self.model_components["backbone"] = backbone
        self.model_components["channel_attention"] = backbones.resolve("brainwave")(
            **CONFIG["backbone"]["kwargs"]
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, n_electrodes, n_timesamples]
        
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, n_electrodes, n_patches, output_dim]
        """
        emb = self.model_components["encoder"](x)      # [B, N, P, D]
        B, N, P, D = emb.shape

        tok = emb.reshape(B * N, P, D)                 # merge batch & channel
        tok = self.model_components["backbone"](tok)   # -> [B * N, P, D2]
        tok = tok.reshape(B, N, P, -1)                 # [B, N, P, D2]

        out = self.model_components["channel_attention"](tok)
        return out
    
    
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
        """Generate frozen features for the given batch."""
        with torch.no_grad():
            x = batch["data"]
            frozen_features = self.forward(x)
        return frozen_features.detach()