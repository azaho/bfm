'''Implementation of BrainWave (https://arxiv.org/abs/2402.10251).'''
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from bfm.model.factory import build_model
from bfm.model.registry import backbones
from bfm.training.setup_registry import setups
from bfm.training.training_setup import TrainingSetup


CONFIG = {
    'device': 'cuda',
    'in_channels': 8,  # TODO: Get this
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


def span_mask(B, T, mask_ratio: float = 0.3, span_len: int =16, device=None):
    """Create random contiguous masks."""
    device = device or "cpu"
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    num_mask = int(T * mask_ratio)
    n_spans  = max(1, num_mask // span_len)
    for b in range(B):
        starts = torch.randint(0, max(1, T - span_len + 1), (n_spans,), device=device)
        for s in starts:
            mask[b, s:s+span_len] = True
    return mask


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
        self.model_components["decoder"] = nn.Linear(
            CONFIG["backbone"]["kwargs"]["hidden_size"],
            CONFIG["in_channels"]
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
    
    
    @torch.no_grad()
    def _make_mask(self, x, mask_ratio=0.3, span_len=16):
        B, _, T = x.shape
        return span_mask(B, T, mask_ratio, span_len, x.device)  # (B, T), True=mask
    
    
    def calculate_pretrain_loss(self, batch: dict[str, Any], output_accuracy: bool = False) -> dict[str, torch.Tensor]:
        """
        Calculate reconstruction loss.

        Args:
            batch (dict): Dictionary containing:
                - 'data' (Tensor): Shape (batch_size, n_electrodes, n_timesamples).
                - 'electrode_index' (Tensor): Shape (batch_size, n_electrodes).
                - 'metadata' (dict): Contains subject identifier, trial ID, sampling rate, etc.
            
        Returns:
            dict: Dictionary containing containing losses and their values.
                The final loss is the mean of all the losses.
                Accuracies are exempt and are just used for logging.
        """
        raw = batch["data"]             # [B, N, T]
        B, N, T = raw.shape

        mask = self._make_mask(raw)     # (B, T)
        
        # Partially observed input
        masked = raw.clone()
        masked[..., mask] = 0.0

        tok = self.forward(masked)                    # [B, N, P, D]
        decoder = self.model_components["decoder"]
        
        # predict per-patch waveforms, then fold back to time axis
        recon_patches = decoder(tok)              # [B, N, P, patch_len]
        recon = recon_patches.reshape(B, N, T)    # [B, N, T]
        
        # L1 reconstruction loss
        l1 = (recon - raw).abs()                  # [B, N, T]
        loss_all = l1.mean(dim=(1, 2))            # per-sample 
        
        # masked-only for logging/comparison
        denom = mask.sum(dim=1).clamp_min(1).float().unsqueeze(1)        # [B,1]
        masked_l1 = (l1 * mask.unsqueeze(1).float()).sum(dim=2) / denom  # [B, N]
        loss_masked = masked_l1.mean(dim=1)                              # [B]

        return {
            "loss_all": loss_all.mean(),
            "loss_masked": loss_masked.mean(),
        }


    def generate_frozen_features(self, batch: dict):
        """Generate frozen features for the given batch."""
        with torch.no_grad():
            x = batch["data"]
            frozen_features = self.forward(x)
        return frozen_features.detach()