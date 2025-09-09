'''Implementation of BrainWave (https://arxiv.org/abs/2402.10251).'''
from typing import Any, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from bfm.model.base import BFModule
from bfm.model.factory import build_model
from bfm.training.setup_registry import setups
from bfm.training.training_setup import TrainingSetup

CONFIG = {
    "in_channels": 2048,  # Get this dynamically
    "components": {
        "spectrogram": {
            "registry": "preprocessors",
            "name": "spectrogram",
            "kwargs": {
                "output_dim": "${transformer.d_model}",
                "spectrogram_parameters": "${signal_preprocessing.spectrogram_parameters}"
            }
        },
        "encoder": {
            "registry": "encoders",
            "name": "convolution",
            "kwargs": {
                "patch_length": 15, # 60 in original paper, but n_timebins is 49 so needs to be less than that
                "n_freqbins": "${transformer.d_model}",
                "output_dim":  "${transformer.d_model}",
                "out_channels": 8,
                "kernel_size": 1 / 4,
                "stride": 1 / 8,
                "padding": 0
            }
        },
        "backbone": {
            "registry": "backbones",
            "name": "brainwave",
            "kwargs": {
                "in_dim": "${transformer.d_model}",
                "hidden_size": "${transformer.d_model}", # paper uses 768
                "n_layers": "${transformer.n_layers}", # paper uses 10
                "n_heads": "${transformer.n_heads}", # paper uses 16
                "ffn_dim": 256, # paper uses 2048
                "max_len": 16,  # 15 signal patches + [CLS]
                "add_cls": False # Important
            },
        },
        "channel_attention": "${components.backbone}"
    },
}


def span_mask(B, T, mask_ratio: float = 0.3, span_len: int = 16, device=None):
    """Create random contiguous masks."""
    device = device or 'cpu'
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    num_mask = int(T * mask_ratio)
    n_spans  = max(1, num_mask // span_len)
    for b in range(B):
        starts = torch.randint(0, max(1, T - span_len + 1), (n_spans,), device=device)
        for s in starts:
            mask[b, s:s+span_len] = True
    return mask


class BrainWaveBackbone(BFModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        build_model(self, cfg, components=[
            "spectrogram", "encoder", "backbone", "channel_attention"
        ])
        self.decoder = nn.Linear(
            cfg.transformer.d_model,
            cfg.in_channels
        )


    def forward(self, batch: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch (dict): Input batch containing:
                - 'data' (torch.Tensor): Input tensor of shape [batch_size, n_electrodes, n_timesamples]
        
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, n_electrodes, n_patches, output_dim]
        """        
        spec = self.spectrogram(batch)                   # [B, N, T, F]
               
        emb = self.encoder(spec)                        # [B, N, P, D]
        B, N, P, D = emb.shape

        tok = emb.reshape(B * N, P, D)                   # merge batch & channel
        tok, _ = self.backbone(tok)                      # -> [B * N, P, D2]

        out, pooled = self.channel_attention(tok)        # out: [B * N, P, D2], pooled: [B * N, D2]
        out = out.reshape(B, N, P, -1)                   # [B, N, P, D2]
        
        return out, pooled


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
        merged = {**self.config["model"], **CONFIG}
        # Remove dtype keys to avoid conflicts for now
        merged.pop("dtype", None)  
        merged.pop("amp_dtype", None)
        cfg = OmegaConf.create(merged)
        self.model = BrainWaveBackbone(cfg)
        
        self.model.to(self.config['device'], dtype=self.config['model']['dtype'])
        self.model_components['model'] = self.model
        
    
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
        batch['data'] = batch['data'].to(
            self.model.device, 
            self.model.dtype,
            non_blocking=True
        )
        
        raw = batch["data"] # [B, N, T]
        B, N, T = raw.shape

        mask = self._make_mask(raw, mask_ratio=0.2, span_len=3)  # [B, T]
        mask = mask.unsqueeze(1)                                 # [B, 1, T]  (broadcast over N)
        
        # Partially observed input
        masked = raw.clone()
        masked.masked_fill_(mask, 0.0)
        
        temp_batch = batch.copy()
        temp_batch['data'] = masked

        tok, _ = self.model.forward(temp_batch)   # [B, N, P, D]
        
        # predict per-patch waveforms, then fold back to time axis
        recon_patches = self.model.decoder(tok)   # [B, N, P, patch_len]
        recon = recon_patches.reshape(B, N, T)    # [B, N, T]
        
        mask = mask.to(self.model.device)
        raw = raw.to(self.model.device)

        # L1 reconstruction loss
        l1 = (recon - raw).abs()                  # [B, N, T]
        # loss_all = l1.mean(dim=(1, 2))          # per-sample 
        
        # masked-only reconstruction
        denom = mask.squeeze(1).sum(dim=1).clamp_min(1).float().unsqueeze(1)  # [B, 1]
        masked_l1 = (l1 * mask.float()).sum(dim=2) / denom                    # [B, N]
        loss_masked = masked_l1.mean(dim=1)                                   # [B]

        return {
            # "loss_all": loss_all.mean(),
            "loss_masked": loss_masked.mean(),
        }


    def generate_frozen_features(self, batch: dict):
        """Generate frozen features for the given batch."""
        with torch.no_grad():
            _, frozen_features = self.model(batch)
        return frozen_features