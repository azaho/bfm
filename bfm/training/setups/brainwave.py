'''Implementation of BrainWave (https://arxiv.org/abs/2402.10251).'''
from typing import Any
import logging

import torch
import torch.nn as nn

from bfm.core.logger import get_logger
from bfm.model.factory import build_model
from bfm.model.registry import backbones
from bfm.model.preprocessing.spectrogram import SpectrogramPreprocessor
from bfm.training.setup_registry import setups
from bfm.training.training_setup import TrainingSetup


from dataclasses import dataclass, field, asdict

@dataclass
class ModelConfig:
    dtype: torch.dtype = torch.bfloat16
@dataclass
class EncoderKwargs:
    patch_length: int = 15  # 60 in original paper, but n_timebins is 49 so needs to be less than that
    n_freqbins: int = 38  # Read dynamically
    output_dim: int = 768
    out_channels: int = 8
    kernel_size: float = 1 / 4
    stride: float = 1 / 8
    padding: int = 0
@dataclass
class EncoderConfig:
    name: str = "convolution"
    kwargs: dict = field(default_factory=lambda: asdict(EncoderKwargs()))
@dataclass
class BackboneKwargs:
    in_dim: int = 768
    hidden_size: int = 768
    n_layers: int = 10
    n_heads: int = 16
    ffn_dim: int = 2048
    max_len: int = 16  # 15 signal patches + [CLS]
    add_cls: bool = False # Important
@dataclass
class BackboneConfig:
    name: str = "brainwave"
    kwargs: dict = field(default_factory=lambda: asdict(BackboneKwargs()))
@dataclass
class Config:
    device: str = "cuda"
    in_channels: int = 2048  # Get this dynamically
    model: ModelConfig = field(default_factory=ModelConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)


def span_mask(B, T, mask_ratio: float = 0.3, span_len: int = 16, device=None):
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


logger = get_logger(__name__)

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
        CONFIG = Config()
        encoder, backbone = build_model(CONFIG)
        self.model_components["spectogram"] = SpectrogramPreprocessor(
            self.config['model']['signal_preprocessing']['spectrogram_parameters'],
        ).to(CONFIG.device, dtype=CONFIG.model.dtype)
        self.model_components["encoder"] = encoder
        self.model_components["backbone"] = backbone
        self.model_components["channel_attention"] = backbones.resolve(
            "brainwave",
            **CONFIG.backbone.kwargs
        ).to(CONFIG.device, dtype=CONFIG.model.dtype)
        self.model_components["decoder"] = nn.Linear(
            CONFIG.backbone.kwargs['hidden_size'],
            CONFIG.in_channels
        ).to(CONFIG.device, dtype=CONFIG.model.dtype)
        
        self.model = self.model_components["backbone"]


    def forward(self, batch: dict) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, n_electrodes, n_timesamples]
        
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, n_electrodes, n_patches, output_dim]
        """
        batch['data'] = batch['data'].to("cuda", non_blocking=True)
        x = batch['data']
        logger.debug(f"BrainWave forward input shape: {x.shape}")
        spec = self.model_components["spectogram"](batch) # [B, N, T, F]
        logger.debug(f"BrainWave forward spectrogram output shape: {spec.shape}") 
               
        emb = self.model_components["encoder"](spec)   # [B, N, P, D]
        logger.debug(f"BrainWave forward encoder output shape: {emb.shape}")
        B, N, P, D = emb.shape

        tok = emb.reshape(B * N, P, D)                   # merge batch & channel
        tok, _ = self.model_components["backbone"](tok)  # -> [B * N, P, D2]

        logger.debug(f"BrainWave forward backbone output shape: {tok.shape}")
        out, pooled = self.model_components["channel_attention"](tok)
        logger.debug(f"BrainWave forward channel_attention output shape: {out.shape}")
        out = out.reshape(B, N, P, -1)                   # [B, N, P, D2]
        return out, pooled
    
    
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
        logger.debug(f"BrainWave calculate_pretrain_loss input shape: {raw.shape}")

        mask = self._make_mask(raw)     # [B, T]
        mask = mask.unsqueeze(1)        # [B, 1, T]  (broadcast over N)

        
        # Partially observed input
        masked = raw.clone()
        masked.masked_fill_(mask, 0.0)
        
        temp_batch = batch.copy()
        temp_batch['data'] = masked

        tok, _ = self.forward(temp_batch)               # [B, N, P, D]
        decoder = self.model_components["decoder"]
        
        # predict per-patch waveforms, then fold back to time axis
        recon_patches = decoder(tok)              # [B, N, P, patch_len]
        recon = recon_patches.reshape(B, N, T)    # [B, N, T]
        
        mask = mask.to(recon.device)
        raw = raw.to(recon.device)

        # L1 reconstruction loss
        l1 = (recon - raw).abs()                  # [B, N, T]
        # loss_all = l1.mean(dim=(1, 2))           # per-sample 
        
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
            _, frozen_features = self.forward(batch)
        return frozen_features.detach()