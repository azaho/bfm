import warnings

import torch
import torch.nn as nn
from typing import Union

from bfm.model.base import BFModule
from bfm.model.registry import encoders


@encoders.register("convolution")
class ConvolutionPreprocessor(BFModule):
    """
    Preprocessor that applies a 2D convolution across time axis + linear mapping.
    
    Based on preprocessing steps in BrainWave (https://arxiv.org/abs/2402.10251).
    
    Args:
        input_dim (int): The input dimension (timebins) for the convolution.
        output_dim (int): The output dimension after the convolution.
        out_channels (int): The number of output channels for the convolution.
        kernel_size (int | float, default=1/4): The kernel size for the convolution. 
        stride (int | float, default=1/8): The stride for the convolution.
        padding (int | float, default=0): The padding for the convolution.
    """
    def __init__(
        self, 
        patch_length: int,
        n_freqbins: int,
        output_dim: int,
        out_channels: int, 
        *,
        kernel_size: Union[int, float] = 1 / 4,
        stride: Union[int, float] = 1 / 8,
        padding: Union[int, float] = 0
    ):
        super().__init__()
        
        self.patch_length = patch_length
        self.n_freqbins = n_freqbins

        k = _normalize_dim_arg(kernel_size, patch_length)
        s = _normalize_dim_arg(stride, patch_length)
        p = _normalize_dim_arg(padding, patch_length)

        self.conv = nn.Conv2d(1, out_channels, (k, 1), (s, 1), (p, 0))

        H_out = (patch_length + 2 * p - (k - 1) - 1) // s + 1
        W_out = n_freqbins  
    
        self.output_dim = output_dim
        self.fc = nn.Linear(out_channels * H_out * W_out, output_dim)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, n_electrodes, n_timebins, n_freqbins]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, n_electrodes, n_patches, output_dim]
            
        Raises:
            ValueError: If the input tensor shape is invalid.
        """
        B, N, T, F = x.shape
        
        if F != self.n_freqbins:
            raise ValueError(
                f"Input n_freqbins {F} does not match expected n_freqbins {self.n_freqbins}"
            )

        if self.patch_length > T:
            raise ValueError(
                f"patch_length ({self.patch_length}) must be less than or equal to n_timebins ({T})"
            )
        
        if T % self.patch_length != 0:
            x = x.narrow(2, 0, T - (T % self.patch_length))
            warnings.warn(
                f"n_timebins {T} is not a multiple of patch_length {self.patch_length}. "
                f"Truncating input tensor to shape {x.shape}..."
            )

        # Split into patches of size PL (= patch_length)
        P = T // self.patch_length # Number of patches
        x = x.unfold(2, self.patch_length, self.patch_length)  # [B, N, P, PL, F]
        x = x.contiguous()

        # Convolve each patch (nn.Conv2d expects input of shape [batch_size, channels, height, width])
        x = x.view(B * N * P, 1, self.patch_length, F)         # [B * N * P, 1, PL, F]
        x = self.conv(x)                                       # [B * N * P, out_channels, H_out, W_out]

        # Flatten and linearly project
        x = x.flatten(1)                                       # [B * N * P, out_channels * H_out * W_out]
        x = x.view(B, N, P, self.output_dim)                   # [B, N, P, output_dim]
        
        return x


def _normalize_dim_arg(arg: Union[int, float], dim: int) -> int:
    """
    Normalize a convolution argument (e.g., stride, kernel_size, padding).
    
    If arg < 1, interpret it as a fraction of dim.
    Otherwise, cast to int.

    Args:
        arg (float | int): The argument to normalize. Must be >= 0.
        dim (int): The dimension length.

    Returns:
        int: Normalized integer value.

    Raises:
        ValueError: If arg is negative.
    """
    if arg < 0:
        raise ValueError("Argument must be non-negative.")
    if arg < 1:
        return max(1, round(dim * arg))
    return int(arg)