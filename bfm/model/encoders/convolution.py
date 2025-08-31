import logging

import torch
import torch.nn as nn
from typing import Union, Optional

from bfm.core.logger import get_logger
from bfm.model.base import BFModule
from bfm.model.registry import encoders


logger = get_logger(__name__, level=logging.DEBUG)

@encoders.register("convolution")
class ConvolutionPreprocessor(BFModule):
    """
    Preprocessor that applies a 2D convolution across time axis + linear mapping.
    
    Based on preprocessing steps in BrainWave (https://arxiv.org/abs/2402.10251).

    - Input: [batch_size, n_electrodes, n_timebins, n_freqbins]
    - Output: [batch_size, n_electrodes, n_patches, output_dim]
    
    Args:
        patch_length (int): The length of each patch (timebins).
        n_freqbins (int): The number of frequency bins.
        output_dim (int): The output dimension after the convolution.
        out_channels (int): The number of output channels for the convolution.
        *
        kernel_size (int | float, default=1/4): The kernel size for the time convolution.
        stride (int | float, default=1/8): The stride for the time convolution.
        padding (int | float, default=0): The padding for the time convolution.
        freq_kernel (int | float, default=1): The kernel size for the frequency convolution.
        freq_stride (int | float, default=1): The stride for the frequency convolution.
        freq_padding (int|float|None): Freq padding; None → SAME on freq. Default None.
        use_bn_gelu (bool): Add BatchNorm2d + GELU after conv. Default False.
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
        padding: Union[int, float] = 0,
        freq_kernel: Union[int, float] = 3,
        freq_stride: Union[int, float] = 1,
        freq_padding: Optional[Union[int, float]] = None,
        use_bn_gelu: bool = False
    ):
        super().__init__()
        
        self.patch_length = patch_length
        self.n_freqbins = n_freqbins
        self._trunc_warned = False

        # Time dimensions
        k_t = _normalize_dim_arg(kernel_size, patch_length)
        s_t = _normalize_dim_arg(stride, patch_length)
        p_t = _normalize_dim_arg(padding, patch_length, min_val=0)

        # Freq dims
        k_f = _normalize_dim_arg(freq_kernel, n_freqbins)
        s_f = _normalize_dim_arg(freq_stride, n_freqbins)
        if freq_padding is None:
            p_f = (k_f - 1) // 2  # SAME on freq (approx; exact SAME with stride > 1 is asymmetric)
            logger.debug(f"Using SAME padding on freq: {p_f}")
        else:
            p_f = _normalize_dim_arg(freq_padding, n_freqbins, min_val=0)

        # Conv: disable bias if BN is used
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=(k_t, k_f),
            stride=(s_t, s_f),
            padding=(p_t, p_f),
            bias=not use_bn_gelu,
        )
        
        self.post = nn.Identity()
        if use_bn_gelu:
            self.post = nn.Sequential(nn.BatchNorm2d(out_channels), nn.GELU())
        
        # Output spatial sizes after conv
        def _out_len(L, k, s, p, d=1):
            return (L + 2 * p - d * (k - 1) - 1) // s + 1

        H_out = _out_len(patch_length, k_t, s_t, p_t)   # time
        W_out = _out_len(n_freqbins, k_f, s_f, p_f)     # freq

        if H_out <= 0 or W_out <= 0:
            raise ValueError(f"Bad conv params: H_out={H_out}, W_out={W_out}")
        logger.debug(f"ConvolutionPreprocessor output shape: [B, N, P, {H_out}, {W_out}]")
        
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
        logger.debug(f"ConvolutionPreprocessor input shape: {x.shape}")
        
        if F != self.n_freqbins:
            raise ValueError(
                f"Input n_freqbins {F} does not match expected n_freqbins {self.n_freqbins}"
            ) # Brainwave applied truncation/padding, we assume n_freqbins is constant

        if self.patch_length > T:
            raise ValueError(
                f"patch_length ({self.patch_length}) must be less than or equal to n_timebins ({T})"
            )
        
        if T % self.patch_length != 0:
            x = x.narrow(2, 0, T - (T % self.patch_length))
            if not self._trunc_warned:
                logger.warning(
                    f"n_timebins {T} is not a multiple of patch_length {self.patch_length}. "
                    f"Truncating input tensor to shape {x.shape}..."
                )
                self._trunc_warned = True

        # Split into P patches of size PL (= patch_length)
        x = x.unfold(2, self.patch_length, self.patch_length)  # [B, N, P, PL, F]
        x = x.contiguous()
        P = x.size(2)
        
        # Convolve each patch (nn.Conv2d expects input of shape [batch_size, channels, height, width])
        x = x.reshape(B * N * P, 1, self.patch_length, F)      # [B * N * P, 1, PL, F]
        x = self.post(self.conv(x))                            # [B * N * P, out_channels, H_out, W_out]

        # Flatten and linearly project
        x = x.flatten(1)                                       # [B * N * P, out_channels * H_out * W_out]
        x = self.fc(x)                                         # [B * N * P, output_dim]
        x = x.reshape(B, N, P, -1)                             # [B, N, P, output_dim]
        
        return x


def _normalize_dim_arg(arg: Union[int, float], dim: int, min_val: int = 1) -> int:
    """
    Normalize a convolution argument (e.g., stride, kernel_size, padding).
    
    If 0 < arg < 1, interpret it as a fraction of dim.
    Otherwise, cast to int.

    Args:
        arg (float | int): The argument to normalize. Must be >= 0.
        dim (int): The dimension length.
        min_val (int): Minimum value for the normalized argument.

    Returns:
        int: Normalized integer value.

    Raises:
        ValueError: If arg is negative.
    """
    if arg < 0:
        raise ValueError("Argument must be non-negative.")
    val = round(dim * arg) if 0 < arg < 1 else int(arg)
    return max(min_val, val)