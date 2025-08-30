from typing import Optional
import torch
import torch.nn as nn
from bfm.model.registry import backbones


@backbones.register("roberta", "brainwave")
class RoBERTa(nn.Module):
    """
    RoBERTa implementation.

    - Input: [batch_size, seq_length, in_dim]
    - Output: [batch_size, seq_length, hidden_size]

    Args:
        in_dim (int): Input dimension.
        hidden_size (int): Hidden size.
        n_layers (int): Number of layers.
        n_heads (int): Number of attention heads.
        ffn_dim (int): Feedforward dimension.
        max_len (int): Maximum sequence length.
        dropout (float): Dropout rate.
        add_cls (bool): Whether to add a CLS token.
        ln_eps (float): LayerNorm epsilon.
    """
    def __init__(
        self, 
        in_dim: int = 256, 
        hidden_size: int = 768,
        n_layers: int = 10, 
        n_heads: int = 16, 
        ffn_dim: int = 2048, 
        max_len: int = 4096,
        dropout: float = 0.1, 
        add_cls: bool = True, 
        ln_eps: float = 1e-5
    ):
        super().__init__()
        
        self.proj = nn.Linear(in_dim, hidden_size)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_size)) if add_cls else None
        self.pos = nn.Embedding(max_len + (1 if add_cls else 0), hidden_size)
        
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=n_heads, 
            dim_feedforward=ffn_dim,
            dropout=dropout, 
            activation="gelu", 
            batch_first=True, 
            norm_first=False
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(hidden_size, eps=ln_eps)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):  
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_length, hidden_size]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The output tensor and the pooled output tensor.
        """
        h = self.proj(x)
        B, _, _ = h.shape

        # Add CLS token if specified
        if self.cls is not None:
            h = torch.cat([self.cls.expand(B, 1, -1), h], 1)
            
            if attn_mask is not None:
                one = h.new_ones(B, 1, dtype=attn_mask.dtype)
                attn_mask = torch.cat([one, attn_mask], 1)

        # Add positional encoding
        pos = torch.arange(h.size(1), device=h.device).unsqueeze(0).expand(B, -1)
        h = h + self.pos(pos)

        # Transformer forward pass
        skpm = (attn_mask == 0) if attn_mask is not None else None
        h = self.ln(self.tr(h, src_key_padding_mask=skpm))

        return h, (h[:, 0] if self.cls is not None else h.mean(1))
