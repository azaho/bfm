#### credit to modded-nanogpt for the transformer blocks (https://github.com/KellerJordan/modded-nanogpt/
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.muon_optimizer import orthogonalize
from model.BFModule import BFModule
from model.transformer_implementation import apply_rotary_emb, CausalSelfAttention, Block, Transformer, Rotary

# Custom CausalSelfAttention that returns attention weights
class CausalSelfAttentionWithReturn(CausalSelfAttention):
    def forward(self, x, attention_mask=None, positions=None):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        if self.rope:
            cos, sin = self.rotary(q, positions)
            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(1) == 0, float('-inf'))
        if self.causal and attention_mask is None:
            causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        y = torch.matmul(attention_weights, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y, attention_weights

# Custom Block that returns attention weights
class BlockWithReturn(Block):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn = CausalSelfAttentionWithReturn(self.attn.n_embd, self.attn.n_head, self.attn.causal, self.attn.rope, self.attn.rope_base, self.attn.dropout.p)
        self.attn.load_state_dict(self.attn.state_dict())
    def forward(self, x, attention_mask=None, positions=None):
        L = self.n_layer
        x_norm = torch.nn.functional.rms_norm(x, (x.size(-1),))
        attn_out, attn_weights = self.attn(x_norm, attention_mask=attention_mask, positions=positions)
        x = (2*L-1)/(2*L) * x + (1/(2*L)) * attn_out
        x_norm2 = torch.nn.functional.rms_norm(x, (x.size(-1),))
        x = (2*L-1)/(2*L) * x + (1/(2*L)) * self.mlp(x_norm2)
        return x, attn_weights
# Custom Transformer that returns all attention weights
class TransformerWithReturn(Transformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blocks = torch.nn.ModuleList([BlockWithReturn(self.n_layer, self.d_model, self.n_head, self.causal, self.rope, self.rope_base, self.dropout.p) for _ in range(self.n_layer)])
        self.embed = self.embed
        self.output_proj = self.output_proj
        self.dropout = self.dropout
    def forward(self, x, attention_mask=None, positions=None, embeddings=None, strict_positions=False, stop_at_block=None):
        batch_size, seq_len, d_input = x.shape
        x = self.embed(x)
        x = self.dropout(x)
        if embeddings is not None:
            x = x + embeddings
        if attention_mask is None and positions is not None:
            if strict_positions:
                attention_mask = positions.unsqueeze(2) == positions.unsqueeze(1)
            else:
                attention_mask = positions.unsqueeze(2) >= positions.unsqueeze(1)
        all_attn_weights = []
        for block_i, block in enumerate(self.blocks):
            x, attn_weights = block(x, attention_mask=attention_mask, positions=positions)
            all_attn_weights.append(attn_weights)
            if stop_at_block is not None and block_i+1 == stop_at_block:
                return x, all_attn_weights
        x = self.output_proj(x)
        return x, all_attn_weights