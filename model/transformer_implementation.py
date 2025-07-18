#### credit to modded-nanogpt for the transformer blocks (https://github.com/KellerJordan/modded-nanogpt/
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.muon_optimizer import orthogonalize
from model.BFModule import BFModule

# XXX remove hard coded data type

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=100):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    def forward(self, x, positions=None):
        seq_len = x.shape[1]
        if positions is None:
            positions = torch.arange(seq_len, device=x.device)
        
        # Calculate frequencies directly without caching
        # positions shape: (batch_size, seq_len)
        freqs = torch.outer(positions.flatten(), self.inv_freq.to(x.device)).to(x.device)
        cos = freqs.cos().to(x.dtype)
        sin = freqs.sin().to(x.dtype)
        
        # Reshape to match batch dimension if needed
        if positions.ndim > 1:
            cos = cos.view(*positions.shape, -1)
            sin = sin.view(*positions.shape, -1)
        
        return cos.unsqueeze(-2), sin.unsqueeze(-2)

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, causal, rope, rope_base, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.n_embd = d_model
        self.causal = causal
        self.rope = rope
        self.rope_base = rope_base
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim, base=self.rope_base)
        self.dropout = nn.Dropout(dropout)

        #self.zero_init()  # Initialize weights to zero
        self.orthogonalize() # Orthogonal init
        
    def orthogonalize(self):
        self.c_q.weight.data = orthogonalize(self.c_q.weight.data)
        self.c_k.weight.data = orthogonalize(self.c_k.weight.data)
        self.c_v.weight.data = orthogonalize(self.c_v.weight.data)
        self.c_proj.weight.data = orthogonalize(self.c_proj.weight.data)  ### XXX - NOT using modded-nanogpt's zero-init for proj, for now
        # self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
    
    def zero_init(self):
        self.c_q.weight.data.zero_()
        self.c_k.weight.data.zero_()
        self.c_v.weight.data.zero_()
        self.c_proj.weight.data.zero_()

    def forward(self, x, attention_mask=None, positions=None):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        if self.rope:
            cos, sin = self.rotary(q, positions)
            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        # Modified attention call to include the mask
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), 
            k.transpose(1, 2), 
            v.transpose(1, 2), 
            attn_mask=attention_mask.unsqueeze(1) if attention_mask is not None else None, # the head dimension is broadcasted
            is_causal=self.causal if attention_mask is None else False,  # Only use causal if no custom mask
            scale=1/q.shape[-1]
        )
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.dropout(y)  # Add dropout after attention
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.c_fc    = nn.Linear(d_model, 4 * d_model, bias=False)
        self.c_proj  = nn.Linear(4 * d_model, d_model, bias=False)
        #self.zero_init()  # Initialize weights to zero
        self.orthogonalize() # Orthogonal init
        self.dropout = nn.Dropout(dropout)

    def orthogonalize(self):
        self.c_fc.weight.data = orthogonalize(self.c_fc.weight.data)
        self.c_proj.weight.data = orthogonalize(self.c_proj.weight.data)  ### XXX - NOT using modded-nanogpt's zero-init for proj, for now
        # self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def zero_init(self):
        self.c_fc.weight.data.zero_()
        self.c_proj.weight.data.zero_()

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.dropout(x)  # Add dropout after activation
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, n_layer, d_model, n_head, causal, rope, rope_base, dropout=0.1):
        super().__init__()
        self.n_layer = n_layer
        self.attn = CausalSelfAttention(d_model, n_head, causal, rope, rope_base, dropout)
        self.mlp = MLP(d_model, dropout)

    def forward(self, x, attention_mask=None, positions=None):
        L = self.n_layer
        x = (2*L-1)/(2*L) * x + (1/(2*L)) * self.attn(F.rms_norm(x, (x.size(-1),)), attention_mask=attention_mask, positions=positions)
        x = (2*L-1)/(2*L) * x + (1/(2*L)) * self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x

class Transformer(BFModule):
    def __init__(self, d_input=64, d_model=192, d_output=192, n_layer=10, n_head=12, causal=True, rope=True, rope_base=1024, dropout=0.1):
        super().__init__()

        self.d_input = d_input
        self.n_layer = n_layer
        self.n_head = n_head 
        self.d_model = d_model  
        self.d_output = d_output
        self.causal = causal
        self.rope = rope
        self.rope_base = rope_base
        self.dropout = nn.Dropout(dropout)

        self.embed = nn.Linear(d_input, d_model, bias=False) if self.d_input > 0 else nn.Identity()
        self.blocks = nn.ModuleList([Block(n_layer, d_model, n_head, causal, rope, rope_base, dropout) for _ in range(n_layer)])
        self.output_proj = nn.Linear(d_model, d_output, bias=False)

        #self.zero_init()  # Initialize weights to zero
        self.orthogonalize() # Orthogonal init
    
    def zero_init(self):
        self.embed.weight.data.zero_()
        self.output_proj.weight.data.zero_()
    
    def orthogonalize(self):
        self.embed.weight.data = orthogonalize(self.embed.weight.data)
        self.output_proj.weight.data = orthogonalize(self.output_proj.weight.data)

    def forward(self, x, attention_mask=None, positions=None, embeddings=None, strict_positions=False, stop_at_block=None):
        # x is of shape (batch_size, seq_len, d_input)
        batch_size, seq_len, d_input = x.shape

        x = self.embed(x)
        x = self.dropout(x)  # Add dropout after embedding

        if embeddings is not None:
            x = x + embeddings


        if attention_mask is None and positions is not None: # XXX this overwrites the "Causal" parameter
            if strict_positions:
                attention_mask = positions.unsqueeze(2) == positions.unsqueeze(1) # Causal mask given the positions
            else:
                attention_mask = positions.unsqueeze(2) >= positions.unsqueeze(1) # Causal mask given the positions

        if stop_at_block < 0: stop_at_block = len(self.blocks) + stop_at_block + 1 # allow for negative indices to count from the end
        for block_i, block in enumerate(self.blocks):
            x = block(x, attention_mask=attention_mask, positions=positions)
            if stop_at_block is not None and block_i+1 == stop_at_block:
                return x
        
        x = self.output_proj(x) # shape: (batch_size, seq_len (+1), d_output)

        return x