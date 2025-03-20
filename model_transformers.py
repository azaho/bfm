#### credit to modded-nanogpt for the transformer blocks (https://github.com/KellerJordan/modded-nanogpt/
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_model import BFModule
from muon import orthogonalize

# XXX remove hard coded data type

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, causal, rope):
        super().__init__()
        self.n_head = n_head
        self.n_embd = d_model
        self.causal = causal
        self.rope = rope
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)

        self.orthogonalize()  ### XXX - orthogonal init
    
    def orthogonalize(self):
        self.c_q.weight.data = orthogonalize(self.c_q.weight.data)
        self.c_k.weight.data = orthogonalize(self.c_k.weight.data)
        self.c_v.weight.data = orthogonalize(self.c_v.weight.data)
        self.c_proj.weight.data = orthogonalize(self.c_proj.weight.data)  ### XXX - NOT using modded-nanogpt's zero-init for proj, for now
        # self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        if self.rope:
            cos, sin = self.rotary(q)
        #q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm suggested by @Grad62304977 # XXX -- remove if too much memory consumption
        if self.rope:
            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=self.causal, scale=1/q.shape[-1]) # XXX usually does **0.5 but this is the Modula way! (https://docs.modula.systems/faq/ --- alignment)
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.c_fc    = nn.Linear(d_model, 4 * d_model, bias=False)
        self.c_proj  = nn.Linear(4 * d_model, d_model, bias=False)
        self.orthogonalize()  ### XXX - orthogonal init

    def orthogonalize(self):
        self.c_fc.weight.data = orthogonalize(self.c_fc.weight.data)
        self.c_proj.weight.data = orthogonalize(self.c_proj.weight.data)  ### XXX - NOT using modded-nanogpt's zero-init for proj, for now
        # self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, n_layer, d_model, n_head, causal, rope):
        super().__init__()
        self.n_layer = n_layer
        self.attn = CausalSelfAttention(d_model, n_head, causal, rope)
        self.mlp = MLP(d_model)

    def forward(self, x):
        L = self.n_layer
        x = (2*L-1)/(2*L) * x + (1/(2*L)) * self.attn(F.rms_norm(x, (x.size(-1),)))  ### XXX - more typical thing is just x + 1/sqrt(2*L) * attn, then same for mlp
        x = (2*L-1)/(2*L) * x + (1/(2*L)) * self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x

class Transformer(BFModule):
    def __init__(self, d_input=64, d_model=192, d_output=192, n_layer=10, n_head=12, causal=True, rope=True, cls_token=True):
        super().__init__()
        self.d_input = d_input
        self.n_layer = n_layer
        self.n_head = n_head 
        self.d_model = d_model  
        self.d_output = d_output
        self.causal = causal
        self.rope = rope

        self.cls_token = nn.Parameter(torch.randn(d_model)) if cls_token else None

        self.embed = nn.Linear(d_input, d_model, bias=False) # XXX --- removed bias; if add bias back, change rms norm to layernorm in block
        self.blocks = nn.ModuleList([Block(n_layer, d_model, n_head, causal, rope) for _ in range(n_layer)])
        self.output_proj = nn.Linear(d_model, d_output, bias=False)

        self.orthogonalize()  ### XXX - orthogonal init
    
    def orthogonalize(self):
        self.embed.weight.data = orthogonalize(self.embed.weight.data)
        self.output_proj.weight.data = orthogonalize(self.output_proj.weight.data)

    def forward(self, x):
        # x is of shape (batch_size, seq_len, d_input)

        batch_size, seq_len, d_input = x.shape

        x = self.embed(x)  # shape: (batch_size, seq_len, d_model)

        if self.cls_token is not None:
            # Expand cls_token to match timebins dimension
            cls_token = self.cls_token.expand(batch_size, 1, -1)
            x = torch.cat([x, cls_token], dim=1)  # shape: (batch_size, seq_len + 1, d_model)

        for block in self.blocks:
            x = block(x)
        
        x = self.output_proj(x) # shape: (batch_size, seq_len (+1), d_output)

        return x