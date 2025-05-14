#### credit to modded-nanogpt for the transformer blocks (https://github.com/KellerJordan/modded-nanogpt/
import torch
import torch.nn as nn
import torch.nn.functional as F
from muon import orthogonalize


class BFModule(nn.Module):
    """
    This module is a base class for all modules that need to be compatible with this project.
    It ensures that the module stores its current device and dtype.
    """
    def __init__(self):
        super().__init__()
        self._device = None
        self._dtype = None
    def to(self, *args, **kwargs):
        output = super().to(*args, **kwargs)
        # Extract device and dtype from args/kwargs
        device = next((torch.device(arg) for arg in args if isinstance(arg, (torch.device, str))), 
                     kwargs.get('device', None))
        dtype = next((arg for arg in args if isinstance(arg, torch.dtype)),
                    kwargs.get('dtype', None))
        if device is not None: self._device = device 
        if dtype is not None: self._dtype = dtype
        return output
    @property
    def device(self):
        if self._device is None:
            self._device = next(self.parameters()).device
        return self._device
    @property 
    def dtype(self):
        if self._dtype is None:
            self._dtype = next(self.parameters()).dtype
        return self._dtype

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
    def __init__(self, d_model, n_head, causal, rope, rope_base):
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
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.c_fc    = nn.Linear(d_model, 4 * d_model, bias=False)
        self.c_proj  = nn.Linear(4 * d_model, d_model, bias=False)
        #self.zero_init()  # Initialize weights to zero
        self.orthogonalize() # Orthogonal init

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
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, n_layer, d_model, n_head, causal, rope, rope_base):
        super().__init__()
        self.n_layer = n_layer
        self.attn = CausalSelfAttention(d_model, n_head, causal, rope, rope_base)
        self.mlp = MLP(d_model)

    def forward(self, x, attention_mask=None, positions=None):
        L = self.n_layer
        x = (2*L-1)/(2*L) * x + (1/(2*L)) * self.attn(F.rms_norm(x, (x.size(-1),)), attention_mask=attention_mask, positions=positions)
        x = (2*L-1)/(2*L) * x + (1/(2*L)) * self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x

class Transformer(BFModule):
    def __init__(self, d_input=64, d_model=192, d_output=192, n_layer=10, n_head=12, causal=True, rope=True, cls_token=True, rope_base=1024, identity_init=True):
        super().__init__()

        assert d_input == d_output, "The identity transformer requires d_input == d_output"
        assert not cls_token, "The identity transformer does not support cls tokens"


        self.d_input = d_input
        self.n_layer = n_layer
        self.n_head = n_head 
        self.d_model = d_model  
        self.d_output = d_output
        self.causal = causal
        self.rope = rope
        self.rope_base = rope_base

        self.cls_token = nn.Parameter(torch.randn(d_model)) if cls_token else None

        self.residual_linear = nn.Linear(d_input, d_output, bias=False)

        self.embed = nn.Linear(d_input, d_model, bias=False)
        self.blocks = nn.ModuleList([Block(n_layer, d_model, n_head, causal, rope, rope_base) for _ in range(n_layer)])
        self.output_proj = nn.Linear(d_model, d_output, bias=False)

        #self.zero_init()  # Initialize weights to zero
        self.orthogonalize() # Orthogonal init

        if identity_init:
            self.output_proj.weight.data.zero_() # Init just the output to zero
            self.residual_linear.weight.data = torch.eye(self.d_input)
        else:
            self.residual_linear.weight.data.zero_() # Init the residual linear to zero
    
    def zero_init(self):
        self.embed.weight.data.zero_()
        self.output_proj.weight.data.zero_()
    
    def orthogonalize(self):
        self.embed.weight.data = orthogonalize(self.embed.weight.data)
        self.output_proj.weight.data = orthogonalize(self.output_proj.weight.data)

    def forward(self, x, attention_mask=None, positions=None, embeddings=None, strict_positions=False, stop_at_block=None):
        # x is of shape (batch_size, seq_len, d_input)
        batch_size, seq_len, d_input = x.shape

        x_original = x

        x = self.embed(x)  # shape: (batch_size, seq_len, d_model)

        if embeddings is not None:
            x = x + embeddings


        if attention_mask is None and positions is not None: # XXX this overwrites the "Causal" parameter
            if strict_positions:
                attention_mask = positions.unsqueeze(2) == positions.unsqueeze(1) # Causal mask given the positions
            else:
                attention_mask = positions.unsqueeze(2) >= positions.unsqueeze(1) # Causal mask given the positions


        if self.cls_token is not None:
            # Expand cls_token to match timebins dimension
            cls_token = self.cls_token.expand(batch_size, 1, -1)
            x = torch.cat([x, cls_token], dim=1)  # shape: (batch_size, seq_len + 1, d_model)
            
            # If there's a mask and cls_token, extend the mask for the cls token
            if attention_mask is not None:
                # # Add a column of True to allow attending to cls token (not needed, and would cause leaking of information)
                # cls_mask_col = torch.ones(batch_size, attention_mask.size(1), 1, device=attention_mask.device, dtype=torch.bool)
                # attention_mask = torch.cat([attention_mask, cls_mask_col], dim=2)
                
                # Add a row of True to allow cls token to attend to all
                cls_mask_row = torch.ones(batch_size, 1, attention_mask.size(2), device=attention_mask.device, dtype=torch.bool)
                attention_mask = torch.cat([attention_mask, cls_mask_row], dim=1)

            # If positions are provided, add a position for cls token
            if positions is not None:
                # Add max_position for cls token
                cls_position = positions.max(dim=1, keepdim=True, device=positions.device)
                positions = torch.cat([positions, cls_position], dim=1)

        for block_i, block in enumerate(self.blocks):
            x = block(x, attention_mask=attention_mask, positions=positions)
            if stop_at_block is not None and block_i+1 == stop_at_block:
                return x
        
        x = self.output_proj(x) # shape: (batch_size, seq_len (+1), d_output)

        return self.residual_linear(x_original) + x
    


class CrossAttentionBlock(nn.Module):
    def __init__(self, n_layer, d_model, n_head, rope, rope_base):
        super().__init__()
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = d_model
        self.rope = rope
        self.rope_base = rope_base
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        
        # Query comes from x, Key and Value come from y
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim, base=self.rope_base)
        
        self.mlp = MLP(d_model)
        self.orthogonalize()

    def orthogonalize(self):
        self.c_q.weight.data = orthogonalize(self.c_q.weight.data)
        self.c_k.weight.data = orthogonalize(self.c_k.weight.data)
        self.c_v.weight.data = orthogonalize(self.c_v.weight.data)
        self.c_proj.weight.data = orthogonalize(self.c_proj.weight.data)

    def forward(self, x, y, attention_mask=None, positions_x=None, positions_y=None):
        L = self.n_layer
        # Cross attention
        attn_out = self.c_q(x).view(x.size(0), x.size(1), self.n_head, self.head_dim)
        k = self.c_k(y).view(y.size(0), y.size(1), self.n_head, self.head_dim)
        v = self.c_v(y).view(y.size(0), y.size(1), self.n_head, self.head_dim)
        
        if self.rope:
            cos_q, sin_q = self.rotary(attn_out, positions_x)
            attn_out = apply_rotary_emb(attn_out, cos_q, sin_q)
            
            cos_k, sin_k = self.rotary(k, positions_y)
            k = apply_rotary_emb(k, cos_k, sin_k)
        
        attn_out = F.scaled_dot_product_attention(
            attn_out.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=attention_mask.unsqueeze(1) if attention_mask is not None else None, # the head dimension is broadcasted
            scale=1/attn_out.shape[-1]
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view_as(x)
        attn_out = self.c_proj(attn_out)
        
        # First residual connection with scaling
        x = (2*L-1)/(2*L) * x + (1/(2*L)) * attn_out
        
        # MLP with same scaling as original transformer
        x = (2*L-1)/(2*L) * x + (1/(2*L)) * self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x


class CrossAttentionTransformer(BFModule):
    def __init__(self, d_input=64, d_model=192, d_output=192, n_layer=10, n_head=12, rope=True, rope_base=1024):
        super().__init__()
        
        assert d_input == d_output, "The identity transformer requires d_input == d_output"
        
        self.d_input = d_input
        self.n_layer = n_layer
        self.n_head = n_head 
        self.d_model = d_model  
        self.d_output = d_output
        self.rope = rope
        self.rope_base = rope_base
        
        # Projections for input sequences
        self.y_embed = nn.Linear(d_input, d_model, bias=False)
        
        # Cross attention blocks
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(n_layer, d_model, n_head, rope, rope_base) 
            for _ in range(n_layer)
        ])
        
        self.output_proj = nn.Linear(d_model, d_output, bias=False)
        
        # Initialize weights
        self.orthogonalize()
        #self.output_proj.weight.data.zero_()
    
    def orthogonalize(self):
        self.y_embed.weight.data = orthogonalize(self.y_embed.weight.data)
        self.output_proj.weight.data = orthogonalize(self.output_proj.weight.data)
    
    def forward(self, x, y, attention_mask=None, positions_x=None, positions_y=None):
        # x: (batch_size, seq_len_x, d_model)
        # y: (batch_size, seq_len_y, d_input)
        x_original = x

        if attention_mask is None and positions_x is not None and positions_y is not None:
            attention_mask = positions_x.unsqueeze(2) >= positions_y.unsqueeze(1) # Causal mask given the positions

        # Embed both sequences
        y = self.y_embed(y)  # (batch_size, seq_len_y, d_model)
        
        # Apply cross attention blocks
        for block in self.blocks:
            x = block(x, y, attention_mask=attention_mask, positions_x=positions_x, positions_y=positions_y)
        
        # Project back to output dimension
        x = self.output_proj(x)
        
        return x

if __name__ == "__main__":
    transformer = Transformer(d_input=16, d_model=192, d_output=16, n_layer=2, n_head=12, causal=True, rope=True, cls_token=False, rope_base=1024)
    x = torch.randn(10, 100, 16)
    y = transformer(x)
    print(x.shape, y.shape)
    print(x-y)

