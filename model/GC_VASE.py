import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class AdaptiveModuleManager(nn.Module):
    """
    Manages subject-specific adaptive encoder/decoder modules.
    NOTE: assumes inputs are the same dimensionality, i.e.
    expects inputs AFTER already handling splitting batches by subject 
    """
    def __init__(self, n_fixed_dim: int):
        super().__init__()
        self.n_fixed_dim = n_fixed_dim
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()

    def add_subject(self, subject_id: str, n_electrodes: int):
        """Register a new subject and create corresponding encoder/decoder."""
        self.encoders[subject_id] = nn.Linear(n_electrodes, self.n_fixed_dim)
        self.decoders[subject_id] = nn.Linear(self.n_fixed_dim, n_electrodes)

    def encode(self, x: torch.Tensor, subject_id: str) -> torch.Tensor:
        """
        Encodes an entire subject-specific batch.

        x: (batch, n_electrodes, seq_len)
        subject_id: str

        returns: (batch, fixed_dim, seq_len)
        """
        enc = self.encoders[subject_id]
        # Transpose to (batch, seq_len, n_elec) for linear
        x = x.permute(0, 2, 1)  # (batch, seq_len, n_elec)
        h = enc(x)              # (batch, seq_len, fixed_dim)
        h = h.permute(0, 2, 1)  # (batch, fixed_dim, seq_len)
        return h

    def decode(self, h: torch.Tensor, subject_id: str) -> torch.Tensor:
        """
        Decodes an entire subject-specific batch.

        h: (batch, fixed_dim, seq_len)
        subject_id: str

        returns: (batch, n_electrodes, seq_len)
        """
        dec = self.decoders[subject_id]
        h = h.permute(0, 2, 1)  # (batch, seq_len, fixed_dim)
        x_recon = dec(h)        # (batch, seq_len, n_elec)
        x_recon = x_recon.permute(0, 2, 1)  # (batch, n_elec, seq_len)
        return x_recon
        
class GraphConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, edge_index):
        super().__init__()
        self.conv = GCNConv(in_c, out_c)
        self.edge_index = edge_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        flat = x.permute(0, 2, 1).reshape(-1, c)
        h = self.conv(flat, self.edge_index)
        h = F.relu(h)
        return h.view(b, t, -1).permute(0, 2, 1)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_ff, dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)

class GC_VASE(nn.Module):
    def __init__(
        self,
        seq_len: int,
        latent_dim_s: int,
        latent_dim_r: int,
        edge_index = None,
        n_fixed_dim: int = 128,
        n_gcnn_layers: int = 0, # not currently used
        n_transformer_layers: int = 4,
        transformer_heads: int = 8,
        transformer_ff: int = 512,
    ):
        super().__init__()
        self.seq_len = seq_len

        # latent dimensionality for subject-specific and residual features
        self.latent_dim_s = latent_dim_s
        self.latent_dim_r = latent_dim_r
        # define input 
        self.adaptive = AdaptiveModuleManager(n_fixed_dim)

        # spatial GCNN
        dims = [n_fixed_dim] * (n_gcnn_layers + 1)
        self.gcn_blocks = nn.ModuleList([
            GraphConvBlock(dims[i], dims[i+1], edge_index)
            for i in range(n_gcnn_layers)
        ])

        # positional embed
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, n_fixed_dim))

        # temporal Transformers
        self.trans_enc = nn.ModuleList([
            TransformerBlock(n_fixed_dim, transformer_heads, transformer_ff)
            for _ in range(n_transformer_layers)
        ])
        self.trans_dec = nn.ModuleList([
            TransformerBlock(n_fixed_dim, transformer_heads, transformer_ff)
            for _ in range(n_transformer_layers)
        ])

        # latent projection (mu/logvar for both s and r)
        flat_dim = n_fixed_dim * seq_len
        total_latent = latent_dim_s + latent_dim_r
        self.fc_latent = nn.Linear(flat_dim, 2 * total_latent)

        # decoder FC
        self.fc_dec = nn.Linear(total_latent, flat_dim)

        # inverse GCNN blocks
        self.gcn_dec = nn.ModuleList([
            GraphConvBlock(n_fixed_dim, n_fixed_dim, edge_index)
            for _ in range(n_gcnn_layers)
        ])

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, subject_id):
        # x: (batch, n_elec_i, seq_len)
        # subject-specific encoding to standard input dim 
        # (note: expects inputs within same batch are same dimensionality)
        h = self.adaptive.encode(x, subject_id)
        
        # spatial encoding
        for g in self.gcn_blocks:
            h = g(h)
            
        # temporal encoding
        h_t = h.permute(0, 2, 1) + self.pos_emb
        for tr in self.trans_enc:
            h_t = tr(h_t)
            
        flat = h_t.reshape(h_t.size(0), -1)
        
        # latent reparameterization
        latent_params = self.fc_latent(flat)
        mu, logvar = latent_params.chunk(2, dim=-1)
        mu_s, mu_r = mu.split([self.latent_dim_s, self.latent_dim_r], dim=-1)
        lv_s, lv_r = logvar.split([self.latent_dim_s, self.latent_dim_r], dim=-1)
        z_s = self.reparam(mu_s, lv_s)
        z_r = self.reparam(mu_r, lv_r)
        z = torch.cat([z_s, z_r], dim=-1)

        # decode
        dec = self.fc_dec(z).view(-1, self.pos_emb.size(-1), self.seq_len)
        d_t = dec.permute(0, 2, 1)
        
        # temporal decoding
        for tr in self.trans_dec:
            d_t = tr(d_t)
        d_h = d_t.permute(0, 2, 1)
        
        # spatial decoding
        for g in self.gcn_dec:
            d_h = g(d_h) 
            
        # adaptive decoding
        out = self.adaptive.decode(d_h, subject_ids)
        return out, mu_s, lv_s, mu_r, lv_r

# Usage:
# model = GC_VASE(seq_len=256, latent_dim_s=32, latent_dim_r=32, edge_index=edge_index)
# for sid, n_e in [('subj1',64), ('subj2',32)]: model.adaptive.add_subject(sid, n_e)
# x = torch.randn(2, n_e, 256)
# recon, mus, lvs, mur, lvr = model(x, sid)
