import torch
import torch.nn as nn
from typing import Optional

from torch_scatter import scatter

from .mlp import ResMLP
from .rbf import RadialBasisFunction
from .sparse_att import SparseScaledDotProduct, Projection, _from_coo

class OutputModule(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 1024,
        dropout: float = 0,
        hidden_layers: int = 3,
        num_gaussians: int = 50,
        rbf_radius: float = 12.,
        avg_len: float = 60.,
    ):
        super().__init__()

        self.avg_len = avg_len
        self.rbf_radius = rbf_radius

        self.energy_mlp = ResMLP(
            input_dim=3*embed_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=hidden_layers,
            dropout=dropout
        )

        self.forces_mlp = ResMLP(
            input_dim=3*embed_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=hidden_layers,
            dropout=dropout
        )

        self.input_rbf = RadialBasisFunction(
            rbf_radius=rbf_radius,
            num_gaussians=num_gaussians,
            embed_dim=embed_dim
        )

    def forward(
            self, 
            x: torch.Tensor,
            edge_index: torch.Tensor,
            batch: torch.Tensor,
            dist: torch.Tensor,
            vec_hat: torch.Tensor,
        ):

        # prepare pair embeddings
        rbf = self.input_rbf(dist)

        inputs = torch.cat([
            x[edge_index[0]], 
            x[edge_index[1]], 
            rbf
        ], dim=-1) # [L, 3D]

        # regress outputs
        energy_pairs = self.energy_mlp(inputs)
        force_pairs = self.forces_mlp(inputs) * vec_hat

        energy = scatter(
            src = energy_pairs, 
            index = batch[edge_index[0]], 
            dim = 0,
            dim_size = batch.max() + 1,
            reduce = "sum"
        ) / (self.avg_len ** 2)

        forces = scatter(
            src = force_pairs, 
            index = edge_index[0], 
            dim = 0,
            dim_size = x.size(0),
            reduce = "sum"
        ) / self.avg_len

        return energy, forces
    
class AttentionOutputModule(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 1024,
        dropout: float = 0.,
        att_dropout: float = 0.,
        layers: int = 3,
        num_heads: int = 8,
        avg_len: float = 60.,
    ):
        super().__init__()

        self.query_proj = Projection(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
        
        self.key_proj = Projection(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

        self.attention = SparseScaledDotProduct(dropout=att_dropout)

        self.energy_mlp = ResMLP(
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=layers,
            dropout=dropout,
        )

        self.forces_mlp = ResMLP(
            input_dim=embed_dim+3*num_heads,
            hidden_dim=hidden_dim,
            output_dim=3,
            num_layers=layers,
            dropout=dropout,
        )

        self.norm_att = nn.LayerNorm(embed_dim)
        self.norm_mlp = nn.LayerNorm(embed_dim)

        self.avg_len = avg_len
        self.num_heads = num_heads

    def forward(
            self, 
            x: torch.Tensor,
            edge_index: torch.Tensor,
            att_bias: torch.Tensor,
            pos: torch.Tensor,
            batch: torch.Tensor,
            need_weights: Optional[bool] = False
        ):

        # normalize inputs
        z_att = self.norm_att(x)

        # prepare inputs to attention
        query = self.query_proj(z_att)
        key = self.key_proj(z_att)
        value = pos.expand(self.num_heads, -1, -1)

        # construct CSR format mask
        mask = _from_coo(x.size(0), x.size(0), edge_index[0], edge_index[1], att_bias)

        # compute scaled dot product attention
        att, _ = self.attention(query, key, value, mask)

        # subtract positions to get displacements
        att = att - value

        # combine batched dimensions
        att = att.permute(1, 0, 2).view(x.size(0), -1)

        # normalize for mlp
        z_mlp = self.norm_mlp(x)
        energy = self.energy_mlp(z_mlp)
        forces = self.forces_mlp(torch.cat([z_mlp, att], dim=-1))

        return energy, forces