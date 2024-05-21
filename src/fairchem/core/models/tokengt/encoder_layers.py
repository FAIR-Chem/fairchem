from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter

from torch_geometric.nn.models.schnet import GaussianSmearing

from .utils import ResMLP

class TokenGTEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        ff_dim: int = 1024,
        dropout: float = 0.,
        num_heads: int = 8,
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.feed_forward = ResMLP(
            input_dim=embed_dim,
            hidden_dim=ff_dim,
            output_dim=embed_dim,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.norm_ff = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        z = self.attn_norm(x)
        self_attn, *_ = self.attn(z, z, z, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = x + self.dropout(self_attn)
        z = self.norm_ff(x)
        ff = self.feed_forward(z)
        x = x + self.dropout(ff)

        return x

class AttentionBias(nn.Module):
    def __init__(
        self,
        ff_dim: int = 1024,
        dropout: float = 0.,
        num_heads: int = 8,
        num_layers: int = 12,
        # use_angle: bool = False,
    ):
        super().__init__()

        self.projection = ResMLP(
            6, ff_dim, num_heads * num_layers, dropout=dropout
        )
        self.num_heads = num_heads
        self.num_layers = num_layers

    def forward(
        self,
        padded_mask: torch.Tensor, 
        padded_node_mask: torch.Tensor, 
        padded_index: torch.Tensor,
    ) -> torch.Tensor:
        # transpose to get the correct shape
        padded_mask, padded_node_mask, padded_index = \
            padded_mask.transpose(0, 1), padded_node_mask.transpose(0, 1), padded_index.transpose(0, 1)
        eq_index = (padded_index[:, None] == padded_index[None, :]) # (L, L, N, 2)
        eq_swap_index = (padded_index[:, None].flip(-1) == padded_index[None, :]) # (L, L, N, 2)
        src_type = padded_node_mask[:, None, :, None].expand(-1, padded_node_mask.size(0), -1, -1) # (L, L, N, 1)
        dst_type = padded_node_mask[None, :, :, None].expand(padded_node_mask.size(0), -1, -1, -1) # (L, L, N, 1)
        inputs = torch.cat([eq_index, eq_swap_index, src_type, dst_type], dim = -1) # (L, L, N, 6)

        # project to attn weights
        attn = self.projection(inputs.float()) # (L, L, N, H * layers)
        attn.masked_fill_(~padded_mask[None, :, :, None], - torch.inf)

        # reshape to the correct shape
        attn = attn.permute(2, 3, 0, 1) # [N, H * layers, L, L]
        attn = attn.reshape(attn.size(0), self.num_layers, self.num_heads, attn.size(2), attn.size(3)) # [N, layers, H, L, L]
        attn = attn.permute(1, 0, 2, 3, 4) # [layers, N, H, L, L]
        attn = attn.reshape(attn.size(0), -1, attn.size(3), attn.size(4)) # [layers, N * H, L, L]

        return attn

class OutputModule(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        ff_dim: int = 1024,
        dropout: float = 0,
        hidden_layers: int = 3,
        num_gaussians: int = 50,
        max_radius: float = 12.
    ):
        super().__init__()

        self.energy_input = ResMLP(
            input_dim=3*embed_dim,
            hidden_dim=ff_dim,
            output_dim=embed_dim,
            num_layers=hidden_layers,
            dropout=dropout
        )

        self.energy_rbf = nn.Sequential(
            GaussianSmearing(0, max_radius, num_gaussians),
            nn.Linear(num_gaussians, embed_dim, bias=False),
        )

        self.energy_output = nn.Linear(embed_dim, 1, bias=False)

        self.forces_input = ResMLP(
            input_dim=3*embed_dim,
            hidden_dim=ff_dim,
            output_dim=embed_dim,
            num_layers=hidden_layers,
            dropout=dropout
        )

        self.forces_rbf = nn.Sequential(
            GaussianSmearing(0, max_radius, num_gaussians),
            nn.Linear(num_gaussians, embed_dim, bias=False),
        )

        self.forces_output = nn.Linear(embed_dim, 1, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        edge_index: torch.Tensor,
        padded_node_mask: torch.Tensor, 
        padded_edge_mask: torch.Tensor, 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # prepare inputs
        x = x.transpose(0, 1)
        nodes = x[padded_node_mask]
        edges = x[padded_edge_mask]
        inputs = torch.cat([nodes[edge_index[0]], nodes[edge_index[1]], edges], dim = -1)
        vec = pos[edge_index[0]] - pos[edge_index[1]]
        vec_hat = F.normalize(vec, dim = -1)
        dist = torch.linalg.norm(vec, dim=-1)

        # regress energy and forces
        energy_pairs = self.energy_input(inputs) * self.energy_rbf(dist)
        force_pairs = self.forces_input(inputs) * self.forces_rbf(dist)
        energy_pairs = self.energy_output(energy_pairs)
        force_pairs = self.forces_output(force_pairs) * vec_hat
        energy = scatter(energy_pairs, batch[edge_index[0]], dim = 0, reduce="sum")
        forces = scatter(force_pairs, edge_index[0], dim = 0, reduce="sum")

        return energy, forces
