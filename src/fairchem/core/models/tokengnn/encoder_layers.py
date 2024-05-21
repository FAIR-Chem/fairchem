import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter
from torch_geometric.nn.models.schnet import GaussianSmearing

from .utils import ResMLP

class TokenGNNEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        ff_dim: int = 1024,
        dropout: float = 0.,
        num_heads: int = 8,
        max_radius: float = 12,
        num_gaussians: int = 50,
    ):
        super().__init__()

        assert (embed_dim % num_heads) == 0

        self.query = Projection(embed_dim=embed_dim, num_heads=num_heads)
        self.key = Projection(embed_dim=embed_dim, num_heads=num_heads)
        self.value = Projection(embed_dim=embed_dim, num_heads=num_heads)

        self.dist_nn = nn.Sequential(
            GaussianSmearing(0, max_radius, num_gaussians),
            nn.Linear(num_gaussians, num_heads)
        )

        self.dist_ne = nn.Sequential(
            GaussianSmearing(0, max_radius, num_gaussians),
            nn.Linear(num_gaussians, num_heads)
        )

        self.dist_en = nn.Sequential(
            GaussianSmearing(0, max_radius, num_gaussians),
            nn.Linear(num_gaussians, num_heads)
        )

        self.angular_dd = nn.Sequential(
            GaussianSmearing(-1, 1, num_gaussians),
            nn.Linear(num_gaussians, num_heads)
        )
        self.angular_ss = nn.Sequential(
            GaussianSmearing(-1, 1, num_gaussians),
            nn.Linear(num_gaussians, num_heads)
        )


        self.output = nn.Linear(embed_dim, embed_dim)
        self.scale = 1 / math.sqrt(embed_dim // num_heads)

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
        edges: torch.Tensor,
        all_dist: torch.Tensor,
        dist: torch.Tensor,
        cosine_dd: torch.Tensor,
        cosine_ss: torch.Tensor,
    ):
        
        # projection
        z = self.attn_norm(x)
        query = self.key(z, edges[0])
        key = self.key(z, edges[1])
        value = self.key(z, edges[1]) # (E, H, d_k)

        # softmax
        logits = self.scale * (query * key).sum(2, keepdim=True) # (E, H, 1)
        attn_mask = torch.cat([
            self.dist_nn(all_dist),
            self.dist_ne(dist),
            self.dist_en(dist),
            self.angular_dd(cosine_dd),
            self.angular_ss(cosine_ss)
        ], dim = 0)
        logits += attn_mask[..., None] # (E, H, 1)
        logits = logits - scatter(logits.detach(), edges[0], dim=0, reduce="max")[edges[0]] # (E, H, 1)
        scores = torch.exp(logits) # (E, H, 1)
        scores = scores / scatter(scores, edges[0], dim=0, reduce="sum")[edges[0]] # (E, H, 1)

        # dropout 
        scores = self.dropout(scores)

        # value aggregation
        self_attn = scatter(scores * value, edges[0], dim=0, reduce="sum") # (L, H, d_k)
        self_attn = self_attn.view(self_attn.size(0), -1)
        self_attn = self.output(self_attn)

        # residual connection
        x = x + self.dropout(self_attn)

        # feed forward and residual connection
        z = self.norm_ff(x)
        ff = self.feed_forward(z)
        x = x + self.dropout(ff)

        return x
    
class Projection(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512, 
        num_heads: int = 8,
    ):
        super().__init__()
        d_k = embed_dim // num_heads
        self.linear = nn.Linear(embed_dim, num_heads * d_k)
        self.heads = num_heads
        self.d_k = d_k

    def forward(
        self,
        x: torch.Tensor,
        index: torch.Tensor,
    ):
        output = self.linear(x)[index]
        output = output.view(index.size(0), self.heads, self.d_k)

        return output


class OutputModule(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        ff_dim: int = 1024,
        hidden_layers: int = 3,
        dropout: float = 0,
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
        dist: torch.Tensor,
        vec_hat: torch.Tensor,
        batch: torch.Tensor,
        edge_index: torch.Tensor,
    ):
        # prepare inputs
        edge_token_pos = torch.arange(pos.size(0), pos.size(0) + edge_index.size(1), device=edge_index.device)
        inputs = torch.cat([x[edge_index[0]], x[edge_index[1]], x[edge_token_pos]], dim = -1)

        # regress energy and forces
        energy_pairs = self.energy_input(inputs) * self.energy_rbf(dist)
        force_pairs = self.forces_input(inputs) * self.forces_rbf(dist)
        energy_pairs = self.energy_output(energy_pairs)
        force_pairs = self.forces_output(force_pairs) * vec_hat
        energy = scatter(energy_pairs, batch[edge_index[0]], dim = 0, reduce="sum")
        forces = scatter(force_pairs, edge_index[0], dim = 0, reduce="sum")

        return energy, forces
