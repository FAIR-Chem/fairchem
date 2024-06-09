import torch
import torch.nn as nn
import logging

from .mlp import ResMLP
from .rbf import RadialBasisFunction

class PairEmbed(nn.Module):
    def __init__(
        self,
        num_elements: int = 100,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_masks: int = 1,
        num_gaussians: int = 50,
        rbf_radius: float = 12.,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.num_masks = num_masks
        self.num_elemenets = num_elements

        self.smearing = RadialBasisFunction(
            rbf_radius=rbf_radius,
            num_gaussians=num_gaussians,
            embed_dim=hidden_dim
        )

        self.embedding = nn.Embedding(
            num_embeddings=num_elements**2,
            embedding_dim=embed_dim
        )

        self.mlp = ResMLP(
            input_dim=hidden_dim+embed_dim,
            hidden_dim=hidden_dim,
            output_dim=num_heads*num_masks
        )

    def forward(
        self,
        anum: torch.Tensor,
        row_index: torch.Tensor, 
        col_index: torch.Tensor,
        dist: torch.Tensor,
    ):
        rbf = self.smearing(dist)
        emb = self.embedding(anum[row_index] + self.num_elemenets * anum[col_index])
        att_bias = self.mlp(torch.cat([rbf, emb], dim=1), gate=rbf)
        att_bias = att_bias.reshape(dist.size(0), self.num_heads, self.num_masks)
        
        return att_bias.permute(2, 1, 0).contiguous()