import math
import torch
import torch.nn as nn

from .mlp import ResMLP
from .rbf import GaussianSmearing

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
        dropout: float = 0.,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.num_masks = num_masks
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_elemenets = num_elements

        self.smearing = GaussianSmearing(
            rbf_radius=rbf_radius,
            num_gaussians=num_gaussians,
        )

        self.gate_linear = nn.Linear(
            num_gaussians, hidden_dim, False
        )

        self.embedding = nn.Embedding(
            num_embeddings=num_elements**2,
            embedding_dim=embed_dim
        )

        self.mlp = ResMLP(
            input_dim=embed_dim+num_gaussians,
            hidden_dim=hidden_dim,
            output_dim=num_heads*num_masks,
            dropout=dropout
        )

        self.reset_parameters()

    def reset_parameters(self):
        # kaiming initialization, with tweaks
        nn.init.uniform_(
            self.mlp.input.weight[:, :self.embed_dim],
            - math.sqrt(3 / self.embed_dim),
            math.sqrt(3 / self.embed_dim)
        )
        # Gaussaian smearing results in a sum of 2.50663 (eliptic function)
        nn.init.uniform_(
            self.mlp.input.weight[:, self.embed_dim:],
            - math.sqrt(3 / 2.50663),
            math.sqrt(3 / 2.50663)
        )
        # initialize such that the Linear layer outputs unit variance
        nn.init.uniform_(
            self.gate_linear.weight,
            - math.sqrt(3 / 2.50663),
            math.sqrt(3 / 2.50663)
        )

    def forward(
        self,
        anum: torch.Tensor,
        edge_index: torch.Tensor, 
        edge_to_src: torch.Tensor,
        dist: torch.Tensor,
    ):
        rbf = self.smearing(dist)
        emb = self.embedding(anum[edge_index[0]] + self.num_elemenets * anum[edge_index[1]])
        if edge_to_src is not None:
            att_bias = self.mlp(torch.cat([emb[edge_to_src], rbf], dim=-1) , gate=self.gate_linear(rbf))
        else:
            att_bias = self.mlp(torch.cat([emb, rbf], dim=-1) , gate=self.gate_linear(rbf))
        att_bias = att_bias.reshape(dist.size(0), self.num_heads, self.num_masks)
        
        return att_bias.permute(2, 1, 0).contiguous()