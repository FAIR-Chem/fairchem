import torch
import torch.nn as nn

from .mlp import ResMLP
from .rbf import RadialBasisFunction, GaussianSmearing
    
class PairEmbed(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_masks: int = 1,
        num_gaussians: int = 50,
        rbf_radius: float = 12.,
        dropout: float = 0.,
        num_layers: int = 2,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.num_masks = num_masks

        self.mlp = ResMLP(
            input_dim=3*embed_dim,
            hidden_dim=hidden_dim,
            output_dim=num_heads*num_masks,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.rbf = RadialBasisFunction(
            rbf_radius=rbf_radius,
            num_gaussians=num_gaussians,
            embed_dim=embed_dim,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        dist: torch.Tensor,
    ):
        # prepare pairs    
        pairs = self.rbf(dist)

        # prepare inputs
        inputs = torch.cat([
            x[edge_index[0]],
            x[edge_index[1]],
            pairs
        ], dim = -1)

        attn_bias = self.mlp(inputs)
        attn_bias = attn_bias.reshape(-1, self.num_masks, self.num_heads)
        attn_bias = attn_bias.permute(1, 0, 2)

        return attn_bias