import math
import torch
import torch.nn as nn

from torch_scatter import scatter

from .mlp import ResMLP
from .rbf import RadialBasisFunction

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
        connectivity: float = 32.,
    ):
        super().__init__()

        self.avg_len = avg_len
        self.connectivity = connectivity
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
        ) / (self.avg_len * self.connectivity)

        forces = scatter(
            src = force_pairs, 
            index = edge_index[0], 
            dim = 0,
            dim_size = x.size(0),
            reduce = "sum"
        ) / (self.connectivity)

        return energy, forces