import torch
import torch.nn as nn
import torch.nn.functional as F

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
        trainable_rbf: bool = False,
        use_gated_mlp: bool = False,
        avg_len: float = 60.,
    ):
        super().__init__()

        self.use_gated_mlp = use_gated_mlp
        self.avg_len = avg_len

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
            embed_dim=embed_dim,
            trainable=trainable_rbf,
        )

        if use_gated_mlp:
            self.energy_rbf = RadialBasisFunction(
                rbf_radius=rbf_radius,
                num_gaussians=num_gaussians,
                embed_dim=hidden_dim,
                trainable=trainable_rbf,
            )

            self.forces_rbf = RadialBasisFunction(
                rbf_radius=rbf_radius,
                num_gaussians=num_gaussians,
                embed_dim=hidden_dim,
                trainable=trainable_rbf,
            )

    def forward(
            self, 
            x: torch.Tensor,
            dist: torch.Tensor,
            vec_hat: torch.Tensor,
            mask: torch.Tensor,
        ):

        # prepare mask
        entries = mask.T[:, None] & mask.T[None, :] # [L, L, N]
        entries = entries[..., None]

        # prepare pair embeddings
        rbf = self.input_rbf(dist)
        x = torch.cat([
            x[:, None].expand_as(rbf), 
            x[None, :].expand_as(rbf), 
            rbf
        ], dim=-1) # [L, L, N, 3D]

        # regress outputs
        if self.use_gated_mlp:
            energy_pairs = self.energy_mlp(x, gate=self.energy_rbf(dist))
            force_pairs = self.forces_mlp(x, gate=self.forces_rbf(dist)) * vec_hat
        else:
            energy_pairs = self.energy_mlp(x)
            force_pairs = self.forces_mlp(x) * vec_hat

        energy = (energy_pairs * entries.float()).sum((0, 1)) / (self.avg_len ** 2) # [N, 1]
        forces = (force_pairs * entries.float()).sum(0) / self.avg_len # [L, N, 3]
        forces = forces.transpose(0, 1)[mask] # [S, 3]

        return energy, forces