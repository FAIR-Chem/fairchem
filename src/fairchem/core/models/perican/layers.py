import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

from .utils import diagonal, diag_embed, masked_mean, masked_var, masked_sum, name_mapping

class RadialBasisFunction(nn.Module):
    def __init__(
        self,
        rbf_radius: float = 12.,
        num_gaussians: int = 50,
        embed_dim: int = 128,
    ):
        super().__init__()

        offset = torch.linspace(0, rbf_radius, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)
        self.linear = nn.Linear(num_gaussians, embed_dim, bias=False)

    def forward(self, x: torch.Tensor):
        # compute rbfs and embeddings
        shape = x.shape
        x = x.view(-1, 1) - self.offset.view(1, -1) # [..., C]
        x = x.view(*shape, -1)
        smeared = torch.exp(self.coeff * x.square())
        return self.linear(smeared)

class ResMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 1024,
        output_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0., 
    ):
        super().__init__()
        assert num_layers >= 2

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()

        self.linears = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) 
            for _ in range(num_layers - 2)
        ])

        self.input = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        x = self.input(x)
        x = self.dropout(x)
        x = self.activation(x)
        for linear in self.linears:
            z = linear(x)
            z = self.dropout(z)
            z = self.activation(z)
            x = x + z
        x = self.output(x)
        return x
    
class OutputModule(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        rbf_radius: float = 12.,
        num_gaussians: int = 50,
        num_layers: int = 4,
        dropout: float = 0.,
        avg_atoms: float = 60.,
    ):
        super().__init__()

        self.energy_input = ResMLP(
            input_dim=embed_dim,
            hidden_dim=embed_dim,
            output_dim=embed_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.forces_input = ResMLP(
            input_dim=embed_dim,
            hidden_dim=embed_dim,
            output_dim=embed_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.energy_rbf = RadialBasisFunction(
            rbf_radius=rbf_radius,
            num_gaussians=num_gaussians,
            embed_dim=embed_dim
        )

        self.forces_rbf = RadialBasisFunction(
            rbf_radius=rbf_radius,
            num_gaussians=num_gaussians,
            embed_dim=embed_dim
        )

        self.energy_output = nn.Linear(embed_dim, 1, False)
        self.forces_output = nn.Linear(embed_dim, 1, False)
        
        self.avg_atoms = avg_atoms
    
    def forward(
        self,
        T: torch.Tensor,
        mask: torch.Tensor,
        dist: torch.Tensor,
        vec_hat: torch.Tensor,
    ):
        energy = self.energy_input(T) * self.energy_rbf(dist)
        forces = self.forces_input(T) * self.forces_rbf(dist)

        energy = self.energy_output(energy)
        forces = self.forces_output(forces) * vec_hat

        energy = masked_sum(energy, mask, dim=(0, 1), keepdim=False, scale=self.avg_atoms**2)
        forces = masked_sum(forces, mask, dim=0, keepdim=False, scale=self.avg_atoms)

        return energy, forces


class T2toT2(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        eq_channels: int = 128,
        out_channels: int = 128,
        avg_len: float = 60.,
        dropout: float = 0.,
        agg_fns: List[str] = ["sum"]
    ):
        super().__init__()

        assert all(name in name_mapping for name in agg_fns)

        num_aggregators = (5 + 10 * len(agg_fns))

        self.linear1 = nn.Linear(in_channels, eq_channels)
        self.linear2 = nn.Linear(num_aggregators * eq_channels, out_channels)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = MaskedBatchNorm2d(in_channels)
        self.avg_len = avg_len

        self.agg_fns = [
            name_mapping[name] for name in agg_fns
        ]
        
    def forward(
        self,
        T: torch.Tensor, # (L, L, N, C)
        mask: torch.Tensor # (L, L, N)
    ):

        # normalization
        T = self.norm(T, mask)

        # first linear pass
        T = self.linear1(T)
        T = self.activation(T)
        T = self.dropout(T)

        # prepare inputs
        L = T.size(0)
        B = T.size(2)
        mask_diag = mask & torch.eye(mask.size(0), dtype=bool, device=mask.device)[:, :, None]
        diag = diagonal(T, dim1=0, dim2=1)
        transpose = T.transpose(0, 1)

        # compute broadcast outputs
        outputs = torch.cat([
            diag_embed(diag, dim1=0, dim2=1),
            diag[None, :].expand_as(T),
            diag[:, None].expand_as(T),
            transpose,
            T
        ], dim=3)

        for agg_fn in self.agg_fns:
            
            diag_agg = agg_fn(T, mask_diag, dim=(0, 1), keepdim=False, scale=self.avg_len)
            row_agg = agg_fn(T, mask, dim=0, keepdim=False, scale=self.avg_len)
            col_agg = agg_fn(T, mask, dim=1, keepdim=False, scale=self.avg_len)
            all_agg = agg_fn(T, mask, dim=(0, 1), keepdim=False, scale=self.avg_len**2)

            # compute aggregation outputs
            outputs = torch.cat([
                outputs,
                diag_embed(row_agg, dim1=0, dim2=1),
                diag_embed(col_agg, dim1=0, dim2=1),
                diag_embed(diag_agg[None].expand(L, -1, -1), dim1=0, dim2=1),
                diag_agg[None, :].expand_as(T),
                row_agg[None, :].expand_as(T),
                col_agg[None, :].expand_as(T),
                diag_embed(all_agg[None], dim1=0, dim2=1).expand_as(T),
                col_agg[:, None].expand_as(T),
                row_agg[:, None].expand_as(T),
                all_agg[None, None].expand_as(T)
            ], dim=3)

        # second linear pass
        return self.linear2(outputs)


class MaskedBatchNorm2d(nn.Module):
    def __init__(
        self,
        num_channels: int = 128,
        eps: float = 1e-5,
        momentum: float = 0.1
    ):
        super().__init__()

        self.register_buffer("running_mean", torch.zeros((num_channels,)))
        self.register_buffer("running_var", torch.ones((num_channels,)))

        self.weight = nn.Parameter(torch.empty((1, 1, 1, num_channels)))
        self.bias = nn.Parameter(torch.empty((1, 1, 1, num_channels)))

        self.momentum = momentum
        self.eps = eps
                                               
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.fill_(0)
        self.running_var.fill_(1)
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        T: torch.Tensor,
        mask: torch.Tensor
    ):
        if self.training:
            mean = masked_mean(T, mask).mean(0)
            var = masked_var(T, mask).mean(0)
            self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * mean.detach()
            self.running_var = (1-self.momentum) * self.running_var + self.momentum * var.detach()
        else:
            mean = self.running_mean
            var = self.running_var

        mean, var = mean[None, None, None], var[None, None, None]
        T = (T - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias

        return T


            
