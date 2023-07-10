"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn


# Different encodings for the atom distance embeddings
class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = -5.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
        basis_width_scalar: float = 1.0,
    ) -> None:
        super(GaussianSmearing, self).__init__()
        self.num_output = num_gaussians
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = (
            -0.5 / (basis_width_scalar * (offset[1] - offset[0])).item() ** 2
        )
        self.register_buffer("offset", offset)

    def forward(self, dist) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class SigmoidSmearing(torch.nn.Module):
    def __init__(
        self, start=-5.0, stop=5.0, num_sigmoid=50, basis_width_scalar=1.0
    ) -> None:
        super(SigmoidSmearing, self).__init__()
        self.num_output = num_sigmoid
        offset = torch.linspace(start, stop, num_sigmoid)
        self.coeff = (basis_width_scalar / (offset[1] - offset[0])).item()
        self.register_buffer("offset", offset)

    def forward(self, dist) -> torch.Tensor:
        exp_dist = self.coeff * (dist.view(-1, 1) - self.offset.view(1, -1))
        return torch.sigmoid(exp_dist)


class LinearSigmoidSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = -5.0,
        stop: float = 5.0,
        num_sigmoid: int = 50,
        basis_width_scalar: float = 1.0,
    ) -> None:
        super(LinearSigmoidSmearing, self).__init__()
        self.num_output = num_sigmoid
        offset = torch.linspace(start, stop, num_sigmoid)
        self.coeff = (basis_width_scalar / (offset[1] - offset[0])).item()
        self.register_buffer("offset", offset)

    def forward(self, dist) -> torch.Tensor:
        exp_dist = self.coeff * (dist.view(-1, 1) - self.offset.view(1, -1))
        x_dist = torch.sigmoid(exp_dist) + 0.001 * exp_dist
        return x_dist


class SiLUSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = -5.0,
        stop: float = 5.0,
        num_output: int = 50,
        basis_width_scalar: float = 1.0,
    ) -> None:
        super(SiLUSmearing, self).__init__()
        self.num_output = num_output
        self.fc1 = nn.Linear(2, num_output)
        self.act = nn.SiLU()

    def forward(self, dist):
        x_dist = dist.view(-1, 1)
        x_dist = torch.cat([x_dist, torch.ones_like(x_dist)], dim=1)
        x_dist = self.act(self.fc1(x_dist))
        return x_dist
