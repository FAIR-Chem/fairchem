"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch.nn as nn
import torch


class BaseModel(nn.Module):
    def __init__(self, **kwargs):
        super(BaseModel, self).__init__()

    def energy_forward(self, data):
        raise NotImplementedError

    def forward(self, data):
        if self.regress_forces:
            data.pos.requires_grad_(True)
        energy = self.energy_forward(data)

        if self.regress_forces:
            forces = -1 * (
                torch.autograd.grad(
                    energy,
                    data.pos,
                    grad_outputs=torch.ones_like(energy),
                    create_graph=True,
                )[0]
            )
            return energy, forces
        else:
            return energy

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
