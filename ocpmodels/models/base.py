"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch.nn as nn
import torch


class BaseModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def reset_parameters(self):
        for child in self.children():
            if hasattr(child, "reset_parameters"):
                assert isinstance(child, nn.Module)
                assert callable(child.reset_parameters)
                child.reset_parameters()
            else:
                if hasattr(child, "weight"):
                    nn.init.xavier_uniform_(child.weight)
                if hasattr(child, "bias"):
                    child.bias.data.fill_(0)

    def energy_forward(self, data):
        raise NotImplementedError

    def forces_forward(self, preds):
        raise NotImplementedError

    def forward(self, data):
        if self.regress_forces_as_grad:
            data.pos.requires_grad_(True)
        preds = self.energy_forward(data)
        forces = self.forces_forward(preds)
        grad_forces = None

        if self.regress_forces_as_grad or self.direct_forces:
            if self.regress_forces_as_grad:
                if "gemnet" in self.__class__.__name__.lower():
                    assert forces is not None
                    grad_forces = forces
                else:
                    assert (
                        forces is None
                    ), "A force decoder and forces_as_grad are mutually exclusive"
            else:
                assert forces is not None, "direct_forces requires a force decoder"

            grad_forces = grad_forces or -1 * (
                torch.autograd.grad(
                    preds["energy"],
                    data.pos,
                    grad_outputs=torch.ones_like(preds["energy"]),
                    create_graph=True,
                )[0]
            )

            if self.regress_forces_as_grad:
                preds["forces"] = grad_forces
            else:
                preds["forces"] = forces
                preds["grad_forces"] = grad_forces

        return preds

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
