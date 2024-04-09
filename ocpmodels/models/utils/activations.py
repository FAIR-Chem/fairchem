"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn.functional as F


class Act(torch.nn.Module):
    def __init__(self, act: str, slope: float = 0.05) -> None:
        super(Act, self).__init__()
        self.act = act
        self.slope = slope
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.act == "relu":
            return F.relu(input)
        elif self.act == "leaky_relu":
            return F.leaky_relu(input)
        elif self.act == "sp":
            return F.softplus(input, beta=1)
        elif self.act == "leaky_sp":
            return F.softplus(input, beta=1) - self.slope * F.relu(-input)
        elif self.act == "elu":
            return F.elu(input, alpha=1)
        elif self.act == "leaky_elu":
            return F.elu(input, alpha=1) - self.slope * F.relu(-input)
        elif self.act == "ssp":
            return F.softplus(input, beta=1) - self.shift
        elif self.act == "leaky_ssp":
            return (
                F.softplus(input, beta=1)
                - self.slope * F.relu(-input)
                - self.shift
            )
        elif self.act == "tanh":
            return torch.tanh(input)
        elif self.act == "leaky_tanh":
            return torch.tanh(input) + self.slope * input
        elif self.act == "swish":
            return torch.sigmoid(input) * input
        else:
            raise RuntimeError(f"Undefined activation called {self.act}")
