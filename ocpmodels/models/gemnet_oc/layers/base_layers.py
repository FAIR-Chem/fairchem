"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
from typing import Optional

import torch

from ..initializers import he_orthogonal_init


class Dense(torch.nn.Module):
    """
    Combines dense layer with scaling for silu activation.

    Arguments
    ---------
    in_features: int
        Input embedding size.
    out_features: int
        Output embedding size.
    bias: bool
        True if use bias.
    activation: str
        Name of the activation function to use.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        activation: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

        if isinstance(activation, str):
            activation = activation.lower()
        if activation in ["silu", "swish"]:
            self._activation = ScaledSiLU()
        elif activation is None:
            self._activation = torch.nn.Identity()
        else:
            raise NotImplementedError(
                "Activation function not implemented for GemNet (yet)."
            )

    def reset_parameters(self, initializer=he_orthogonal_init) -> None:
        initializer(self.linear.weight)
        if self.linear.bias is not None:
            self.linear.bias.data.fill_(0)

    def forward(self, x):
        x = self.linear(x)
        x = self._activation(x)
        return x


class ScaledSiLU(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale_factor = 1 / 0.6
        self._activation = torch.nn.SiLU()

    def forward(self, x):
        return self._activation(x) * self.scale_factor


class ResidualLayer(torch.nn.Module):
    """
    Residual block with output scaled by 1/sqrt(2).

    Arguments
    ---------
    units: int
        Input and output embedding size.
    nLayers: int
        Number of dense layers.
    layer: torch.nn.Module
        Class for the layers inside the residual block.
    layer_kwargs: str
        Keyword arguments for initializing the layers.
    """

    def __init__(
        self, units: int, nLayers: int = 2, layer=Dense, **layer_kwargs
    ) -> None:
        super().__init__()
        self.dense_mlp = torch.nn.Sequential(
            *[
                layer(
                    in_features=units,
                    out_features=units,
                    bias=False,
                    **layer_kwargs
                )
                for _ in range(nLayers)
            ]
        )
        self.inv_sqrt_2 = 1 / math.sqrt(2)

    def forward(self, input):
        x = self.dense_mlp(input)
        x = input + x
        x = x * self.inv_sqrt_2
        return x
