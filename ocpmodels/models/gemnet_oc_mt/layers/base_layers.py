"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
from typing import Callable

import torch.nn as nn

from ..config import BackboneConfig
from ..initializers import he_orthogonal_init


class Dense(nn.Module):
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
        in_features,
        out_features,
        bias=False,
        activation=None,
        ln: bool | str | None = None,
        dropout: float | None = None,
        scale_dim: bool = False,
    ):
        super().__init__()

        self.scale_dim = scale_dim

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

        if isinstance(activation, str):
            activation = activation.lower()
        if activation in ["scaled_silu", "scaled_swish"]:
            self.activation = ScaledSiLU()
        elif activation in ["silu", "swish"]:
            # self.activation = nn.SiLU()
            self.activation = ScaledSiLU()
        elif activation is None:
            self.activation = nn.Identity()
        else:
            raise NotImplementedError(
                "Activation function not implemented for GemNet (yet)."
            )

        if ln is None:
            ln = BackboneConfig.instance().ln

        if dropout is None:
            dropout = BackboneConfig.instance().dropout

        if isinstance(self.activation, nn.Identity):
            ln = False
            # dropout = None

        self.dropout = (
            nn.Dropout(dropout) if dropout is not None else nn.Identity()
        )

        self.ln_kind = "pre" if isinstance(ln, bool) else ln
        match ln:
            case True | "pre":
                self.ln = nn.LayerNorm(in_features)
            case "post":
                self.ln = nn.LayerNorm(out_features)
            case False:
                self.ln = nn.Identity()
            case _:
                raise ValueError(
                    f"ln must be bool or 'pre' or 'post' but got {ln}"
                )

    def reset_parameters(
        self,
        initializer=he_orthogonal_init,
        ln_initializer: Callable[[nn.Parameter], None] | None = None,
    ):
        initializer(self.linear.weight)
        if self.linear.bias is not None:
            _ = self.linear.bias.data.fill_(0)

        if ln_initializer is not None and isinstance(self.ln, nn.LayerNorm):
            ln_initializer(self.ln.weight)
            _ = self.ln.bias.data.fill_(0)

    def forward(self, x):
        if self.ln_kind == "pre":
            x = self.ln(x)
        x = self.linear(x)
        x = self.activation(x)
        if self.ln_kind == "post":
            x = self.ln(x)
        x = self.dropout(x)
        if self.scale_dim:
            x = x * (self.linear.weight.shape[1] ** -0.5)
        return x


class ScaledSiLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1 / 0.6
        self._activation = nn.SiLU()

    def forward(self, x):
        return self._activation(x) * self.scale_factor


class ResidualLayer(nn.Module):
    """
    Residual block with output scaled by 1/sqrt(2).

    Arguments
    ---------
    units: int
        Input and output embedding size.
    nLayers: int
        Number of dense layers.
    layer: nn.Module
        Class for the layers inside the residual block.
    layer_kwargs: str
        Keyword arguments for initializing the layers.
    """

    def __init__(
        self,
        units: int,
        nLayers: int = 2,
        layer=Dense,
        **layer_kwargs,
    ):
        super().__init__()

        self.dense_mlp = nn.Sequential(
            *[
                layer(
                    in_features=units,
                    out_features=units,
                    bias=False,
                    **layer_kwargs,
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
