"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import math

import torch

from fairchem.core.models.equiformer_v2.radial_function import RadialFunction
from fairchem.core.models.equiformer_v2.so3 import SO3_LinearV2


def eqv2_init_weights(m, weight_init):
    if isinstance(m, (torch.nn.Linear, SO3_LinearV2)):
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
        if weight_init == "normal":
            std = 1 / math.sqrt(m.in_features)
            torch.nn.init.normal_(m.weight, 0, std)
    elif isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.constant_(m.bias, 0)
        torch.nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, RadialFunction):
        m.apply(eqv2_uniform_init_linear_weights)


def eqv2_uniform_init_linear_weights(m):
    if isinstance(m, torch.nn.Linear):
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
        std = 1 / math.sqrt(m.in_features)
        torch.nn.init.uniform_(m.weight, -std, std)
