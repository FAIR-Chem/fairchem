"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import torch
from torch import nn

from fairchem.core.common import gp_utils
from fairchem.core.common.registry import registry
from fairchem.core.models.base import GraphData, HeadInterface
from fairchem.core.models.equiformer_v2.transformer_block import FeedForwardNetwork
from fairchem.core.models.equiformer_v2.weight_initialization import eqv2_init_weights

if TYPE_CHECKING:
    from torch_geometric.data import Batch


@registry.register_model("equiformerV2_scalar_head")
class EqV2ScalarHead(nn.Module, HeadInterface):
    def __init__(self, backbone, output_name: str = "energy", reduce: str = "sum"):
        super().__init__()
        self.output_name = output_name
        self.reduce = reduce
        self.avg_num_nodes = backbone.avg_num_nodes
        self.energy_block = FeedForwardNetwork(
            backbone.sphere_channels,
            backbone.ffn_hidden_channels,
            1,
            backbone.lmax_list,
            backbone.mmax_list,
            backbone.SO3_grid,
            backbone.ffn_activation,
            backbone.use_gate_act,
            backbone.use_grid_mlp,
            backbone.use_sep_s2_act,
        )
        self.apply(partial(eqv2_init_weights, weight_init=backbone.weight_init))

    def forward(self, data: Batch, emb: dict[str, torch.Tensor | GraphData]):
        node_output = self.energy_block(emb["node_embedding"])
        node_output = node_output.embedding.narrow(1, 0, 1)
        if gp_utils.initialized():
            node_output = gp_utils.gather_from_model_parallel_region(node_output, dim=0)
        output = torch.zeros(
            len(data.natoms),
            device=node_output.device,
            dtype=node_output.dtype,
        )

        output.index_add_(0, data.batch, node_output.view(-1))
        if self.reduce == "sum":
            return {self.output_name: output / self.avg_num_nodes}
        elif self.reduce == "mean":
            return {self.output_name: output / data.natoms}
        else:
            raise ValueError(
                f"reduce can only be sum or mean, user provided: {self.reduce}"
            )
