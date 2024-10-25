from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import torch
from torch import nn

from fairchem.core.common import gp_utils
from fairchem.core.common.registry import registry
from fairchem.core.models.base import HeadInterface
from fairchem.core.models.equiformer_v2.equiformer_v2 import eqv2_init_weights
from fairchem.core.models.equiformer_v2.transformer_block import (
    SO2EquivariantGraphAttention,
)

if TYPE_CHECKING:
    from torch_geometric.data import Batch


@registry.register_model("equiformerV2_vector_head")
class EqV2VectorHead(nn.Module, HeadInterface):
    def __init__(self, backbone):
        super().__init__()

        self.activation_checkpoint = backbone.activation_checkpoint
        self.force_block = SO2EquivariantGraphAttention(
            backbone.sphere_channels,
            backbone.attn_hidden_channels,
            backbone.num_heads,
            backbone.attn_alpha_channels,
            backbone.attn_value_channels,
            1,
            backbone.lmax_list,
            backbone.mmax_list,
            backbone.SO3_rotation,
            backbone.mappingReduced,
            backbone.SO3_grid,
            backbone.max_num_elements,
            backbone.edge_channels_list,
            backbone.block_use_atom_edge_embedding,
            backbone.use_m_share_rad,
            backbone.attn_activation,
            backbone.use_s2_act_attn,
            backbone.use_attn_renorm,
            backbone.use_gate_act,
            backbone.use_sep_s2_act,
            alpha_drop=0.0,
        )
        self.apply(partial(eqv2_init_weights, weight_init=backbone.weight_init))

    def forward(self, data: Batch, emb: dict[str, torch.Tensor]):
        if self.activation_checkpoint:
            forces = torch.utils.checkpoint.checkpoint(
                self.force_block,
                emb["node_embedding"],
                emb["graph"].atomic_numbers_full,
                emb["graph"].edge_distance,
                emb["graph"].edge_index,
                emb["graph"].node_offset,
                use_reentrant=not self.training,
            )
        else:
            forces = self.force_block(
                emb["node_embedding"],
                emb["graph"].atomic_numbers_full,
                emb["graph"].edge_distance,
                emb["graph"].edge_index,
                node_offset=emb["graph"].node_offset,
            )
        forces = forces.embedding.narrow(1, 1, 3)
        forces = forces.view(-1, 3).contiguous()
        if gp_utils.initialized():
            forces = gp_utils.gather_from_model_parallel_region(forces, dim=0)
        return {"forces": forces}
