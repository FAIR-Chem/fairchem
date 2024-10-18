"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

import math
from functools import partial
from typing import TYPE_CHECKING

import torch

from fairchem.core.common import gp_utils
from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad

try:
    from e3nn import o3
except ImportError:
    import contextlib
    contextlib.suppress(ImportError)

if TYPE_CHECKING:
    from torch_geometric.data.batch import Batch

from fairchem.core.models.base import GraphData, HeadInterface
from fairchem.core.models.equiformer_v2.equiformer_v2 import (
    EquiformerV2Backbone,
    eqv2_init_weights,
)
from fairchem.core.models.equiformer_v2.prediction_heads.rank2 import (
    Rank2SymmetricTensorHead,
)
from fairchem.core.models.equiformer_v2.so3 import SO3_Embedding, SO3_LinearV2
from fairchem.core.models.equiformer_v2.transformer_block import (
    FeedForwardNetwork,
    SO2EquivariantGraphAttention,
)

_AVG_NUM_NODES = 31.16592360068011
_AVG_DEGREE = 61.94676351484548

@registry.register_model("equiformer_v2_dens_backbone")
class EqV2DeNSBackbone(EquiformerV2Backbone):
    """
    DeNS extra Args:
        use_force_encoding (bool):                  For ablation study, whether to encode forces during denoising positions. Default: True.
        use_noise_schedule_sigma_encoding (bool):   For ablation study, whether to encode the sigma (sampled std of Gaussian noises) during
                                                    denoising positions when `fixed_noise_std` = False in config files. Default: False.
        use_denoising_energy (bool):                For ablation study, whether to predict the energy of the original structure given
                                                    a corrupted structure. If `False`, we zero out the energy prediction. Default: True.
    """
    def __init__(
        self,
        use_pbc=True,
        use_pbc_single=True,
        regress_forces=True,
        otf_graph=True,
        max_neighbors=500,
        max_radius=5.0,
        max_num_elements=90,

        num_layers=12,
        sphere_channels=128,
        attn_hidden_channels=128,
        num_heads=8,
        attn_alpha_channels=32,
        attn_value_channels=16,
        ffn_hidden_channels=512,

        norm_type="rms_norm_sh",

        lmax_list=None,
        mmax_list=None,
        grid_resolution=None,

        num_sphere_samples=128,

        edge_channels=128,
        use_atom_edge_embedding=True,
        share_atom_edge_embedding=False,
        use_m_share_rad=False,
        distance_function="gaussian",
        num_distance_basis=512,

        attn_activation="scaled_silu",
        use_s2_act_attn=False,
        use_attn_renorm=True,
        ffn_activation="scaled_silu",
        use_gate_act=False,
        use_grid_mlp=False,
        use_sep_s2_act=True,

        alpha_drop=0.1,
        drop_path_rate=0.05,
        proj_drop=0.0,

        weight_init="normal",

        enforce_max_neighbors_strictly=True,

        avg_num_nodes=_AVG_NUM_NODES,
        avg_degree=_AVG_DEGREE,

        use_force_encoding=True,
        use_noise_schedule_sigma_encoding=False,
        use_denoising_energy=True,
        use_denoising_stress=True,

        use_energy_lin_ref=False,
        load_energy_lin_ref=False,
        activation_checkpoint=False,
    ):
        if mmax_list is None:
            mmax_list = [2]
        if lmax_list is None:
            lmax_list = [6]
        super().__init__(
            use_pbc,
            use_pbc_single,
            regress_forces,
            otf_graph,
            max_neighbors,
            max_radius,
            max_num_elements,

            num_layers,
            sphere_channels,
            attn_hidden_channels,
            num_heads,
            attn_alpha_channels,
            attn_value_channels,
            ffn_hidden_channels,

            norm_type,

            lmax_list,
            mmax_list,
            grid_resolution,

            num_sphere_samples,

            edge_channels,
            use_atom_edge_embedding,
            share_atom_edge_embedding,
            use_m_share_rad,
            distance_function,
            num_distance_basis,

            attn_activation,
            use_s2_act_attn,
            use_attn_renorm,
            ffn_activation,
            use_gate_act,
            use_grid_mlp,
            use_sep_s2_act,

            alpha_drop,
            drop_path_rate,
            proj_drop,

            weight_init,

            enforce_max_neighbors_strictly,

            avg_num_nodes,
            avg_degree,

            use_energy_lin_ref,
            load_energy_lin_ref,
            activation_checkpoint,
        )

        # for denoising position
        self.use_force_encoding = use_force_encoding
        self.use_noise_schedule_sigma_encoding = use_noise_schedule_sigma_encoding
        self.use_denoising_energy = use_denoising_energy
        self.use_denoising_stress = use_denoising_stress

        # for denoising position, encode node-wise forces as node features
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=max(self.lmax_list), p=1)
        self.force_embedding = SO3_LinearV2(
            in_features=1,
            out_features=self.sphere_channels,
            lmax=max(self.lmax_list)
        )

        if self.use_noise_schedule_sigma_encoding:
            self.noise_schedule_sigma_embedding = torch.nn.Linear(
                in_features=1,
                out_features=self.sphere_channels
            )

        self.apply(partial(eqv2_init_weights, weight_init=self.weight_init))

    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        self.batch_size = len(data.natoms)
        self.dtype = data.pos.dtype
        self.device = data.pos.device
        num_atoms = len(data.atomic_numbers)
        atomic_numbers = data.atomic_numbers.long()
        graph = self.generate_graph(
            data,
            enforce_max_neighbors_strictly=self.enforce_max_neighbors_strictly,
        )

        data_batch = data.batch
        if gp_utils.initialized():
            (
                atomic_numbers,
                data_batch,
                node_offset,
                edge_index,
                edge_distance,
                edge_distance_vec,
            ) = self._init_gp_partitions(
                graph.atomic_numbers_full,
                graph.batch_full,
                graph.edge_index,
                graph.edge_distance,
                graph.edge_distance_vec,
            )
            graph.node_offset = node_offset
            graph.edge_index = edge_index
            graph.edge_distance = edge_distance
            graph.edge_distance_vec = edge_distance_vec

        ###############################################################
        # Entering Graph Parallel Region
        # after this point, if using gp, then node, edge tensors are split
        # across the graph parallel ranks, some full tensors such as
        # atomic_numbers_full are required because we need to index into the
        # full graph when computing edge embeddings or reducing nodes from neighbors
        #
        # all tensors that do not have the suffix "_full" refer to the partial tensors.
        # if not using gp, the full values are equal to the partial values
        # ie: atomic_numbers_full == atomic_numbers
        ###############################################################

        ###############################################################
        # Initialize data structures
        ###############################################################

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = self._init_edge_rot_mat(
            data, graph.edge_index, graph.edge_distance_vec
        )

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        x = SO3_Embedding(
            len(atomic_numbers),
            self.lmax_list,
            self.sphere_channels,
            self.device,
            self.dtype,
        )

        offset_res = 0
        offset = 0
        # Initialize the l = 0, m = 0 coefficients for each resolution
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)
            else:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)[
                    :, offset : offset + self.sphere_channels
                ]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        ##################
        ### DeNS Start ###
        ##################

        # Node-wise force encoding during denoising positions
        force_embedding = SO3_Embedding(num_atoms, self.lmax_list, 1, self.device, self.dtype)
        if hasattr(data, "denoising_pos_forward") and data.denoising_pos_forward:
            assert hasattr(data, "forces")
            force_data = data.forces
            force_sh = o3.spherical_harmonics(
                l=self.irreps_sh,
                x=force_data,
                normalize=True,
                normalization="component"
            )
            force_sh = force_sh.view(num_atoms, (max(self.lmax_list) + 1) ** 2, 1)
            force_norm = force_data.norm(dim=-1, keepdim=True)
            if hasattr(data, "noise_mask"):
                noise_mask_tensor = data.noise_mask.view(-1, 1, 1)
                force_sh = force_sh * noise_mask_tensor
        else:
            force_sh = torch.zeros((num_atoms, (max(self.lmax_list) + 1) ** 2, 1), dtype=data.pos.dtype, device=data.pos.device)
            force_norm = torch.zeros((num_atoms, 1), dtype=data.pos.dtype, device=data.pos.device)

        if not self.use_force_encoding:
            # for ablation study, we enforce the force encoding to be zero.
            force_sh = torch.zeros((num_atoms, (max(self.lmax_list) + 1) ** 2, 1), dtype=data.pos.dtype, device=data.pos.device)
            force_norm = torch.zeros((num_atoms, 1), dtype=data.pos.dtype, device=data.pos.device)

        force_norm = force_norm.view(-1, 1, 1)
        force_norm = force_norm / math.sqrt(3.0)  # since we use `component` normalization
        force_embedding.embedding = force_sh * force_norm

        force_embedding = self.force_embedding(force_embedding)
        x.embedding = x.embedding + force_embedding.embedding

        # noise schedule sigma encoding
        if self.use_noise_schedule_sigma_encoding:
            if hasattr(data, "denoising_pos_forward") and data.denoising_pos_forward:
                assert hasattr(data, "sigmas")
                sigmas = data.sigmas
            else:
                sigmas = torch.zeros((num_atoms, 1), dtype=data.pos.dtype, device=data.pos.device)
            noise_schedule_sigma_enbedding = self.noise_schedule_sigma_embedding(sigmas)
            x.embedding[:, 0, :] = x.embedding[:, 0, :] + noise_schedule_sigma_enbedding

        ##################
        ###  DeNS End  ###
        ##################

        # Edge encoding (distance and atom edge)
        graph.edge_distance = self.distance_expansion(graph.edge_distance)
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            source_element = graph.atomic_numbers_full[
                graph.edge_index[0]
            ]  # Source atom atomic number
            target_element = graph.atomic_numbers_full[
                graph.edge_index[1]
            ]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            graph.edge_distance = torch.cat(
                (graph.edge_distance, source_embedding, target_embedding), dim=1
            )

        # Edge-degree embedding
        edge_degree = self.edge_degree_embedding(
            graph.atomic_numbers_full,
            graph.edge_distance,
            graph.edge_index,
            len(atomic_numbers),
            graph.node_offset,
        )
        x.embedding = x.embedding + edge_degree.embedding

        ###############################################################
        # Update spherical node embeddings
        ###############################################################

        for i in range(self.num_layers):
            if self.activation_checkpoint:
                x = torch.utils.checkpoint.checkpoint(
                    self.blocks[i],
                    x,  # SO3_Embedding
                    graph.atomic_numbers_full,
                    graph.edge_distance,
                    graph.edge_index,
                    data_batch,  # for GraphDropPath
                    graph.node_offset,
                    use_reentrant=not self.training,
                )
            else:
                x = self.blocks[i](
                    x,  # SO3_Embedding
                    graph.atomic_numbers_full,
                    graph.edge_distance,
                    graph.edge_index,
                    batch=data_batch,  # for GraphDropPath
                    node_offset=graph.node_offset,
                )

        # Final layer norm
        x.embedding = self.norm(x.embedding)

        return {"node_embedding": x, "graph": graph}

@registry.register_model("equiformer_v2_dens_energy_head")
class DeNSEnergyHead(torch.nn.Module, HeadInterface):
    def __init__(self, backbone, reduce: str = "sum"):
        super().__init__()
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
        self.use_denoising_energy = backbone.use_denoising_energy

    def forward(self, data: Batch, emb: dict[str, torch.Tensor | GraphData]):
        node_energy = self.energy_block(emb["node_embedding"])
        node_energy = node_energy.embedding.narrow(1, 0, 1)
        if gp_utils.initialized():
            node_energy = gp_utils.gather_from_model_parallel_region(node_energy, dim=0)
        energy = torch.zeros(
            len(data.natoms),
            device=node_energy.device,
            dtype=node_energy.dtype,
        )

        energy.index_add_(0, data.batch, node_energy.view(-1))

        if (hasattr(data, "denoising_pos_forward") and
            data.denoising_pos_forward and
            not self.use_denoising_energy):
            energy = energy * 0.0

        if self.reduce == "sum":
            return {"energy": energy / self.avg_num_nodes}
        elif self.reduce == "mean":
            return {"energy": energy / data.natoms}
        else:
            raise ValueError(
                f"reduce can only be sum or mean, user provided: {self.reduce}"
            )

@registry.register_model("equiformer_v2_dens_force_head")
class DeNSForceHead(torch.nn.Module, HeadInterface):
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

        self.denoising_pos_block = SO2EquivariantGraphAttention(
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
            denoising_pos_vec = torch.utils.checkpoint.checkpoint(
                self.denoising_pos_block,
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
            denoising_pos_vec = self.denoising_pos_block(
                emb["node_embedding"],
                emb["graph"].atomic_numbers_full,
                emb["graph"].edge_distance,
                emb["graph"].edge_index,
                node_offset=emb["graph"].node_offset,
            )
        forces = forces.embedding.narrow(1, 1, 3)
        forces = forces.view(-1, 3).contiguous()
        denoising_pos_vec = denoising_pos_vec.embedding.narrow(1, 1, 3)
        denoising_pos_vec = denoising_pos_vec.view(-1, 3)
        if gp_utils.initialized():
            forces = gp_utils.gather_from_model_parallel_region(forces, dim=0)
            denoising_pos_vec = gp_utils.gather_from_model_parallel_region(denoising_pos_vec, dim=0)

        if hasattr(data, "denoising_pos_forward") and data.denoising_pos_forward:
            if hasattr(data, "noise_mask"):
                noise_mask_tensor = data.noise_mask.view(-1, 1)
                forces = denoising_pos_vec * noise_mask_tensor + forces * (~noise_mask_tensor)
            else:
                forces = denoising_pos_vec + 0 * forces
        else:
            forces = 0 * denoising_pos_vec + forces

        return {"forces": forces}


@registry.register_model("dens_rank2_symmetric_head")
class DeNSRank2Head(Rank2SymmetricTensorHead):
    def __init__(self, backbone, *args, **kwargs):
        super().__init__(backbone, *args, **kwargs)
        self.use_denoising_stress = backbone.use_denoising_stress

    def forward(self, data: Batch, emb: dict[str, torch.Tensor]):
        output = super().forward(data, emb)
        if (hasattr(data, "denoising_pos_forward") and
        data.denoising_pos_forward and
        not self.use_denoising_stress):
            for k in output:
                output[k] = output[k] * 0.0
        return output
