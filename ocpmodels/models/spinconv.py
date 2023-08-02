"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import logging
import math
import time
from math import pi as PI

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_scatter import scatter

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad
from ocpmodels.models.base import BaseModel

try:
    from e3nn import o3
    from e3nn.o3 import FromS2Grid
except Exception:
    pass


@registry.register_model("spinconv")
class spinconv(BaseModel):
    def __init__(
        self,
        num_atoms: int,  # not used
        bond_feat_dim: int,  # not used
        num_targets: int,
        use_pbc: bool = True,
        regress_forces: bool = True,
        otf_graph: bool = False,
        hidden_channels: int = 32,
        mid_hidden_channels: int = 200,
        num_interactions: int = 1,
        num_basis_functions: int = 200,
        basis_width_scalar: float = 1.0,
        max_num_neighbors: int = 20,
        sphere_size_lat: int = 15,
        sphere_size_long: int = 9,
        cutoff: float = 10.0,
        distance_block_scalar_max: float = 2.0,
        max_num_elements: int = 90,
        embedding_size: int = 32,
        show_timing_info: bool = False,
        sphere_message: str = "fullconv",  # message block sphere representation
        output_message: str = "fullconv",  # output block sphere representation
        lmax: bool = False,
        force_estimator: str = "random",
        model_ref_number: int = 0,
        readout: str = "add",
        num_rand_rotations: int = 5,
        scale_distances: bool = True,
    ) -> None:
        super(spinconv, self).__init__()

        self.num_targets = num_targets
        self.num_random_rotations = num_rand_rotations
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.cutoff = cutoff
        self.otf_graph = otf_graph
        self.show_timing_info = show_timing_info
        self.max_num_elements = max_num_elements
        self.mid_hidden_channels = mid_hidden_channels
        self.sphere_size_lat = sphere_size_lat
        self.sphere_size_long = sphere_size_long
        self.num_atoms = 0
        self.hidden_channels = hidden_channels
        self.embedding_size = embedding_size
        self.max_num_neighbors = self.max_neighbors = max_num_neighbors
        self.sphere_message = sphere_message
        self.output_message = output_message
        self.force_estimator = force_estimator
        self.num_basis_functions = num_basis_functions
        self.distance_block_scalar_max = distance_block_scalar_max
        self.grad_forces = False
        self.num_embedding_basis = 8
        self.lmax = lmax
        self.scale_distances = scale_distances
        self.basis_width_scalar = basis_width_scalar

        if self.sphere_message in ["spharm", "rotspharmroll", "rotspharmwd"]:
            assert self.lmax, "lmax must be defined for spherical harmonics"
        if self.output_message in ["spharm", "rotspharmroll", "rotspharmwd"]:
            assert self.lmax, "lmax must be defined for spherical harmonics"

        # variables used for display purposes
        self.counter = 0
        self.start_time: float = time.time()
        self.total_time: float = 0.0
        self.model_ref_number = model_ref_number

        if self.force_estimator == "grad":
            self.grad_forces = True

        # self.act = ShiftedSoftplus()
        self.act = Swish()

        self.distance_expansion_forces: GaussianSmearing = GaussianSmearing(
            0.0,
            cutoff,
            num_basis_functions,
            basis_width_scalar,
        )

        # Weights for message initialization
        self.embeddingblock2: EmbeddingBlock = EmbeddingBlock(
            self.mid_hidden_channels,
            self.hidden_channels,
            self.mid_hidden_channels,
            self.embedding_size,
            self.num_embedding_basis,
            self.max_num_elements,
            self.act,
        )
        self.distfc1: nn.Linear = nn.Linear(
            self.mid_hidden_channels, self.mid_hidden_channels
        )
        self.distfc2: nn.Linear = nn.Linear(
            self.mid_hidden_channels, self.mid_hidden_channels
        )

        self.dist_block: DistanceBlock = DistanceBlock(
            self.num_basis_functions,
            self.mid_hidden_channels,
            self.max_num_elements,
            self.distance_block_scalar_max,
            self.distance_expansion_forces,
            self.scale_distances,
        )

        self.message_blocks = ModuleList()
        for _ in range(num_interactions):
            block = MessageBlock(
                hidden_channels,
                hidden_channels,
                mid_hidden_channels,
                embedding_size,
                self.sphere_size_lat,
                self.sphere_size_long,
                self.max_num_elements,
                self.sphere_message,
                self.act,
                self.lmax,
            )
            self.message_blocks.append(block)

        self.energyembeddingblock = EmbeddingBlock(
            hidden_channels,
            1,
            mid_hidden_channels,
            embedding_size,
            8,
            self.max_num_elements,
            self.act,
        )

        if force_estimator == "random":
            self.force_output_block = ForceOutputBlock(
                hidden_channels,
                2,
                mid_hidden_channels,
                embedding_size,
                self.sphere_size_lat,
                self.sphere_size_long,
                self.max_num_elements,
                self.output_message,
                self.act,
                self.lmax,
            )

    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        self.device = data.pos.device
        self.num_atoms = len(data.batch)
        self.batch_size = len(data.natoms)

        pos = data.pos
        if self.regress_forces:
            pos = pos.requires_grad_(True)

        (
            edge_index,
            edge_distance,
            edge_distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(data)

        edge_index, edge_distance, edge_distance_vec = self._filter_edges(
            edge_index,
            edge_distance,
            edge_distance_vec,
            self.max_num_neighbors,
        )

        outputs = self._forward_helper(
            data, edge_index, edge_distance, edge_distance_vec
        )
        if self.show_timing_info is True:
            torch.cuda.synchronize()
            logging.info(
                "Memory: {}\t{}\t{}".format(
                    len(edge_index[0]),
                    torch.cuda.memory_allocated()
                    / (1000 * len(edge_index[0])),
                    torch.cuda.max_memory_allocated() / 1000000,
                )
            )

        return outputs

    # restructure forward helper for conditional grad
    def _forward_helper(
        self, data, edge_index, edge_distance, edge_distance_vec
    ):
        ###############################################################
        # Initialize messages
        ###############################################################

        source_element = data.atomic_numbers[edge_index[0, :]].long()
        target_element = data.atomic_numbers[edge_index[1, :]].long()

        x_dist = self.dist_block(edge_distance, source_element, target_element)

        x = x_dist
        x = self.distfc1(x)
        x = self.act(x)
        x = self.distfc2(x)
        x = self.act(x)
        x = self.embeddingblock2(x, source_element, target_element)

        ###############################################################
        # Update messages using block interactions
        ###############################################################

        edge_rot_mat = self._init_edge_rot_mat(
            data, edge_index, edge_distance_vec
        )
        (
            proj_edges_index,
            proj_edges_delta,
            proj_edges_src_index,
        ) = self._project2D_edges_init(
            edge_rot_mat, edge_index, edge_distance_vec
        )

        for block_index, interaction in enumerate(self.message_blocks):
            x_out = interaction(
                x,
                x_dist,
                source_element,
                target_element,
                proj_edges_index,
                proj_edges_delta,
                proj_edges_src_index,
            )

            if block_index > 0:
                x = x + x_out
            else:
                x = x_out

        ###############################################################
        # Decoder
        # Compute the forces and energies from the messages
        ###############################################################
        assert self.force_estimator in ["random", "grad"]

        energy = scatter(x, edge_index[1], dim=0, dim_size=data.num_nodes) / (
            self.max_num_neighbors / 2.0 + 1.0
        )
        atomic_numbers = data.atomic_numbers.long()
        energy = self.energyembeddingblock(
            energy, atomic_numbers, atomic_numbers
        )
        energy = scatter(energy, data.batch, dim=0)

        if self.regress_forces:
            if self.force_estimator == "grad":
                forces = -1 * (
                    torch.autograd.grad(
                        energy,
                        data.pos,
                        grad_outputs=torch.ones_like(energy),
                        create_graph=True,
                    )[0]
                )
            if self.force_estimator == "random":
                forces = self._compute_forces_random_rotations(
                    x,
                    self.num_random_rotations,
                    data.atomic_numbers.long(),
                    edge_index,
                    edge_distance_vec,
                    data.batch,
                )

        if not self.regress_forces:
            return energy
        else:
            return energy, forces

    def _compute_forces_random_rotations(
        self,
        x,
        num_random_rotations: int,
        target_element,
        edge_index,
        edge_distance_vec,
        batch,
    ) -> torch.Tensor:
        # Compute the forces and energy by randomly rotating the system and taking the average

        device = x.device

        rot_mat_x = torch.zeros(3, 3, device=device)
        rot_mat_x[0][0] = 1.0
        rot_mat_x[1][1] = 1.0
        rot_mat_x[2][2] = 1.0

        rot_mat_y = torch.zeros(3, 3, device=device)
        rot_mat_y[0][1] = 1.0
        rot_mat_y[1][0] = -1.0
        rot_mat_y[2][2] = 1.0

        rot_mat_z = torch.zeros(3, 3, device=device)
        rot_mat_z[0][2] = 1.0
        rot_mat_z[1][1] = 1.0
        rot_mat_z[2][0] = -1.0

        rot_mat_x = rot_mat_x.view(-1, 3, 3).repeat(self.num_atoms, 1, 1)
        rot_mat_y = rot_mat_y.view(-1, 3, 3).repeat(self.num_atoms, 1, 1)
        rot_mat_z = rot_mat_z.view(-1, 3, 3).repeat(self.num_atoms, 1, 1)

        # compute the random rotations
        random_rot_mat = self._random_rot_mat(
            self.num_atoms * num_random_rotations, device
        )
        random_rot_mat = random_rot_mat.view(
            num_random_rotations, self.num_atoms, 3, 3
        )

        # the first matrix is the identity with the rest being random
        # atom_rot_mat = torch.cat([torch.eye(3, device=device).view(1, 1, 3, 3).repeat(1, self.num_atoms, 1, 1), random_rot_mat], dim=0)
        # or they are all random
        atom_rot_mat = random_rot_mat

        forces = torch.zeros(self.num_atoms, 3, device=device)

        for rot_index in range(num_random_rotations):
            rot_mat_x_perturb = torch.bmm(rot_mat_x, atom_rot_mat[rot_index])
            rot_mat_y_perturb = torch.bmm(rot_mat_y, atom_rot_mat[rot_index])
            rot_mat_z_perturb = torch.bmm(rot_mat_z, atom_rot_mat[rot_index])

            # project neighbors using the random rotations
            (
                proj_nodes_index_x,
                proj_nodes_delta_x,
                proj_nodes_src_index_x,
            ) = self._project2D_nodes_init(
                rot_mat_x_perturb, edge_index, edge_distance_vec
            )
            (
                proj_nodes_index_y,
                proj_nodes_delta_y,
                proj_nodes_src_index_y,
            ) = self._project2D_nodes_init(
                rot_mat_y_perturb, edge_index, edge_distance_vec
            )
            (
                proj_nodes_index_z,
                proj_nodes_delta_z,
                proj_nodes_src_index_z,
            ) = self._project2D_nodes_init(
                rot_mat_z_perturb, edge_index, edge_distance_vec
            )

            # estimate the force in each perpendicular direction
            force_x = self.force_output_block(
                x,
                self.num_atoms,
                target_element,
                proj_nodes_index_x,
                proj_nodes_delta_x,
                proj_nodes_src_index_x,
            )
            force_y = self.force_output_block(
                x,
                self.num_atoms,
                target_element,
                proj_nodes_index_y,
                proj_nodes_delta_y,
                proj_nodes_src_index_y,
            )
            force_z = self.force_output_block(
                x,
                self.num_atoms,
                target_element,
                proj_nodes_index_z,
                proj_nodes_delta_z,
                proj_nodes_src_index_z,
            )
            forces_perturb = torch.cat(
                [force_x[:, 0:1], force_y[:, 0:1], force_z[:, 0:1]], dim=1
            )

            # rotate the predicted forces back into the global reference frame
            rot_mat_inv = torch.transpose(rot_mat_x_perturb, 1, 2)
            forces_perturb = torch.bmm(
                rot_mat_inv, forces_perturb.view(-1, 3, 1)
            ).view(-1, 3)

            forces = forces + forces_perturb

        forces = forces / (num_random_rotations)

        return forces

    def _filter_edges(
        self,
        edge_index,
        edge_distance,
        edge_distance_vec,
        max_num_neighbors: int,
    ):
        # Remove edges that aren't within the closest max_num_neighbors from either the target or source atom.
        # This ensures all edges occur in pairs, i.e., if X -> Y exists then Y -> X is included.
        # However, if both X -> Y and Y -> X don't both exist in the original list, this isn't guaranteed.
        # Since some edges may have exactly the same distance, this function is not deterministic
        device = edge_index.device
        length = len(edge_distance)

        # Assuming the edges are consecutive based on the target index
        target_node_index, neigh_count = torch.unique_consecutive(
            edge_index[1], return_counts=True
        )
        max_neighbors = torch.max(neigh_count)

        # handle special case where an atom doesn't have any neighbors
        target_neigh_count = torch.zeros(self.num_atoms, device=device).long()
        target_neigh_count.index_copy_(
            0, target_node_index.long(), neigh_count
        )

        # Create a list of edges for each atom
        index_offset = (
            torch.cumsum(target_neigh_count, dim=0) - target_neigh_count
        )
        neigh_index = torch.arange(length, device=device)
        neigh_index = neigh_index - index_offset[edge_index[1]]

        edge_map_index = (edge_index[1] * max_neighbors + neigh_index).long()
        target_lookup = (
            torch.zeros(self.num_atoms * max_neighbors, device=device) - 1
        ).long()
        target_lookup.index_copy_(
            0, edge_map_index, torch.arange(length, device=device).long()
        )

        # Get the length of each edge
        distance_lookup = (
            torch.zeros(self.num_atoms * max_neighbors, device=device)
            + 1000000.0
        )
        distance_lookup.index_copy_(0, edge_map_index, edge_distance)
        distance_lookup = distance_lookup.view(self.num_atoms, max_neighbors)

        # Sort the distances
        distance_sorted_no_op, indices = torch.sort(distance_lookup, dim=1)

        # Create a hash that maps edges that go from X -> Y and Y -> X in the same bin
        edge_index_min, no_op = torch.min(edge_index, dim=0)
        edge_index_max, no_op = torch.max(edge_index, dim=0)
        edge_index_hash = edge_index_min * self.num_atoms + edge_index_max
        edge_count_start = torch.zeros(
            self.num_atoms * self.num_atoms, device=device
        )
        edge_count_start.index_add_(
            0, edge_index_hash, torch.ones(len(edge_index_hash), device=device)
        )

        # Find index into the original edge_index
        indices = indices + (
            torch.arange(len(indices), device=device) * max_neighbors
        ).view(-1, 1).repeat(1, max_neighbors)
        indices = indices.view(-1)
        target_lookup_sorted = (
            torch.zeros(self.num_atoms * max_neighbors, device=device) - 1
        ).long()
        target_lookup_sorted = target_lookup[indices]
        target_lookup_sorted = target_lookup_sorted.view(
            self.num_atoms, max_neighbors
        )

        # Select the closest max_num_neighbors for each edge and remove the unused entries
        target_lookup_below_thres = (
            target_lookup_sorted[:, 0:max_num_neighbors].contiguous().view(-1)
        )
        target_lookup_below_thres = target_lookup_below_thres.view(-1)
        mask_unused = target_lookup_below_thres.ge(0)
        target_lookup_below_thres = torch.masked_select(
            target_lookup_below_thres, mask_unused
        )

        # Find edges that are used at least once and create a mask to keep
        edge_count = torch.zeros(
            self.num_atoms * self.num_atoms, device=device
        )
        edge_count.index_add_(
            0,
            edge_index_hash[target_lookup_below_thres],
            torch.ones(len(target_lookup_below_thres), device=device),
        )
        edge_count_mask = edge_count.ne(0)
        edge_keep = edge_count_mask[edge_index_hash]

        # Finally remove all edges that are too long in distance as indicated by the mask
        edge_index_mask = edge_keep.view(1, -1).repeat(2, 1)
        edge_index = torch.masked_select(edge_index, edge_index_mask).view(
            2, -1
        )
        edge_distance = torch.masked_select(edge_distance, edge_keep)
        edge_distance_vec_mask = edge_keep.view(-1, 1).repeat(1, 3)
        edge_distance_vec = torch.masked_select(
            edge_distance_vec, edge_distance_vec_mask
        ).view(-1, 3)

        return edge_index, edge_distance, edge_distance_vec

    def _random_rot_mat(self, num_matrices: int, device) -> torch.Tensor:
        ang_a = 2.0 * math.pi * torch.rand(num_matrices, device=device)
        ang_b = 2.0 * math.pi * torch.rand(num_matrices, device=device)
        ang_c = 2.0 * math.pi * torch.rand(num_matrices, device=device)

        cos_a = torch.cos(ang_a)
        cos_b = torch.cos(ang_b)
        cos_c = torch.cos(ang_c)
        sin_a = torch.sin(ang_a)
        sin_b = torch.sin(ang_b)
        sin_c = torch.sin(ang_c)

        rot_a = (
            torch.eye(3, device=device)
            .view(1, 3, 3)
            .repeat(num_matrices, 1, 1)
        )
        rot_b = (
            torch.eye(3, device=device)
            .view(1, 3, 3)
            .repeat(num_matrices, 1, 1)
        )
        rot_c = (
            torch.eye(3, device=device)
            .view(1, 3, 3)
            .repeat(num_matrices, 1, 1)
        )

        rot_a[:, 1, 1] = cos_a
        rot_a[:, 1, 2] = sin_a
        rot_a[:, 2, 1] = -sin_a
        rot_a[:, 2, 2] = cos_a

        rot_b[:, 0, 0] = cos_b
        rot_b[:, 0, 2] = -sin_b
        rot_b[:, 2, 0] = sin_b
        rot_b[:, 2, 2] = cos_b

        rot_c[:, 0, 0] = cos_c
        rot_c[:, 0, 1] = sin_c
        rot_c[:, 1, 0] = -sin_c
        rot_c[:, 1, 1] = cos_c

        return torch.bmm(torch.bmm(rot_a, rot_b), rot_c)

    def _init_edge_rot_mat(
        self, data, edge_index, edge_distance_vec
    ) -> torch.Tensor:
        device = data.pos.device
        num_atoms = len(data.batch)

        edge_vec_0 = edge_distance_vec
        edge_vec_0_distance = torch.sqrt(torch.sum(edge_vec_0**2, dim=1))

        if torch.min(edge_vec_0_distance) < 0.0001:
            logging.error(
                "Error edge_vec_0_distance: {}".format(
                    torch.min(edge_vec_0_distance)
                )
            )
            (minval, minidx) = torch.min(edge_vec_0_distance, 0)
            logging.error(
                "Error edge_vec_0_distance: {} {} {} {} {}".format(
                    minidx,
                    edge_index[0, minidx],
                    edge_index[1, minidx],
                    data.pos[edge_index[0, minidx]],
                    data.pos[edge_index[1, minidx]],
                )
            )

        avg_vector = torch.zeros(num_atoms, 3, device=device)
        weight = 0.5 * (
            torch.cos(edge_vec_0_distance * PI / self.cutoff) + 1.0
        )
        avg_vector.index_add_(
            0, edge_index[1, :], edge_vec_0 * weight.view(-1, 1).expand(-1, 3)
        )

        edge_vec_2 = avg_vector[edge_index[1, :]] + 0.0001
        edge_vec_2_distance = torch.sqrt(torch.sum(edge_vec_2**2, dim=1))

        if torch.min(edge_vec_2_distance) < 0.000001:
            logging.error(
                "Error edge_vec_2_distance: {}".format(
                    torch.min(edge_vec_2_distance)
                )
            )

        norm_x = edge_vec_0 / (edge_vec_0_distance.view(-1, 1))
        norm_0_2 = edge_vec_2 / (edge_vec_2_distance.view(-1, 1))
        norm_z = torch.cross(norm_x, norm_0_2, dim=1)
        norm_z = norm_z / (
            torch.sqrt(torch.sum(norm_z**2, dim=1, keepdim=True)) + 0.0000001
        )
        norm_y = torch.cross(norm_x, norm_z, dim=1)
        norm_y = norm_y / (
            torch.sqrt(torch.sum(norm_y**2, dim=1, keepdim=True)) + 0.0000001
        )

        norm_x = norm_x.view(-1, 3, 1)
        norm_y = norm_y.view(-1, 3, 1)
        norm_z = norm_z.view(-1, 3, 1)

        edge_rot_mat_inv = torch.cat([norm_x, norm_y, norm_z], dim=2)
        edge_rot_mat = torch.transpose(edge_rot_mat_inv, 1, 2)

        return edge_rot_mat

    def _project2D_edges_init(self, rot_mat, edge_index, edge_distance_vec):
        torch.set_printoptions(sci_mode=False)
        length = len(edge_distance_vec)
        device = edge_distance_vec.device

        # Assuming the edges are consecutive based on the target index
        target_node_index, neigh_count = torch.unique_consecutive(
            edge_index[1], return_counts=True
        )
        max_neighbors = torch.max(neigh_count)
        target_neigh_count = torch.zeros(self.num_atoms, device=device).long()
        target_neigh_count.index_copy_(
            0, target_node_index.long(), neigh_count
        )

        index_offset = (
            torch.cumsum(target_neigh_count, dim=0) - target_neigh_count
        )
        neigh_index = torch.arange(length, device=device)
        neigh_index = neigh_index - index_offset[edge_index[1]]

        edge_map_index = edge_index[1] * max_neighbors + neigh_index
        target_lookup = (
            torch.zeros(self.num_atoms * max_neighbors, device=device) - 1
        ).long()
        target_lookup.index_copy_(
            0,
            edge_map_index.long(),
            torch.arange(length, device=device).long(),
        )
        target_lookup = target_lookup.view(self.num_atoms, max_neighbors)

        # target_lookup - For each target node, a list of edge indices
        # target_neigh_count - number of neighbors for each target node
        source_edge = target_lookup[edge_index[0]]
        target_edge = (
            torch.arange(length, device=device)
            .long()
            .view(-1, 1)
            .repeat(1, max_neighbors)
        )

        source_edge = source_edge.view(-1)
        target_edge = target_edge.view(-1)

        mask_unused = source_edge.ge(0)
        source_edge = torch.masked_select(source_edge, mask_unused)
        target_edge = torch.masked_select(target_edge, mask_unused)

        return self._project2D_init(
            source_edge, target_edge, rot_mat, edge_distance_vec
        )

    def _project2D_nodes_init(self, rot_mat, edge_index, edge_distance_vec):
        torch.set_printoptions(sci_mode=False)
        length = len(edge_distance_vec)
        device = edge_distance_vec.device

        target_node = edge_index[1]
        source_edge = torch.arange(length, device=device)

        return self._project2D_init(
            source_edge, target_node, rot_mat, edge_distance_vec
        )

    def _project2D_init(
        self, source_edge, target_edge, rot_mat, edge_distance_vec
    ):
        edge_distance_norm = F.normalize(edge_distance_vec)
        source_edge_offset = edge_distance_norm[source_edge]

        source_edge_offset_rot = torch.bmm(
            rot_mat[target_edge], source_edge_offset.view(-1, 3, 1)
        )

        source_edge_X = torch.atan2(
            source_edge_offset_rot[:, 1], source_edge_offset_rot[:, 2]
        ).view(-1)

        # source_edge_X ranges from -pi to pi
        source_edge_X = (source_edge_X + math.pi) / (2.0 * math.pi)

        # source_edge_Y ranges from -1 to 1
        source_edge_Y = source_edge_offset_rot[:, 0].view(-1)
        source_edge_Y = torch.clamp(source_edge_Y, min=-1.0, max=1.0)
        source_edge_Y = (source_edge_Y.asin() + (math.pi / 2.0)) / (
            math.pi
        )  # bin by angle
        # source_edge_Y = (source_edge_Y + 1.0) / 2.0 # bin by sin
        source_edge_Y = 0.99 * (source_edge_Y) + 0.005

        source_edge_X = source_edge_X * self.sphere_size_long
        source_edge_Y = source_edge_Y * (
            self.sphere_size_lat - 1.0
        )  # not circular so pad by one

        source_edge_X_0 = torch.floor(source_edge_X).long()
        source_edge_X_del = source_edge_X - source_edge_X_0
        source_edge_X_0 = source_edge_X_0 % self.sphere_size_long
        source_edge_X_1 = (source_edge_X_0 + 1) % self.sphere_size_long

        source_edge_Y_0 = torch.floor(source_edge_Y).long()
        source_edge_Y_del = source_edge_Y - source_edge_Y_0
        source_edge_Y_0 = source_edge_Y_0 % self.sphere_size_lat
        source_edge_Y_1 = (source_edge_Y_0 + 1) % self.sphere_size_lat

        # Compute the values needed to bilinearly splat the values onto the spheres
        index_0_0 = (
            target_edge * self.sphere_size_lat * self.sphere_size_long
            + source_edge_Y_0 * self.sphere_size_long
            + source_edge_X_0
        )
        index_0_1 = (
            target_edge * self.sphere_size_lat * self.sphere_size_long
            + source_edge_Y_0 * self.sphere_size_long
            + source_edge_X_1
        )
        index_1_0 = (
            target_edge * self.sphere_size_lat * self.sphere_size_long
            + source_edge_Y_1 * self.sphere_size_long
            + source_edge_X_0
        )
        index_1_1 = (
            target_edge * self.sphere_size_lat * self.sphere_size_long
            + source_edge_Y_1 * self.sphere_size_long
            + source_edge_X_1
        )

        delta_0_0 = (1.0 - source_edge_X_del) * (1.0 - source_edge_Y_del)
        delta_0_1 = (source_edge_X_del) * (1.0 - source_edge_Y_del)
        delta_1_0 = (1.0 - source_edge_X_del) * (source_edge_Y_del)
        delta_1_1 = (source_edge_X_del) * (source_edge_Y_del)

        index_0_0 = index_0_0.view(1, -1)
        index_0_1 = index_0_1.view(1, -1)
        index_1_0 = index_1_0.view(1, -1)
        index_1_1 = index_1_1.view(1, -1)

        # NaNs otherwise
        if self.grad_forces:
            with torch.no_grad():
                delta_0_0 = delta_0_0.view(1, -1)
                delta_0_1 = delta_0_1.view(1, -1)
                delta_1_0 = delta_1_0.view(1, -1)
                delta_1_1 = delta_1_1.view(1, -1)
        else:
            delta_0_0 = delta_0_0.view(1, -1)
            delta_0_1 = delta_0_1.view(1, -1)
            delta_1_0 = delta_1_0.view(1, -1)
            delta_1_1 = delta_1_1.view(1, -1)

        return (
            torch.cat([index_0_0, index_0_1, index_1_0, index_1_1]),
            torch.cat([delta_0_0, delta_0_1, delta_1_0, delta_1_1]),
            source_edge,
        )

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MessageBlock(torch.nn.Module):
    def __init__(
        self,
        in_hidden_channels: int,
        out_hidden_channels: int,
        mid_hidden_channels: int,
        embedding_size: int,
        sphere_size_lat: int,
        sphere_size_long: int,
        max_num_elements: int,
        sphere_message: str,
        act,
        lmax,
    ) -> None:
        super(MessageBlock, self).__init__()
        self.in_hidden_channels = in_hidden_channels
        self.out_hidden_channels = out_hidden_channels
        self.act = act
        self.lmax = lmax
        self.embedding_size = embedding_size
        self.mid_hidden_channels = mid_hidden_channels
        self.sphere_size_lat = sphere_size_lat
        self.sphere_size_long = sphere_size_long
        self.sphere_message = sphere_message
        self.max_num_elements = max_num_elements
        self.num_embedding_basis = 8

        self.spinconvblock = SpinConvBlock(
            self.in_hidden_channels,
            self.mid_hidden_channels,
            self.sphere_size_lat,
            self.sphere_size_long,
            self.sphere_message,
            self.act,
            self.lmax,
        )

        self.embeddingblock1: EmbeddingBlock = EmbeddingBlock(
            self.mid_hidden_channels,
            self.mid_hidden_channels,
            self.mid_hidden_channels,
            self.embedding_size,
            self.num_embedding_basis,
            self.max_num_elements,
            self.act,
        )
        self.embeddingblock2: EmbeddingBlock = EmbeddingBlock(
            self.mid_hidden_channels,
            self.out_hidden_channels,
            self.mid_hidden_channels,
            self.embedding_size,
            self.num_embedding_basis,
            self.max_num_elements,
            self.act,
        )

        self.distfc1 = nn.Linear(
            self.mid_hidden_channels, self.mid_hidden_channels
        )
        self.distfc2 = nn.Linear(
            self.mid_hidden_channels, self.mid_hidden_channels
        )

    def forward(
        self,
        x,
        x_dist,
        source_element,
        target_element,
        proj_index,
        proj_delta,
        proj_src_index,
    ):
        out_size = len(x)

        x = self.spinconvblock(
            x, out_size, proj_index, proj_delta, proj_src_index
        )

        x = self.embeddingblock1(x, source_element, target_element)

        x_dist = self.distfc1(x_dist)
        x_dist = self.act(x_dist)
        x_dist = self.distfc2(x_dist)
        x = x + x_dist

        x = self.act(x)
        x = self.embeddingblock2(x, source_element, target_element)

        return x


class ForceOutputBlock(torch.nn.Module):
    def __init__(
        self,
        in_hidden_channels: int,
        out_hidden_channels: int,
        mid_hidden_channels: int,
        embedding_size: int,
        sphere_size_lat: int,
        sphere_size_long: int,
        max_num_elements: int,
        sphere_message: str,
        act,
        lmax,
    ) -> None:
        super(ForceOutputBlock, self).__init__()
        self.in_hidden_channels = in_hidden_channels
        self.out_hidden_channels = out_hidden_channels
        self.act = act
        self.lmax = lmax
        self.embedding_size = embedding_size
        self.mid_hidden_channels = mid_hidden_channels
        self.sphere_size_lat = sphere_size_lat
        self.sphere_size_long = sphere_size_long
        self.sphere_message = sphere_message
        self.max_num_elements = max_num_elements
        self.num_embedding_basis = 8

        self.spinconvblock: SpinConvBlock = SpinConvBlock(
            self.in_hidden_channels,
            self.mid_hidden_channels,
            self.sphere_size_lat,
            self.sphere_size_long,
            self.sphere_message,
            self.act,
            self.lmax,
        )

        self.block1: EmbeddingBlock = EmbeddingBlock(
            self.mid_hidden_channels,
            self.mid_hidden_channels,
            self.mid_hidden_channels,
            self.embedding_size,
            self.num_embedding_basis,
            self.max_num_elements,
            self.act,
        )
        self.block2: EmbeddingBlock = EmbeddingBlock(
            self.mid_hidden_channels,
            self.out_hidden_channels,
            self.mid_hidden_channels,
            self.embedding_size,
            self.num_embedding_basis,
            self.max_num_elements,
            self.act,
        )

    def forward(
        self,
        x,
        out_size,
        target_element,
        proj_index,
        proj_delta,
        proj_src_index,
    ):
        x = self.spinconvblock(
            x, out_size, proj_index, proj_delta, proj_src_index
        )

        x = self.block1(x, target_element, target_element)
        x = self.act(x)
        x = self.block2(x, target_element, target_element)

        return x


class SpinConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_hidden_channels: int,
        mid_hidden_channels: int,
        sphere_size_lat: int,
        sphere_size_long: int,
        sphere_message: str,
        act,
        lmax,
    ) -> None:
        super(SpinConvBlock, self).__init__()
        self.in_hidden_channels = in_hidden_channels
        self.mid_hidden_channels = mid_hidden_channels
        self.sphere_size_lat = sphere_size_lat
        self.sphere_size_long = sphere_size_long
        self.sphere_message = sphere_message
        self.act = act
        self.lmax = lmax
        self.num_groups = self.in_hidden_channels // 8

        self.ProjectLatLongSphere = ProjectLatLongSphere(
            sphere_size_lat, sphere_size_long
        )
        assert self.sphere_message in [
            "fullconv",
            "rotspharmwd",
        ]
        if self.sphere_message in ["rotspharmwd"]:
            self.sph_froms2grid = FromS2Grid(
                (self.sphere_size_lat, self.sphere_size_long), self.lmax
            )
            self.mlp = nn.Linear(
                self.in_hidden_channels * (self.lmax + 1) ** 2,
                self.mid_hidden_channels,
            )
            self.sphlength = (self.lmax + 1) ** 2
            rotx = torch.zeros(self.sphere_size_long) + (
                2 * math.pi / self.sphere_size_long
            )
            roty = torch.zeros(self.sphere_size_long)
            rotz = torch.zeros(self.sphere_size_long)

            self.wigner = []
            for xrot, yrot, zrot in zip(rotx, roty, rotz):
                _blocks = []
                for l_degree in range(self.lmax + 1):
                    _blocks.append(o3.wigner_D(l_degree, xrot, yrot, zrot))
                self.wigner.append(torch.block_diag(*_blocks))

        if self.sphere_message == "fullconv":
            padding = self.sphere_size_long // 2
            self.conv1 = nn.Conv1d(
                self.in_hidden_channels * self.sphere_size_lat,
                self.mid_hidden_channels,
                self.sphere_size_long,
                groups=self.in_hidden_channels // 8,
                padding=padding,
                padding_mode="circular",
            )
            self.pool = nn.AvgPool1d(sphere_size_long)

        self.GroupNorm = nn.GroupNorm(
            self.num_groups, self.mid_hidden_channels
        )

    def forward(self, x, out_size, proj_index, proj_delta, proj_src_index):
        x = self.ProjectLatLongSphere(
            x, out_size, proj_index, proj_delta, proj_src_index
        )
        if self.sphere_message == "rotspharmwd":
            sph_harm_calc = torch.zeros(
                ((x.shape[0], self.mid_hidden_channels)),
                device=x.device,
            )

            sph_harm = self.sph_froms2grid(x)
            sph_harm = sph_harm.view(-1, self.sphlength, 1)
            for wD_diag in self.wigner:
                wD_diag = wD_diag.to(x.device)
                sph_harm_calc += self.act(
                    self.mlp(sph_harm.reshape(x.shape[0], -1))
                )
                wd = wD_diag.view(1, self.sphlength, self.sphlength).expand(
                    len(x) * self.in_hidden_channels, -1, -1
                )
                sph_harm = torch.bmm(wd, sph_harm)
            x = sph_harm_calc

        if self.sphere_message in ["fullconv"]:
            x = x.view(
                -1,
                self.in_hidden_channels * self.sphere_size_lat,
                self.sphere_size_long,
            )
            x = self.conv1(x)
            x = self.act(x)
            # Pool in the longitudal direction
            x = self.pool(x[:, :, 0 : self.sphere_size_long])
            x = x.view(out_size, -1)

        x = self.GroupNorm(x)

        return x


class EmbeddingBlock(torch.nn.Module):
    def __init__(
        self,
        in_hidden_channels: int,
        out_hidden_channels: int,
        mid_hidden_channels: int,
        embedding_size: int,
        num_embedding_basis: int,
        max_num_elements: int,
        act,
    ) -> None:
        super(EmbeddingBlock, self).__init__()
        self.in_hidden_channels = in_hidden_channels
        self.out_hidden_channels = out_hidden_channels
        self.act = act
        self.embedding_size = embedding_size
        self.mid_hidden_channels = mid_hidden_channels
        self.num_embedding_basis = num_embedding_basis
        self.max_num_elements = max_num_elements

        self.fc1 = nn.Linear(self.in_hidden_channels, self.mid_hidden_channels)
        self.fc2 = nn.Linear(
            self.mid_hidden_channels,
            self.num_embedding_basis * self.mid_hidden_channels,
        )
        self.fc3 = nn.Linear(
            self.mid_hidden_channels, self.out_hidden_channels
        )

        self.source_embedding = nn.Embedding(
            max_num_elements, self.embedding_size
        )
        self.target_embedding = nn.Embedding(
            max_num_elements, self.embedding_size
        )
        nn.init.uniform_(self.source_embedding.weight.data, -0.0001, 0.0001)
        nn.init.uniform_(self.target_embedding.weight.data, -0.0001, 0.0001)

        self.embed_fc1 = nn.Linear(
            2 * self.embedding_size, self.num_embedding_basis
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(
        self, x: torch.Tensor, source_element, target_element
    ) -> torch.Tensor:
        source_embedding = self.source_embedding(source_element)
        target_embedding = self.target_embedding(target_element)
        embedding = torch.cat([source_embedding, target_embedding], dim=1)
        embedding = self.embed_fc1(embedding)
        embedding = self.softmax(embedding)

        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = (
            x.view(-1, self.num_embedding_basis, self.mid_hidden_channels)
        ) * (embedding.view(-1, self.num_embedding_basis, 1))
        x = torch.sum(x, dim=1)
        x = self.fc3(x)

        return x


class DistanceBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        max_num_elements: int,
        scalar_max,
        distance_expansion,
        scale_distances,
    ) -> None:
        super(DistanceBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_num_elements = max_num_elements
        self.distance_expansion = distance_expansion
        self.scalar_max = scalar_max
        self.scale_distances = scale_distances

        if self.scale_distances:
            self.dist_scalar = nn.Embedding(
                self.max_num_elements * self.max_num_elements, 1
            )
            self.dist_offset = nn.Embedding(
                self.max_num_elements * self.max_num_elements, 1
            )
            nn.init.uniform_(self.dist_scalar.weight.data, -0.0001, 0.0001)
            nn.init.uniform_(self.dist_offset.weight.data, -0.0001, 0.0001)

        self.fc1 = nn.Linear(self.in_channels, self.out_channels)

    def forward(self, edge_distance, source_element, target_element):
        if self.scale_distances:
            embedding_index = (
                source_element * self.max_num_elements + target_element
            )

            # Restrict the scalar to range from 1 / self.scalar_max to self.scalar_max
            scalar_max = math.log(self.scalar_max)
            scalar = (
                2.0 * torch.sigmoid(self.dist_scalar(embedding_index).view(-1))
                - 1.0
            )
            scalar = torch.exp(scalar_max * scalar)
            offset = self.dist_offset(embedding_index).view(-1)
            x = self.distance_expansion(scalar * edge_distance + offset)
        else:
            x = self.distance_expansion(edge_distance)

        x = self.fc1(x)

        return x


class ProjectLatLongSphere(torch.nn.Module):
    def __init__(self, sphere_size_lat: int, sphere_size_long: int) -> None:
        super(ProjectLatLongSphere, self).__init__()
        self.sphere_size_lat = sphere_size_lat
        self.sphere_size_long = sphere_size_long

    def forward(
        self, x, length: int, index, delta, source_edge_index
    ) -> torch.Tensor:
        device = x.device
        hidden_channels = len(x[0])

        x_proj = torch.zeros(
            length * self.sphere_size_lat * self.sphere_size_long,
            hidden_channels,
            device=device,
        )
        splat_values = x[source_edge_index]

        # Perform bilinear splatting
        x_proj.index_add_(0, index[0], splat_values * (delta[0].view(-1, 1)))
        x_proj.index_add_(0, index[1], splat_values * (delta[1].view(-1, 1)))
        x_proj.index_add_(0, index[2], splat_values * (delta[2].view(-1, 1)))
        x_proj.index_add_(0, index[3], splat_values * (delta[3].view(-1, 1)))

        x_proj = x_proj.view(
            length,
            self.sphere_size_lat * self.sphere_size_long,
            hidden_channels,
        )
        x_proj = torch.transpose(x_proj, 1, 2).contiguous()
        x_proj = x_proj.view(
            length,
            hidden_channels,
            self.sphere_size_lat,
            self.sphere_size_long,
        )

        return x_proj


class Swish(torch.nn.Module):
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = -5.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
        basis_width_scalar: float = 1.0,
    ) -> None:
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = (
            -0.5 / (basis_width_scalar * (offset[1] - offset[0])).item() ** 2
        )
        self.register_buffer("offset", offset)

    def forward(self, dist) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))
