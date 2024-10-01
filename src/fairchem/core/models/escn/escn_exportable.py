"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import contextlib
import logging
import os
import typing

import torch
import torch.nn as nn

if typing.TYPE_CHECKING:
    from torch_geometric.data.batch import Batch

from fairchem.core.common.registry import registry
from fairchem.core.models.base import GraphModelMixin
from fairchem.core.models.escn.so3_exportable import (
    CoefficientMapping,
    SO3_Grid,
    rotation_to_wigner,
)
from fairchem.core.models.scn.sampling import CalcSpherePoints
from fairchem.core.models.scn.smearing import (
    GaussianSmearing,
    LinearSigmoidSmearing,
    SigmoidSmearing,
    SiLUSmearing,
)

with contextlib.suppress(ImportError):
    from e3nn import o3


@registry.register_model("escn_export")
class eSCN(nn.Module, GraphModelMixin):
    """Equivariant Spherical Channel Network
    Paper: Reducing SO(3) Convolutions to SO(2) for Efficient Equivariant GNNs


    Args:
        max_neighbors(int):           Max neighbors to take per node, when using the graph generation
        cutoff (float):               Maximum distance between nieghboring atoms in Angstroms
        max_num_elements (int):       Maximum atomic number
        num_layers (int):             Number of layers in the GNN
        lmax (int):                   maximum degree of the spherical harmonics (1 to 10)
        mmax (int):                   maximum order of the spherical harmonics (0 to lmax)
        sphere_channels (int):        Number of spherical channels (one set per resolution)
        hidden_channels (int):        Number of hidden units in message passing
        num_sphere_samples (int):     Number of samples used to approximate the integration of the sphere in the output blocks
        edge_channels (int):          Number of channels for the edge invariant features
        distance_function ("gaussian", "sigmoid", "linearsigmoid", "silu"):  Basis function used for distances
        basis_width_scalar (float):   Width of distance basis function
        distance_resolution (float):  Distance between distance basis functions in Angstroms
        compile (bool):               use torch.compile on the forward
        export (bool):                use the exportable version of the module
    """

    def __init__(
        self,
        max_neighbors: int = 300,
        cutoff: float = 8.0,
        max_num_elements: int = 100,
        num_layers: int = 8,
        lmax: int = 4,
        mmax: int = 2,
        sphere_channels: int = 128,
        hidden_channels: int = 256,
        edge_channels: int = 128,
        num_sphere_samples: int = 128,
        distance_function: str = "gaussian",
        basis_width_scalar: float = 1.0,
        distance_resolution: float = 0.02,
        resolution: int | None = None,
        compile: bool = False,
        export: bool = False,
        rescale_grid: bool = False,
    ) -> None:
        super().__init__()

        import sys

        if "e3nn" not in sys.modules:
            logging.error("You need to install the e3nn library to use the SCN model")
            raise ImportError

        self.max_neighbors = max_neighbors
        self.cutoff = cutoff
        self.max_num_elements = max_num_elements
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_sphere_samples = num_sphere_samples
        self.sphere_channels = sphere_channels
        self.edge_channels = edge_channels
        self.distance_resolution = distance_resolution
        self.lmax = lmax
        self.mmax = mmax
        self.basis_width_scalar = basis_width_scalar
        self.distance_function = distance_function
        self.compile = compile
        self.export = export
        self.rescale_grid = rescale_grid

        # non-linear activation function used throughout the network
        self.act = nn.SiLU()

        # Weights for message initialization
        self.sphere_embedding = nn.Embedding(
            self.max_num_elements, self.sphere_channels
        )

        # Initialize the function used to measure the distances between atoms
        assert self.distance_function in [
            "gaussian",
            "sigmoid",
            "linearsigmoid",
            "silu",
        ]
        self.num_gaussians = int(cutoff / self.distance_resolution)
        if self.distance_function == "gaussian":
            self.distance_expansion = GaussianSmearing(
                0.0,
                cutoff,
                self.num_gaussians,
                basis_width_scalar,
            )
        if self.distance_function == "sigmoid":
            self.distance_expansion = SigmoidSmearing(
                0.0,
                cutoff,
                self.num_gaussians,
                basis_width_scalar,
            )
        if self.distance_function == "linearsigmoid":
            self.distance_expansion = LinearSigmoidSmearing(
                0.0,
                cutoff,
                self.num_gaussians,
                basis_width_scalar,
            )
        if self.distance_function == "silu":
            self.distance_expansion = SiLUSmearing(
                0.0,
                cutoff,
                self.num_gaussians,
                basis_width_scalar,
            )

        # Initialize the transformations between spherical and grid representations
        self.SO3_grid = nn.ModuleDict()
        self.SO3_grid["lmax_lmax"] = SO3_Grid(
            self.lmax, self.lmax, resolution=resolution, rescale=self.rescale_grid
        )
        self.SO3_grid["lmax_mmax"] = SO3_Grid(
            self.lmax, self.mmax, resolution=resolution, rescale=self.rescale_grid
        )

        # Initialize the blocks for each layer of the GNN
        self.layer_blocks = nn.ModuleList()
        for i in range(self.num_layers):
            block = LayerBlock(
                i,
                self.sphere_channels,
                self.hidden_channels,
                self.edge_channels,
                self.lmax,
                self.mmax,
                self.distance_expansion,
                self.max_num_elements,
                self.SO3_grid,
                self.act,
            )
            self.layer_blocks.append(block)

        # Output blocks for energy and forces
        self.energy_block = EnergyBlock(
            self.sphere_channels, self.num_sphere_samples, self.act
        )
        self.force_block = ForceBlock(
            self.sphere_channels, self.num_sphere_samples, self.act
        )

        # Create a roughly evenly distributed point sampling of the sphere for the output blocks
        self.sphere_points = nn.Parameter(
            CalcSpherePoints(self.num_sphere_samples), requires_grad=False
        )

        # For each spherical point, compute the spherical harmonic coefficient weights
        self.sphharm_weights: nn.Parameter = nn.Parameter(
            o3.spherical_harmonics(
                torch.arange(0, self.lmax + 1).tolist(),
                self.sphere_points,
                False,
            ),
            requires_grad=False,
        )

        self.sph_feature_size = int((self.lmax + 1) ** 2)
        # Pre-load Jd tensors for wigner matrices
        # Borrowed from e3nn @ 0.4.0:
        # https://github.com/e3nn/e3nn/blob/0.4.0/e3nn/o3/_wigner.py#L10
        # _Jd is a list of tensors of shape (2l+1, 2l+1)
        # TODO: we should probably just bake this into the file as strings to avoid
        # carrying this extra file around
        Jd_list = torch.load(os.path.join(os.path.dirname(__file__), "Jd.pt"))
        for l in range(self.lmax + 1):
            self.register_buffer(f"Jd_{l}", Jd_list[l])

        if self.compile:
            logging.info("Using the compiled escn forward function...")
            self.forward = torch.compile(
                options={"triton.cudagraphs": True}, fullgraph=True, dynamic=True
            )(self.forward)

        # torch.export only works with nn.module with an unaltered forward function,
        # furthermore AOT Inductor currently requires a flat list of inputs
        # this we need keep the module.forward function as the fully exportable region
        # When not using export, ie for training, we swap out the forward with a version
        # that wraps it with the graph generator
        #
        # TODO: this is really ugly and confusing to read, find a better way to deal
        # with partially exportable model
        if not self.export:
            self._forward = self.forward
            self.forward = self.forward_trainable

    def forward_trainable(self, data: Batch) -> dict[str, torch.Tensor]:
        # standard forward call that generates the graph on-the-fly with generate_graph
        # this part of the code is not compile/export friendly so we keep it separated and wrap the exportaable forward
        graph = self.generate_graph(
            data,
            max_neighbors=self.max_neighbors,
            otf_graph=True,
            use_pbc=True,
            use_pbc_single=True,
        )
        energy, forces = self._forward(
            data.pos,
            data.batch,
            data.natoms,
            data.atomic_numbers.long(),
            graph.edge_index,
            graph.edge_distance,
            graph.edge_distance_vec,
        )
        return {"energy": energy, "forces": forces}

    # a fully compilable/exportable forward function
    # takes a full graph with edges as input
    def forward(
        self,
        pos: torch.Tensor,
        batch_idx: torch.Tensor,
        natoms: torch.Tensor,
        atomic_numbers: torch.Tensor,
        edge_index: torch.Tensor,
        edge_distance: torch.Tensor,
        edge_distance_vec: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        N: num atoms
        N: batch size
        E: num edges

        pos: [N, 3] atom positions
        batch_idx: [N] batch index of each atom
        natoms: [B] number of atoms in each batch
        atomic_numbers: [N] atomic number per atom
        edge_index: [2, E] edges between source and target atoms
        edge_distance: [E] cartesian distance for each edge
        edge_distance_vec: [E, 3] direction vector of edges (includes pbc)
        """
        if not self.export and not self.compile:
            assert atomic_numbers.max().item() < self.max_num_elements
        num_atoms = len(atomic_numbers)

        ###############################################################
        # Initialize data structures
        ###############################################################

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = self._init_edge_rot_mat(edge_distance_vec)
        Jd_buffers = [
            getattr(self, f"Jd_{l}").type(edge_rot_mat.dtype)
            for l in range(self.lmax + 1)
        ]
        wigner = rotation_to_wigner(edge_rot_mat, 0, self.lmax, Jd_buffers).detach()

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        x_message = torch.zeros(
            num_atoms,
            self.sph_feature_size,
            self.sphere_channels,
            device=pos.device,
            dtype=pos.dtype,
        )
        x_message[:, 0, :] = self.sphere_embedding(atomic_numbers)

        ###############################################################
        # Update spherical node embeddings
        ###############################################################
        for i in range(self.num_layers):
            if i > 0:
                x_message_new = self.layer_blocks[i](
                    x_message,
                    atomic_numbers,
                    edge_distance,
                    edge_index,
                    wigner,
                )
                # Residual layer for all layers past the first
                x_message = x_message + x_message_new
            else:
                # No residual for the first layer
                x_message = self.layer_blocks[i](
                    x_message,
                    atomic_numbers,
                    edge_distance,
                    edge_index,
                    wigner,
                )

        # Sample the spherical channels (node embeddings) at evenly distributed points on the sphere.
        # These values are fed into the output blocks.
        x_pt = torch.einsum(
            "abc, pb->apc", x_message, self.sphharm_weights
        ).contiguous()

        ###############################################################
        # Energy estimation
        ###############################################################
        node_energy = self.energy_block(x_pt)
        energy = torch.zeros(len(natoms), device=node_energy.device)
        energy.index_add_(0, batch_idx, node_energy.view(-1))
        # Scale energy to help balance numerical precision w.r.t. forces
        energy = energy * 0.001

        ###############################################################
        # Force estimation
        ###############################################################
        forces = self.force_block(x_pt, self.sphere_points)

        return energy, forces

    # Initialize the edge rotation matrics
    def _init_edge_rot_mat(self, edge_distance_vec):
        edge_vec_0 = edge_distance_vec
        edge_vec_0_distance = torch.sqrt(torch.sum(edge_vec_0**2, dim=1))

        # Make sure the atoms are far enough apart
        # assert torch.min(edge_vec_0_distance) < 0.0001

        norm_x = edge_vec_0 / (edge_vec_0_distance.view(-1, 1))

        edge_vec_2 = torch.rand_like(edge_vec_0) - 0.5
        edge_vec_2 = edge_vec_2 / (
            torch.sqrt(torch.sum(edge_vec_2**2, dim=1)).view(-1, 1)
        )
        # Create two rotated copys of the random vectors in case the random vector is aligned with norm_x
        # With two 90 degree rotated vectors, at least one should not be aligned with norm_x
        edge_vec_2b = edge_vec_2.clone()
        edge_vec_2b[:, 0] = -edge_vec_2[:, 1]
        edge_vec_2b[:, 1] = edge_vec_2[:, 0]
        edge_vec_2c = edge_vec_2.clone()
        edge_vec_2c[:, 1] = -edge_vec_2[:, 2]
        edge_vec_2c[:, 2] = edge_vec_2[:, 1]
        vec_dot_b = torch.abs(torch.sum(edge_vec_2b * norm_x, dim=1)).view(-1, 1)
        vec_dot_c = torch.abs(torch.sum(edge_vec_2c * norm_x, dim=1)).view(-1, 1)

        vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
        edge_vec_2 = torch.where(torch.gt(vec_dot, vec_dot_b), edge_vec_2b, edge_vec_2)
        vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
        edge_vec_2 = torch.where(torch.gt(vec_dot, vec_dot_c), edge_vec_2c, edge_vec_2)

        vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1))
        # Check the vectors aren't aligned
        assert torch.max(vec_dot) < 0.99

        norm_z = torch.cross(norm_x, edge_vec_2, dim=1)
        norm_z = norm_z / (torch.sqrt(torch.sum(norm_z**2, dim=1, keepdim=True)))
        norm_z = norm_z / (torch.sqrt(torch.sum(norm_z**2, dim=1)).view(-1, 1))
        norm_y = torch.cross(norm_x, norm_z, dim=1)
        norm_y = norm_y / (torch.sqrt(torch.sum(norm_y**2, dim=1, keepdim=True)))

        # Construct the 3D rotation matrix
        norm_x = norm_x.view(-1, 3, 1)
        norm_y = -norm_y.view(-1, 3, 1)
        norm_z = norm_z.view(-1, 3, 1)

        edge_rot_mat_inv = torch.cat([norm_z, norm_x, norm_y], dim=2)
        edge_rot_mat = torch.transpose(edge_rot_mat_inv, 1, 2)

        return edge_rot_mat.detach()

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class LayerBlock(torch.nn.Module):
    """
    Layer block: Perform one layer (message passing and aggregation) of the GNN

    Args:
        layer_idx (int):            Layer number
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during the SO(2) conv
        edge_channels (int):        Size of invariant edge embedding
        lmax (int)                  degrees (l) for each resolution
        mmax (int):                 orders (m) for each resolution
        distance_expansion (func):  Function used to compute distance embedding
        max_num_elements (int):     Maximum number of atomic numbers
        SO3_grid (SO3_grid):        Class used to convert from grid the spherical harmonic representations
        act (function):             Non-linear activation function
    """

    def __init__(
        self,
        layer_idx: int,
        sphere_channels: int,
        hidden_channels: int,
        edge_channels: int,
        lmax: int,
        mmax: int,
        distance_expansion,
        max_num_elements: int,
        SO3_grid: SO3_Grid,
        act,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.act = act
        self.lmax = lmax
        self.mmax = mmax
        self.sphere_channels = sphere_channels
        self.SO3_grid = SO3_grid

        # Message block
        self.message_block = MessageBlock(
            self.layer_idx,
            self.sphere_channels,
            hidden_channels,
            edge_channels,
            self.lmax,
            self.mmax,
            distance_expansion,
            max_num_elements,
            self.SO3_grid,
            self.act,
        )

        # Non-linear point-wise comvolution for the aggregated messages
        self.fc1_sphere = nn.Linear(
            2 * self.sphere_channels, self.sphere_channels, bias=False
        )

        self.fc2_sphere = nn.Linear(
            self.sphere_channels, self.sphere_channels, bias=False
        )

        self.fc3_sphere = nn.Linear(
            self.sphere_channels, self.sphere_channels, bias=False
        )

    def forward(
        self,
        x: torch.Tensor,
        atomic_numbers: torch.Tensor,
        edge_distance: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: torch.Tensor,
    ) -> torch.Tensor:
        # Compute messages by performing message block
        x_message = self.message_block(
            x,
            atomic_numbers,
            edge_distance,
            edge_index,
            wigner,
        )

        # Compute point-wise spherical non-linearity on aggregated messages
        # Project to grid
        to_grid_mat = self.SO3_grid["lmax_lmax"].to_grid_mat[
            :,
            :,
            self.SO3_grid["lmax_lmax"].mapping.coefficient_idx(self.lmax, self.lmax),
        ]
        x_grid_message = torch.einsum("bai,zic->zbac", to_grid_mat, x_message)

        # x_grid = x.to_grid(self.SO3_grid["lmax_lmax"])
        x_grid = torch.einsum("bai,zic->zbac", to_grid_mat, x)
        x_grid = torch.cat([x_grid, x_grid_message], dim=3)

        # Perform point-wise convolution
        x_grid = self.act(self.fc1_sphere(x_grid))
        x_grid = self.act(self.fc2_sphere(x_grid))
        x_grid = self.fc3_sphere(x_grid)

        # Project back to spherical harmonic coefficients
        from_grid_mat = self.SO3_grid["lmax_lmax"].from_grid_mat[
            :,
            :,
            self.SO3_grid["lmax_lmax"].mapping.coefficient_idx(self.lmax, self.lmax),
        ]
        return torch.einsum("bai,zbac->zic", from_grid_mat, x_grid)


class MessageBlock(torch.nn.Module):
    """
    Message block: Perform message passing

    Args:
        layer_idx (int):            Layer number
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during the SO(2) conv
        edge_channels (int):        Size of invariant edge embedding
        lmax (int):                 degrees (l) for each resolution
        mmax (int):                 orders (m) for each resolution
        distance_expansion (func):  Function used to compute distance embedding
        max_num_elements (int):     Maximum number of atomic numbers
        SO3_grid (SO3_grid):        Class used to convert from grid the spherical harmonic representations
        act (function):             Non-linear activation function
    """

    def __init__(
        self,
        layer_idx: int,
        sphere_channels: int,
        hidden_channels: int,
        edge_channels: int,
        lmax: int,
        mmax: int,
        distance_expansion,
        max_num_elements: int,
        SO3_grid: SO3_Grid,
        act,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.act = act
        self.hidden_channels = hidden_channels
        self.sphere_channels = sphere_channels
        self.SO3_grid = SO3_grid
        self.lmax = lmax
        self.mmax = mmax
        self.edge_channels = edge_channels
        self.out_mask = CoefficientMapping([self.lmax], [self.lmax]).coefficient_idx(
            self.lmax, self.mmax
        )

        # Create edge scalar (invariant to rotations) features
        self.edge_block = EdgeBlock(
            self.edge_channels,
            distance_expansion,
            max_num_elements,
            self.act,
        )

        # Create SO(2) convolution blocks
        self.so2_block_source = SO2Block(
            self.sphere_channels,
            self.hidden_channels,
            self.edge_channels,
            self.lmax,
            self.mmax,
            self.act,
        )
        self.so2_block_target = SO2Block(
            self.sphere_channels,
            self.hidden_channels,
            self.edge_channels,
            self.lmax,
            self.mmax,
            self.act,
        )

    def forward(
        self,
        x: torch.Tensor,
        atomic_numbers: torch.Tensor,
        edge_distance: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: torch.Tensor,
    ) -> torch.Tensor:
        ###############################################################
        # Compute messages
        ###############################################################
        # Compute edge scalar features (invariant to rotations)
        # Uses atomic numbers and edge distance as inputs
        x_edge = self.edge_block(
            edge_distance,
            atomic_numbers[edge_index[0]],  # Source atom atomic number
            atomic_numbers[edge_index[1]],  # Target atom atomic number
        )

        # Copy embeddings for each edge's source and target nodes
        x_source = x.clone()
        x_target = x.clone()
        x_source = x_source[edge_index[0, :]]
        x_target = x_target[edge_index[1, :]]

        # Rotate the irreps to align with the edge
        x_source = torch.bmm(wigner[:, self.out_mask, :], x_source)
        x_target = torch.bmm(wigner[:, self.out_mask, :], x_target)

        # Compute messages
        x_source = self.so2_block_source(x_source, x_edge)
        x_target = self.so2_block_target(x_target, x_edge)

        # Add together the source and target results
        x_target = x_source + x_target

        # Point-wise spherical non-linearity
        to_grid_mat = self.SO3_grid["lmax_mmax"].get_to_grid_mat()
        from_grid_mat = self.SO3_grid["lmax_mmax"].get_from_grid_mat()
        x_grid = torch.einsum("bai,zic->zbac", to_grid_mat, x_target)
        x_grid = self.act(x_grid)
        x_target = torch.einsum("bai,zbac->zic", from_grid_mat, x_grid)

        # Rotate back the irreps
        wigner_inv = torch.transpose(wigner, 1, 2).contiguous().detach()
        x_target = torch.bmm(wigner_inv[:, :, self.out_mask], x_target)

        # Compute the sum of the incoming neighboring messages for each target node
        new_embedding = torch.zeros(
            x.shape, dtype=x_target.dtype, device=x_target.device
        )
        new_embedding.index_add_(0, edge_index[1], x_target)

        return new_embedding


class SO2Block(torch.nn.Module):
    """
    SO(2) Block: Perform SO(2) convolutions for all m (orders)

    Args:
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during the SO(2) conv
        edge_channels (int):        Size of invariant edge embedding
        lmax (int):                 degrees (l) for each resolution
        mmax (int):                 orders (m) for each resolution
        act (function):             Non-linear activation function
    """

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        edge_channels: int,
        lmax: int,
        mmax: int,
        act,
    ) -> None:
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax
        self.act = act
        self.mappingReduced = CoefficientMapping([self.lmax], [self.mmax])

        num_channels_m0 = (self.lmax + 1) * self.sphere_channels

        # SO(2) convolution for m=0
        self.fc1_dist0 = nn.Linear(edge_channels, self.hidden_channels)
        self.fc1_m0 = nn.Linear(num_channels_m0, self.hidden_channels, bias=False)
        self.fc2_m0 = nn.Linear(self.hidden_channels, num_channels_m0, bias=False)

        # SO(2) convolution for non-zero m
        self.so2_conv = nn.ModuleList()
        for m in range(1, self.mmax + 1):
            so2_conv = SO2Conv(
                m,
                self.sphere_channels,
                self.hidden_channels,
                edge_channels,
                self.lmax,
                self.mmax,
                self.act,
            )
            self.so2_conv.append(so2_conv)

    def forward(
        self,
        x: torch.Tensor,
        x_edge: torch.Tensor,
    ):
        num_edges = len(x_edge)

        # Reshape the spherical harmonics based on m (order)
        x = torch.einsum("nac,ba->nbc", x, self.mappingReduced.to_m)

        # Compute m=0 coefficients separately since they only have real values (no imaginary)

        # Compute edge scalar features for m=0
        x_edge_0 = self.act(self.fc1_dist0(x_edge))

        x_0 = x[:, 0 : self.mappingReduced.m_size[0]].contiguous()
        x_0 = x_0.view(num_edges, -1)

        x_0 = self.fc1_m0(x_0)
        x_0 = x_0 * x_edge_0
        x_0 = self.fc2_m0(x_0)
        x_0 = x_0.view(num_edges, -1, self.sphere_channels)

        # Update the m=0 coefficients
        x[:, 0 : self.mappingReduced.m_size[0]] = x_0

        # Compute the values for the m > 0 coefficients
        offset = self.mappingReduced.m_size[0]
        for m in range(1, self.mmax + 1):
            # Get the m order coefficients
            x_m = x[:, offset : offset + 2 * self.mappingReduced.m_size[m]].contiguous()
            x_m = x_m.view(num_edges, 2, -1)
            # Perform SO(2) convolution
            x_m = self.so2_conv[m - 1](x_m, x_edge)
            x_m = x_m.view(num_edges, -1, self.sphere_channels)
            x[:, offset : offset + 2 * self.mappingReduced.m_size[m]] = x_m

            offset = offset + 2 * self.mappingReduced.m_size[m]

        # Reshape the spherical harmonics based on l (degree)
        return torch.einsum("nac,ab->nbc", x, self.mappingReduced.to_m)


class SO2Conv(torch.nn.Module):
    """
    SO(2) Conv: Perform an SO(2) convolution

    Args:
        m (int):                    Order of the spherical harmonic coefficients
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during the SO(2) conv
        edge_channels (int):        Size of invariant edge embedding
        lmax (int):                 degrees (l) for each resolution
        mmax (int):                 orders (m) for each resolution
        act (function):             Non-linear activation function
    """

    def __init__(
        self,
        m: int,
        sphere_channels: int,
        hidden_channels: int,
        edge_channels: int,
        lmax: int,
        mmax: int,
        act,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax
        self.sphere_channels = sphere_channels
        self.m = m
        self.act = act

        num_coefficents = 0
        if self.mmax >= m:
            num_coefficents = self.lmax - m + 1

        num_channels = num_coefficents * self.sphere_channels

        assert num_channels > 0

        # Embedding function of the distance
        self.fc1_dist = nn.Linear(edge_channels, 2 * self.hidden_channels)

        # Real weights of SO(2) convolution
        self.fc1_r = nn.Linear(num_channels, self.hidden_channels, bias=False)
        self.fc2_r = nn.Linear(self.hidden_channels, num_channels, bias=False)

        # Imaginary weights of SO(2) convolution
        self.fc1_i = nn.Linear(num_channels, self.hidden_channels, bias=False)
        self.fc2_i = nn.Linear(self.hidden_channels, num_channels, bias=False)

    def forward(self, x_m, x_edge) -> torch.Tensor:
        # Compute edge scalar features
        x_edge = self.act(self.fc1_dist(x_edge))
        x_edge = x_edge.view(-1, 2, self.hidden_channels)

        # Perform the complex weight multiplication
        x_r = self.fc1_r(x_m)
        x_r = x_r * x_edge[:, 0:1, :]
        x_r = self.fc2_r(x_r)

        x_i = self.fc1_i(x_m)
        x_i = x_i * x_edge[:, 1:2, :]
        x_i = self.fc2_i(x_i)

        x_m_r = x_r[:, 0] - x_i[:, 1]
        x_m_i = x_r[:, 1] + x_i[:, 0]

        return torch.stack((x_m_r, x_m_i), dim=1).contiguous()


class EdgeBlock(torch.nn.Module):
    """
    Edge Block: Compute invariant edge representation from edge diatances and atomic numbers

    Args:
        edge_channels (int):        Size of invariant edge embedding
        distance_expansion (func):  Function used to compute distance embedding
        max_num_elements (int):     Maximum number of atomic numbers
        act (function):             Non-linear activation function
    """

    def __init__(
        self,
        edge_channels,
        distance_expansion,
        max_num_elements,
        act,
    ) -> None:
        super().__init__()
        self.in_channels = distance_expansion.num_output
        self.distance_expansion = distance_expansion
        self.act = act
        self.edge_channels = edge_channels
        self.max_num_elements = max_num_elements

        # Embedding function of the distance
        self.fc1_dist = nn.Linear(self.in_channels, self.edge_channels)

        # Embedding function of the atomic numbers
        self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels)
        self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels)
        nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
        nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)

        # Embedding function of the edge
        self.fc1_edge_attr = nn.Linear(
            self.edge_channels,
            self.edge_channels,
        )

    def forward(self, edge_distance, source_element, target_element):
        # Compute distance embedding
        x_dist = self.distance_expansion(edge_distance)
        x_dist = self.fc1_dist(x_dist)

        # Compute atomic number embeddings
        source_embedding = self.source_embedding(source_element)
        target_embedding = self.target_embedding(target_element)

        # Compute invariant edge embedding
        x_edge = self.act(source_embedding + target_embedding + x_dist)
        return self.act(self.fc1_edge_attr(x_edge))


class EnergyBlock(torch.nn.Module):
    """
    Energy Block: Output block computing the energy

    Args:
        num_channels (int):         Number of channels
        num_sphere_samples (int):   Number of samples used to approximate the integral on the sphere
        act (function):             Non-linear activation function
    """

    def __init__(
        self,
        num_channels: int,
        num_sphere_samples: int,
        act,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.num_sphere_samples = num_sphere_samples
        self.act = act

        self.fc1 = nn.Linear(self.num_channels, self.num_channels)
        self.fc2 = nn.Linear(self.num_channels, self.num_channels)
        self.fc3 = nn.Linear(self.num_channels, 1, bias=False)

    def forward(self, x_pt) -> torch.Tensor:
        # x_pt are the values of the channels sampled at different points on the sphere
        x_pt = self.act(self.fc1(x_pt))
        x_pt = self.act(self.fc2(x_pt))
        x_pt = self.fc3(x_pt)
        x_pt = x_pt.view(-1, self.num_sphere_samples, 1)
        return torch.sum(x_pt, dim=1) / self.num_sphere_samples


class ForceBlock(torch.nn.Module):
    """
    Force Block: Output block computing the per atom forces

    Args:
        num_channels (int):         Number of channels
        num_sphere_samples (int):   Number of samples used to approximate the integral on the sphere
        act (function):             Non-linear activation function
    """

    def __init__(
        self,
        num_channels: int,
        num_sphere_samples: int,
        act,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.num_sphere_samples = num_sphere_samples
        self.act = act

        self.fc1 = nn.Linear(self.num_channels, self.num_channels)
        self.fc2 = nn.Linear(self.num_channels, self.num_channels)
        self.fc3 = nn.Linear(self.num_channels, 1, bias=False)

    def forward(self, x_pt, sphere_points) -> torch.Tensor:
        # x_pt are the values of the channels sampled at different points on the sphere
        x_pt = self.act(self.fc1(x_pt))
        x_pt = self.act(self.fc2(x_pt))
        x_pt = self.fc3(x_pt)
        x_pt = x_pt.view(-1, self.num_sphere_samples, 1)
        forces = x_pt * sphere_points.view(1, self.num_sphere_samples, 3)
        return torch.sum(forces, dim=1) / self.num_sphere_samples
