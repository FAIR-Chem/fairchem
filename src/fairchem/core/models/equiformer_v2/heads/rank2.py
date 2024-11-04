"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from functools import partial

import torch
from e3nn import o3
from torch import nn
from torch_scatter import scatter

from fairchem.core.common.registry import registry
from fairchem.core.models.base import BackboneInterface, HeadInterface
from fairchem.core.models.equiformer_v2.layer_norm import get_normalization_layer
from fairchem.core.models.equiformer_v2.weight_initialization import eqv2_init_weights


class Rank2Block(nn.Module):
    """
    Output block for predicting rank-2 tensors (stress, dielectric tensor).
    Applies outer product between edges and computes node-wise or edge-wise MLP.

    Args:
        emb_size (int):     Size of edge embedding used to compute outer product
        num_layers (int):   Number of layers of the MLP
        edge_level (bool):  If true apply MLP at edge level before pooling, otherwise use MLP at nodes after pooling
        extensive (bool):   Whether to sum or average the outer products
    """

    def __init__(
        self,
        emb_size: int,
        num_layers: int = 2,
        edge_level: bool = False,
        extensive: bool = False,
    ):
        super().__init__()

        self.edge_level = edge_level
        self.emb_size = emb_size
        self.extensive = extensive
        self.scalar_nonlinearity = nn.SiLU()
        self.r2tensor_MLP = nn.Sequential()
        for i in range(num_layers):
            if i < num_layers - 1:
                self.r2tensor_MLP.append(nn.Linear(emb_size, emb_size))
                self.r2tensor_MLP.append(self.scalar_nonlinearity)
            else:
                self.r2tensor_MLP.append(nn.Linear(emb_size, 1))

    def forward(self, edge_distance_vec, x_edge, edge_index, data):
        """
        Args:
            edge_distance_vec (torch.Tensor):   Tensor of shape (..., 3)
            x_edge (torch.Tensor):              Tensor of shape (..., emb_size)
            edge_index (torch.Tensor):          Tensor of shape (2, nEdges)
            data:                               LMDBDataset sample
        """

        outer_product_edge = torch.bmm(
            edge_distance_vec.unsqueeze(2), edge_distance_vec.unsqueeze(1)
        )

        edge_outer = (
            x_edge[:, :, None] * outer_product_edge.view(-1, 9)[:, None, :]
        )  # should end up as 2400 x 128 x 9

        # edge_outer: (nEdges, emb_size_edge, 9)
        if self.edge_level:
            # MLP at edge level before pooling.
            edge_outer = edge_outer.transpose(1, 2)  # (nEdges, 9, emb_size_edge)
            edge_outer = self.r2tensor_MLP(edge_outer)  # (nEdges, 9, 1)
            edge_outer = edge_outer.reshape(-1, 9)  # (nEdges, 9)

            node_outer = scatter(edge_outer, edge_index, dim=0, reduce="mean")
        else:
            # operates at edge level before mixing / MLP => mixing / MLP happens at node level
            node_outer = scatter(edge_outer, edge_index, dim=0, reduce="mean")

            node_outer = node_outer.transpose(1, 2)  # (natoms, 9, emb_size_edge)
            node_outer = self.r2tensor_MLP(node_outer)  # (natoms, 9, 1)
            node_outer = node_outer.reshape(-1, 9)  # (natoms, 9)

        # node_outer: nAtoms, 9 => average across all atoms at the structure level
        if self.extensive:
            r2_tensor = scatter(node_outer, data.batch, dim=0, reduce="sum")
        else:
            r2_tensor = scatter(node_outer, data.batch, dim=0, reduce="mean")
        return r2_tensor


class Rank2DecompositionEdgeBlock(nn.Module):
    """
    Output block for predicting rank-2 tensors (stress, dielectric tensor, etc).
    Decomposes a rank-2 symmetric tensor into irrep degree 0 and 2.

    Args:
        emb_size (int):     Size of edge embedding used to compute outer product
        num_layers (int):   Number of layers of the MLP
        edge_level (bool):   If true apply MLP at edge level before pooling, otherwise use MLP at nodes after pooling
        extensive (bool):   Whether to sum or average the outer products
    """

    def __init__(
        self,
        emb_size: int,
        num_layers: int = 2,
        edge_level: bool = False,
        extensive: bool = False,
    ):
        super().__init__()
        self.emb_size = emb_size
        self.edge_level = edge_level
        self.extensive = extensive
        self.scalar_nonlinearity = nn.SiLU()
        self.scalar_MLP = nn.Sequential()
        self.irrep2_MLP = nn.Sequential()
        for i in range(num_layers):
            if i < num_layers - 1:
                self.scalar_MLP.append(nn.Linear(emb_size, emb_size))
                self.irrep2_MLP.append(nn.Linear(emb_size, emb_size))
                self.scalar_MLP.append(self.scalar_nonlinearity)
                self.irrep2_MLP.append(self.scalar_nonlinearity)
            else:
                self.scalar_MLP.append(nn.Linear(emb_size, 1))
                self.irrep2_MLP.append(nn.Linear(emb_size, 1))

        # Change of basis obtained by stacking the C-G coefficients
        self.change_mat = torch.transpose(
            torch.tensor(
                [
                    [3 ** (-0.5), 0, 0, 0, 3 ** (-0.5), 0, 0, 0, 3 ** (-0.5)],
                    [0, 0, 0, 0, 0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0],
                    [0, 0, -(2 ** (-0.5)), 0, 0, 0, 2 ** (-0.5), 0, 0],
                    [0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0, 0, 0, 0, 0],
                    [0, 0, 0.5**0.5, 0, 0, 0, 0.5**0.5, 0, 0],
                    [0, 2 ** (-0.5), 0, 2 ** (-0.5), 0, 0, 0, 0, 0],
                    [
                        -(6 ** (-0.5)),
                        0,
                        0,
                        0,
                        2 * 6 ** (-0.5),
                        0,
                        0,
                        0,
                        -(6 ** (-0.5)),
                    ],
                    [0, 0, 0, 0, 0, 2 ** (-0.5), 0, 2 ** (-0.5), 0],
                    [-(2 ** (-0.5)), 0, 0, 0, 0, 0, 0, 0, 2 ** (-0.5)],
                ]
            ).detach(),
            0,
            1,
        )

    def forward(self, edge_distance_vec, x_edge, edge_index, data):
        """
        Args:
            edge_distance_vec (torch.Tensor):   Tensor of shape (..., 3)
            x_edge (torch.Tensor):              Tensor of shape (..., emb_size)
            edge_index (torch.Tensor):          Tensor of shape (2, nEdges)
            data:                               LMDBDataset sample
        """
        # Calculate spherical harmonics of degree 2 of the points sampled
        sphere_irrep2 = o3.spherical_harmonics(
            2, edge_distance_vec, True
        ).detach()  # (nEdges, 5)

        if self.edge_level:
            # MLP at edge level before pooling.

            # Irrep 0 prediction
            edge_scalar = x_edge
            edge_scalar = self.scalar_MLP(edge_scalar)

            # Irrep 2 prediction
            edge_irrep2 = (
                sphere_irrep2[:, :, None] * x_edge[:, None, :]
            )  # (nEdges, 5, emb_size)
            edge_irrep2 = self.irrep2_MLP(edge_irrep2)

            node_scalar = scatter(edge_scalar, edge_index, dim=0, reduce="mean")
            node_irrep2 = scatter(edge_irrep2, edge_index, dim=0, reduce="mean")
        else:
            edge_irrep2 = (
                sphere_irrep2[:, :, None] * x_edge[:, None, :]
            )  # (nAtoms, 5, emb_size)

            node_scalar = scatter(x_edge, edge_index, dim=0, reduce="mean")
            node_irrep2 = scatter(edge_irrep2, edge_index, dim=0, reduce="mean")

            # Irrep 0 prediction
            for module in self.scalar_MLP:
                node_scalar = module(node_scalar)

            # Irrep 2 prediction
            for module in self.irrep2_MLP:
                node_irrep2 = module(node_irrep2)

        scalar = scatter(
            node_scalar.view(-1),
            data.batch,
            dim=0,
            reduce="sum" if self.extensive else "mean",
        )
        irrep2 = scatter(
            node_irrep2.view(-1, 5),
            data.batch,
            dim=0,
            reduce="sum" if self.extensive else "mean",
        )

        # Note (@abhshkdz): If we have separate normalizers on the isotropic and
        # anisotropic components (implemented in the trainer), combining the
        # scalar and irrep2 predictions here would lead to the incorrect result.
        # Instead, we should combine the predictions after the normalizers.

        return scalar.reshape(-1), irrep2


@registry.register_model("rank2_symmetric_head")
class Rank2SymmetricTensorHead(nn.Module, HeadInterface):
    """A rank 2 symmetric tensor prediction head.

    Attributes:
        ouput_name: name of output prediction property (ie, stress)
        sphharm_norm: layer normalization for spherical harmonic edge weights
        xedge_layer_norm: embedding layer norm
        block: rank 2 equivariant symmetric tensor block
    """

    def __init__(
        self,
        backbone: BackboneInterface,
        output_name: str = "stress",
        decompose: bool = False,
        edge_level_mlp: bool = False,
        num_mlp_layers: int = 2,
        use_source_target_embedding: bool = False,
        extensive: bool = False,
        avg_num_nodes: int = 1.0,
        default_norm_type: str = "layer_norm_sh",
    ):
        """
        Args:
            backbone: Backbone model that the head is attached to
            decompose: Whether to decompose the rank2 tensor into isotropic and anisotropic components
            edge_level_mlp: If true apply MLP at edge level before pooling, otherwise use MLP at nodes after pooling
            num_mlp_layers: number of MLP layers
            use_source_target_embedding: Whether to use both source and target atom embeddings
            extensive: Whether to do sum-pooling (extensive) vs mean pooling (intensive).
            avg_num_nodes: Used only if extensive to divide prediction by avg num nodes.
        """
        super().__init__()
        self.output_name = output_name
        self.decompose = decompose
        self.use_source_target_embedding = use_source_target_embedding
        self.avg_num_nodes = avg_num_nodes

        self.sphharm_norm = get_normalization_layer(
            getattr(backbone, "norm_type", default_norm_type),
            lmax=max(backbone.lmax_list),
            num_channels=1,
        )

        if use_source_target_embedding:
            r2_tensor_sphere_channels = backbone.sphere_channels * 2
        else:
            r2_tensor_sphere_channels = backbone.sphere_channels

        self.xedge_layer_norm = nn.LayerNorm(r2_tensor_sphere_channels)

        if decompose:
            self.block = Rank2DecompositionEdgeBlock(
                emb_size=r2_tensor_sphere_channels,
                num_layers=num_mlp_layers,
                edge_level=edge_level_mlp,
                extensive=extensive,
            )
        else:
            self.block = Rank2Block(
                emb_size=r2_tensor_sphere_channels,
                num_layers=num_mlp_layers,
                edge_level=edge_level_mlp,
                extensive=extensive,
            )

        # initialize weights
        self.block.apply(partial(eqv2_init_weights, weight_init="uniform"))

    def forward(
        self, data: dict[str, torch.Tensor] | torch.Tensor, emb: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            data: data batch
            emb: dictionary with embedding object and graph data

        Returns: dict of {output property name: predicted value}
        """
        node_emb, graph = emb["node_embedding"], emb["graph"]

        sphharm_weights_edge = o3.spherical_harmonics(
            torch.arange(0, node_emb.lmax_list[-1] + 1).tolist(),
            graph.edge_distance_vec,
            False,
        ).detach()

        # layer norm because sphharm_weights_edge values become large and causes infs with amp
        sphharm_weights_edge = self.sphharm_norm(
            sphharm_weights_edge[:, :, None]
        ).squeeze()

        if self.use_source_target_embedding:
            x_source = node_emb.expand_edge(graph.edge_index[0]).embedding
            x_target = node_emb.expand_edge(graph.edge_index[1]).embedding
            x_edge = torch.cat((x_source, x_target), dim=2)
        else:
            x_edge = node_emb.expand_edge(graph.edge_index[1]).embedding

        x_edge = torch.einsum("abc, ab->ac", x_edge, sphharm_weights_edge)

        # layer norm because x_edge values become large and causes infs with amp
        x_edge = self.xedge_layer_norm(x_edge)

        if self.decompose:
            tensor_0, tensor_2 = self.block(
                graph.edge_distance_vec, x_edge, graph.edge_index[1], data
            )

            if self.block.extensive:  # legacy, may be interesting to try
                tensor_0 = tensor_0 / self.avg_num_nodes
                tensor_2 = tensor_2 / self.avg_num_nodes

            output = {
                f"{self.output_name}_isotropic": tensor_0.unsqueeze(1),
                f"{self.output_name}_anisotropic": tensor_2,
            }
        else:
            out_tensor = self.block(
                graph.edge_distance_vec, x_edge, graph.edge_index[1], data
            )
            output = {self.output_name: out_tensor.reshape((-1, 3))}

        return output
