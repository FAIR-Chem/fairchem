"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging

import torch
import torch.nn as nn
from e3nn import o3
from torch_geometric.nn import radius_graph

from ocpmodels.common.utils import (
    compute_neighbors,
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc,
    scatter_det,
)
from ocpmodels.models.gemnet_oc.layers.base_layers import Dense


class BaseModel(nn.Module):
    def __init__(
        self, output_targets, node_embedding_dim, edge_embedding_dim
    ) -> None:
        super(BaseModel, self).__init__()

        self.output_targets = output_targets
        self.num_targets = len(output_targets)

        self.module_dict = nn.ModuleDict({})
        for target in output_targets:
            if self.output_targets[target].get("custom_head", False):
                if "irrep_dim" in self.output_targets[target]:
                    embedding_dim = edge_embedding_dim
                    output_shape = 1
                else:
                    embedding_dim = node_embedding_dim
                    output_shape = self.output_targets[target].get("shape", 1)

                layers = [
                    Dense(
                        embedding_dim,
                        embedding_dim,
                        activation="silu",
                    )
                ] * self.output_targets[target].get("num_layers", 2)

                layers.append(
                    Dense(edge_embedding_dim, output_shape, activation=None)
                )

                self.module_dict[target] = nn.Sequential(*layers)

    def forward(self, data):
        batch = data.batch
        num_atoms = data.atomic_numbers.shape[0]
        num_systems = data.natoms.shape[0]
        out = self._forward(data)

        results = {}

        for target in self.output_targets:
            # for models that directly return desired property, add
            # result directly and continue
            if target not in self.module_dict:
                results[target] = out[target]
                continue

            if "irrep_dim" in self.output_targets[target]:
                irrep = self.output_targets[target]["irrep_dim"]
                edge_vec = out["edge_vec"]
                edge_idx = out["edge_idx"]

                # (nedges, (2*irrep_dim+1))
                spharm = o3.spherical_harmonics(irrep, edge_vec, True).detach()
                # (nedges, 1)
                pred = self.module_dict[target](out["edge_embedding"])
                # (nedges, 2*irrep-dim+1)
                pred = pred * spharm
                # aggregate edges per node
                # (nnodes, 2*irrep-dim+1)
                pred = scatter_det(
                    pred, edge_idx, dim=0, dim_size=num_atoms, reduce="add"
                )
                # TODO: Add support for equivariant models internal spherical harmonics
            else:
                pred = self.module_dict[target](out["node_embedding"])

            if self.output_targets[target].get("level", "system") == "system":
                pred = scatter_det(
                    pred,
                    batch,
                    dim=0,
                    dim_size=num_systems,
                    reduce=self.output_targets[target].get("reduce", "add"),
                )

            results[target] = pred.squeeze(1)

        return results

    def generate_graph(
        self,
        data,
        cutoff=None,
        max_neighbors=None,
        use_pbc=None,
        otf_graph=None,
        enforce_max_neighbors_strictly=None,
    ):
        cutoff = cutoff or self.cutoff
        max_neighbors = max_neighbors or self.max_neighbors
        use_pbc = use_pbc or self.use_pbc
        otf_graph = otf_graph or self.otf_graph

        if enforce_max_neighbors_strictly is not None:
            pass
        elif hasattr(self, "enforce_max_neighbors_strictly"):
            # Not all models will have this attribute
            enforce_max_neighbors_strictly = (
                self.enforce_max_neighbors_strictly
            )
        else:
            # Default to old behavior
            enforce_max_neighbors_strictly = True

        if not otf_graph:
            try:
                edge_index = data.edge_index

                if use_pbc:
                    cell_offsets = data.cell_offsets
                    neighbors = data.neighbors

            except AttributeError:
                logging.warning(
                    "Turning otf_graph=True as required attributes not present in data object"
                )
                otf_graph = True

        if use_pbc:
            if otf_graph:
                edge_index, cell_offsets, neighbors = radius_graph_pbc(
                    data,
                    cutoff,
                    max_neighbors,
                    enforce_max_neighbors_strictly,
                )

            out = get_pbc_distances(
                data.pos,
                edge_index,
                data.cell,
                cell_offsets,
                neighbors,
                return_offsets=True,
                return_distance_vec=True,
            )

            edge_index = out["edge_index"]
            edge_dist = out["distances"]
            cell_offset_distances = out["offsets"]
            distance_vec = out["distance_vec"]
        else:
            if otf_graph:
                edge_index = radius_graph(
                    data.pos,
                    r=cutoff,
                    batch=data.batch,
                    max_num_neighbors=max_neighbors,
                )

            j, i = edge_index
            distance_vec = data.pos[j] - data.pos[i]

            edge_dist = distance_vec.norm(dim=-1)
            cell_offsets = torch.zeros(
                edge_index.shape[1], 3, device=data.pos.device
            )
            cell_offset_distances = torch.zeros_like(
                cell_offsets, device=data.pos.device
            )
            neighbors = compute_neighbors(data, edge_index)

        return (
            edge_index,
            edge_dist,
            distance_vec,
            cell_offsets,
            cell_offset_distances,
            neighbors,
        )

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
