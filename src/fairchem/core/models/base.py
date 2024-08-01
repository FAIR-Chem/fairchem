"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch_geometric.nn import radius_graph

from fairchem.core.common.utils import (
    compute_neighbors,
    get_pbc_distances,
    radius_graph_pbc,
)


@dataclass
class GraphData:
    """Class to keep graph attributes nicely packaged."""

    edge_index: torch.Tensor
    edge_distance: torch.Tensor
    edge_distance_vec: torch.Tensor
    cell_offsets: torch.Tensor
    offset_distances: torch.Tensor
    neighbors: torch.Tensor
    batch_full: torch.Tensor  # used for GP functionality
    atomic_numbers_full: torch.Tensor  # used for GP functionality
    node_offset: int = 0  # used for GP functionality


class GraphModelMixin:
    """Mixin Model class implementing some general convenience properties and methods."""

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
            enforce_max_neighbors_strictly = self.enforce_max_neighbors_strictly
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
            cell_offsets = torch.zeros(edge_index.shape[1], 3, device=data.pos.device)
            cell_offset_distances = torch.zeros_like(
                cell_offsets, device=data.pos.device
            )
            neighbors = compute_neighbors(data, edge_index)

        return GraphData(
            edge_index=edge_index,
            edge_distance=edge_dist,
            edge_distance_vec=distance_vec,
            cell_offsets=cell_offsets,
            offset_distances=cell_offset_distances,
            neighbors=neighbors,
            node_offset=0,
            batch_full=data.batch,
            atomic_numbers_full=data.atomic_numbers.long(),
        )

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.jit.ignore
    def no_weight_decay(self) -> list:
        """Returns a list of parameters with no weight decay."""
        no_wd_list = []
        for name, _ in self.named_parameters():
            if "embedding" in name or "frequencies" in name or "bias" in name:
                no_wd_list.append(name)
        return no_wd_list
