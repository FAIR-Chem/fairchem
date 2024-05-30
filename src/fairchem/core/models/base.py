"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph

from fairchem.core.common.utils import (
    compute_neighbors,
    get_pbc_distances,
    radius_graph_pbc,
)


class BaseModel(nn.Module):
    def __init__(self, num_atoms=None, bond_feat_dim=None, num_targets=None) -> None:
        super().__init__()
        self.num_atoms = num_atoms
        self.bond_feat_dim = bond_feat_dim
        self.num_targets = num_targets

    def forward(self, data):
        raise NotImplementedError

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
                edge_index_per_system, cell_offsets_per_system, neighbors_per_system = (
                    list(
                        zip(
                            *[
                                radius_graph_pbc(
                                    data[idx],
                                    cutoff,
                                    max_neighbors,
                                    enforce_max_neighbors_strictly,
                                )
                                for idx in range(len(data))
                            ]
                        )
                    )
                )

                # atom indexs in the edge_index need to be offset
                atom_index_offset = data.natoms.cumsum(dim=0).roll(1)
                atom_index_offset[0] = 0
                edge_index = torch.hstack(
                    [
                        edge_index_per_system[idx] + atom_index_offset[idx]
                        for idx in range(len(data))
                    ]
                )
                cell_offsets = torch.vstack(cell_offsets_per_system)
                neighbors = torch.hstack(neighbors_per_system)

                ## TODO this is the original call, but blows up with memory
                ## using two different samples
                ## sid='mp-675045-mp-675045-0-7' (MPTRAJ)
                ## sid='75396' (OC22)
                # edge_index, cell_offsets, neighbors = radius_graph_pbc(
                #     data,
                #     cutoff,
                #     max_neighbors,
                #     enforce_max_neighbors_strictly,
                # )
                # assert (_edge_index == edge_index).all().item()
                # assert (_cell_offsets == cell_offsets).all().item()
                # assert (_neighbors == neighbors).all().item()

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

    @torch.jit.ignore
    def no_weight_decay(self) -> list:
        """Returns a list of parameters with no weight decay."""
        no_wd_list = []
        for name, _ in self.named_parameters():
            if "embedding" in name or "frequencies" in name or "bias" in name:
                no_wd_list.append(name)
        return no_wd_list
