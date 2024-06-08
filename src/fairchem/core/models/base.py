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
from torch_geometric.utils import to_dense_adj

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
        loop=False,
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
                    loop,
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
                    loop=loop,
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
    
    def calc_lap(
        self,
        data,
        edge_index,
        lap_dim
    ):
        lap_vec = lap_eigvec(data.batch, edge_index, data.natoms, dim=lap_dim)
        return lap_vec

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

def lap_eigvec(batch, edge_index, natoms, dim=128):
    """
    Graph positional encoding v/ Laplacian eigenvectors
    https://github.com/DevinKreuzer/SAN/blob/main/data/molecules.py
    Modified to git OTF graph construction
    """
    output = torch.zeros(len(batch), dim, device=batch.device)
    for i, offset in enumerate(torch.cumsum(natoms, dim=0) - natoms):
        edges = edge_index[:, (batch[edge_index] == i).all(0)]
        edges = edges - offset
        dense_adj = to_dense_adj(edges, max_num_nodes=natoms[i]).float()[0]
        in_degree = dense_adj.sum(1)
        # Laplacian
        A = dense_adj
        N = torch.diag(in_degree.clip(1) ** -0.5)
        L = torch.eye(natoms[i], device=natoms.device) - N @ A @ N

        eigval, eigvec = torch.linalg.eigh(L)
        if natoms[i] <= dim:
            output[offset: offset+natoms[i], :natoms[i]] = eigvec.float()
        else:
            output[offset: offset+natoms[i]] = eigvec[:, :dim].float()

    return output 