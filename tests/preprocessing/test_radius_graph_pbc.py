"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os

import ase
import numpy as np
import pytest
import torch
from ase.io import read
from ase.lattice.cubic import FaceCenteredCubic
from ase.build import molecule
from pymatgen.io.ase import AseAtomsAdaptor
from torch_geometric.transforms.radius_graph import RadiusGraph
from torch_geometric.utils.sort_edge_index import sort_edge_index

from ocpmodels.common.utils import get_pbc_distances, radius_graph_pbc
from ocpmodels.datasets import data_list_collater
from ocpmodels.preprocessing import AtomsToGraphs


@pytest.fixture(scope="class")
def load_data(request):
    atoms = read(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "atoms.json"),
        index=0,
        format="json",
    )
    a2g = AtomsToGraphs(
        max_neigh=200,
        radius=6,
        r_energy=True,
        r_forces=True,
        r_distances=True,
    )
    data_list = a2g.convert_all([atoms])
    request.cls.data = data_list[0]


@pytest.mark.usefixtures("load_data")
class TestRadiusGraphPBC:
    def test_radius_graph_pbc(self):
        data = self.data
        batch = data_list_collater([data] * 5)

        out = radius_graph_pbc(
            batch,
            radius=6,
            max_num_neighbors_threshold=200,
            pbc=[True, True, False],
        )

        edge_index, cell_offsets, neighbors = out

        # Combine both edge indices and offsets to one tensor
        a2g_features = torch.cat(
            (batch.edge_index, batch.cell_offsets.T), dim=0
        ).T
        rgpbc_features = torch.cat(
            (edge_index, cell_offsets.T), dim=0
        ).T.long()

        # Convert rows of tensors to sets. The order of edges is not guaranteed
        a2g_features = {tuple(x.tolist()) for x in a2g_features}
        rgpbc_features = {tuple(x.tolist()) for x in rgpbc_features}

        # Ensure sets are not empty
        assert len(a2g_features) > 0
        assert len(rgpbc_features) > 0

        # Ensure sets are the same
        assert a2g_features == rgpbc_features

    def test_bulk(self):
        radius = 10

        # Must be sufficiently large to ensure all edges are retained
        max_neigh = 2000

        a2g = AtomsToGraphs(radius=radius, max_neigh=max_neigh)
        structure = FaceCenteredCubic("Pt", size=[1, 2, 3])

        # Use the radius as a multiplier to ensure adequate distance between repeated cells
        structure.cell[0] *= radius
        structure.cell[1] *= radius
        structure.cell[2] *= radius

        data = a2g.convert(structure)
        non_pbc = data.edge_index.shape[1]

        # Get number of neighbors for all possible PBC combinations
        structure.cell[0] /= radius
        data = a2g.convert(structure)
        pbc_x = data.edge_index.shape[1]

        structure.cell[1] /= radius
        data = a2g.convert(structure)
        pbc_xy = data.edge_index.shape[1]

        structure.cell[0] *= radius
        data = a2g.convert(structure)
        pbc_y = data.edge_index.shape[1]

        structure.cell[2] /= radius
        data = a2g.convert(structure)
        pbc_yz = data.edge_index.shape[1]

        structure.cell[1] *= radius
        data = a2g.convert(structure)
        pbc_z = data.edge_index.shape[1]

        structure.cell[0] /= radius
        data = a2g.convert(structure)
        pbc_xz = data.edge_index.shape[1]

        structure.cell[1] /= radius
        data = a2g.convert(structure)
        pbc_all = data.edge_index.shape[1]

        # Ensure edges are actually found
        assert non_pbc > 0
        assert pbc_x > non_pbc
        assert pbc_y > non_pbc
        assert pbc_z > non_pbc
        assert pbc_xy > max(pbc_x, pbc_y)
        assert pbc_yz > max(pbc_y, pbc_z)
        assert pbc_xz > max(pbc_x, pbc_z)
        assert pbc_all > max(pbc_xy, pbc_yz, pbc_xz)

        structure = FaceCenteredCubic("Pt", size=[1, 2, 3])
        data = a2g.convert(structure)

        batch = data_list_collater([data])

        # Ensure radius_graph_pbc matches radius_graph for non-PBC condition
        RG = RadiusGraph(r=radius, max_num_neighbors=max_neigh)

        out = radius_graph_pbc(
            batch,
            radius=radius,
            max_num_neighbors_threshold=max_neigh,
            pbc=[False, False, False],
        )
        assert out[-1].item() == non_pbc

        radgraph = RG(batch)
        assert (
            sort_edge_index(out[0]) == sort_edge_index(radgraph.edge_index)
        ).all()

        # Ensure radius_graph_pbc matches AtomsToGraphs for all PBC combinations
        out = radius_graph_pbc(
            batch,
            radius=radius,
            max_num_neighbors_threshold=max_neigh,
            pbc=[True, False, False],
        )
        assert out[-1].item() == pbc_x

        out = radius_graph_pbc(
            batch,
            radius=radius,
            max_num_neighbors_threshold=max_neigh,
            pbc=[False, True, False],
        )
        assert out[-1].item() == pbc_y

        out = radius_graph_pbc(
            batch,
            radius=radius,
            max_num_neighbors_threshold=max_neigh,
            pbc=[False, False, True],
        )
        assert out[-1].item() == pbc_z

        out = radius_graph_pbc(
            batch,
            radius=radius,
            max_num_neighbors_threshold=max_neigh,
            pbc=[True, True, False],
        )
        assert out[-1].item() == pbc_xy

        out = radius_graph_pbc(
            batch,
            radius=radius,
            max_num_neighbors_threshold=max_neigh,
            pbc=[False, True, True],
        )
        assert out[-1].item() == pbc_yz

        out = radius_graph_pbc(
            batch,
            radius=radius,
            max_num_neighbors_threshold=max_neigh,
            pbc=[True, False, True],
        )
        assert out[-1].item() == pbc_xz

        out = radius_graph_pbc(
            batch,
            radius=radius,
            max_num_neighbors_threshold=max_neigh,
            pbc=[True, True, True],
        )
        assert out[-1].item() == pbc_all

    def test_molecule(self):
        radius = 6
        max_neigh = 100
        a2g = AtomsToGraphs(radius=radius, max_neigh=max_neigh)
        structure = molecule("CH3COOH")
        structure.cell = [[20, 0, 0], [0, 20, 0], [0, 0, 20]]
        data = a2g.convert(structure)
        batch = data_list_collater([data] * 5)
        out = radius_graph_pbc(
            batch,
            radius=radius,
            max_num_neighbors_threshold=max_neigh,
            pbc=[False, False, False],
        )
        edge_index, cell_offsets, neighbors = out

        # Combine both edge indices and offsets to one tensor
        a2g_features = torch.cat(
            (batch.edge_index, batch.cell_offsets.T), dim=0
        ).T
        rgpbc_features = torch.cat(
            (edge_index, cell_offsets.T), dim=0
        ).T.long()

        # Convert rows of tensors to sets. The order of edges is not guaranteed
        a2g_features = {tuple(x.tolist()) for x in a2g_features}
        rgpbc_features = {tuple(x.tolist()) for x in rgpbc_features}

        # Ensure sets are not empty
        assert len(a2g_features) > 0
        assert len(rgpbc_features) > 0

        # Ensure sets are the same
        assert a2g_features == rgpbc_features
