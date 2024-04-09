"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os

import pytest
import torch
from ase.build import molecule
from ase.io import read
from ase.lattice.cubic import FaceCenteredCubic
from torch_geometric.transforms.radius_graph import RadiusGraph
from torch_geometric.utils.sort_edge_index import sort_edge_index

from ocpmodels.common.utils import radius_graph_pbc
from ocpmodels.datasets import data_list_collater
from ocpmodels.preprocessing import AtomsToGraphs


@pytest.fixture(scope="class")
def load_data(request) -> None:
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


def check_features_match(
    edge_index_1, cell_offsets_1, edge_index_2, cell_offsets_2
) -> bool:
    # Combine both edge indices and offsets to one tensor
    features_1 = torch.cat((edge_index_1, cell_offsets_1.T), dim=0).T
    features_2 = torch.cat((edge_index_2, cell_offsets_2.T), dim=0).T.long()

    # Convert rows of tensors to sets. The order of edges is not guaranteed
    features_1_set = {tuple(x.tolist()) for x in features_1}
    features_2_set = {tuple(x.tolist()) for x in features_2}

    # Ensure sets are not empty
    assert len(features_1_set) > 0
    assert len(features_2_set) > 0

    # Ensure sets are the same
    assert features_1_set == features_2_set

    return True


@pytest.mark.usefixtures("load_data")
class TestRadiusGraphPBC:
    def test_radius_graph_pbc(self) -> None:
        data = self.data
        batch = data_list_collater([data] * 5)

        edge_index, cell_offsets, neighbors = radius_graph_pbc(
            batch,
            radius=6,
            max_num_neighbors_threshold=2000,
            pbc=[True, True, False],
        )

        assert check_features_match(
            batch.edge_index, batch.cell_offsets, edge_index, cell_offsets
        )

    def test_bulk(self) -> None:
        radius = 10

        # Must be sufficiently large to ensure all edges are retained
        max_neigh = 2000

        a2g = AtomsToGraphs(radius=radius, max_neigh=max_neigh)
        structure = FaceCenteredCubic("Pt", size=[1, 2, 3])
        data = a2g.convert(structure)
        batch = data_list_collater([data])

        # Ensure adequate distance between repeated cells
        structure.cell[0] *= radius
        structure.cell[1] *= radius
        structure.cell[2] *= radius

        # [False, False, False]
        data = a2g.convert(structure)
        non_pbc = data.edge_index.shape[1]

        out = radius_graph_pbc(
            batch,
            radius=radius,
            max_num_neighbors_threshold=max_neigh,
            pbc=[False, False, False],
        )

        assert check_features_match(
            data.edge_index, data.cell_offsets, out[0], out[1]
        )

        # [True, False, False]
        structure.cell[0] /= radius
        data = a2g.convert(structure)
        pbc_x = data.edge_index.shape[1]

        out = radius_graph_pbc(
            batch,
            radius=radius,
            max_num_neighbors_threshold=max_neigh,
            pbc=[True, False, False],
        )
        assert check_features_match(
            data.edge_index, data.cell_offsets, out[0], out[1]
        )

        # [True, True, False]
        structure.cell[1] /= radius
        data = a2g.convert(structure)
        pbc_xy = data.edge_index.shape[1]

        out = radius_graph_pbc(
            batch,
            radius=radius,
            max_num_neighbors_threshold=max_neigh,
            pbc=[True, True, False],
        )
        assert check_features_match(
            data.edge_index, data.cell_offsets, out[0], out[1]
        )

        # [False, True, False]
        structure.cell[0] *= radius
        data = a2g.convert(structure)
        pbc_y = data.edge_index.shape[1]

        out = radius_graph_pbc(
            batch,
            radius=radius,
            max_num_neighbors_threshold=max_neigh,
            pbc=[False, True, False],
        )
        assert check_features_match(
            data.edge_index, data.cell_offsets, out[0], out[1]
        )

        # [False, True, True]
        structure.cell[2] /= radius
        data = a2g.convert(structure)
        pbc_yz = data.edge_index.shape[1]

        out = radius_graph_pbc(
            batch,
            radius=radius,
            max_num_neighbors_threshold=max_neigh,
            pbc=[False, True, True],
        )
        assert check_features_match(
            data.edge_index, data.cell_offsets, out[0], out[1]
        )

        # [False, False, True]
        structure.cell[1] *= radius
        data = a2g.convert(structure)
        pbc_z = data.edge_index.shape[1]

        out = radius_graph_pbc(
            batch,
            radius=radius,
            max_num_neighbors_threshold=max_neigh,
            pbc=[False, False, True],
        )
        assert check_features_match(
            data.edge_index, data.cell_offsets, out[0], out[1]
        )

        # [True, False, True]
        structure.cell[0] /= radius
        data = a2g.convert(structure)
        pbc_xz = data.edge_index.shape[1]

        out = radius_graph_pbc(
            batch,
            radius=radius,
            max_num_neighbors_threshold=max_neigh,
            pbc=[True, False, True],
        )
        assert check_features_match(
            data.edge_index, data.cell_offsets, out[0], out[1]
        )

        # [True, True, True]
        structure.cell[1] /= radius
        data = a2g.convert(structure)
        pbc_all = data.edge_index.shape[1]

        out = radius_graph_pbc(
            batch,
            radius=radius,
            max_num_neighbors_threshold=max_neigh,
            pbc=[True, True, True],
        )

        assert check_features_match(
            data.edge_index, data.cell_offsets, out[0], out[1]
        )

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

        # Ensure radius_graph_pbc matches radius_graph for non-PBC condition
        RG = RadiusGraph(r=radius, max_num_neighbors=max_neigh)
        radgraph = RG(batch)

        out = radius_graph_pbc(
            batch,
            radius=radius,
            max_num_neighbors_threshold=max_neigh,
            pbc=[False, False, False],
        )
        assert (
            sort_edge_index(out[0]) == sort_edge_index(radgraph.edge_index)
        ).all()

    def test_molecule(self) -> None:
        radius = 6
        max_neigh = 1000
        a2g = AtomsToGraphs(radius=radius, max_neigh=max_neigh)
        structure = molecule("CH3COOH")
        structure.cell = [[20, 0, 0], [0, 20, 0], [0, 0, 20]]
        data = a2g.convert(structure)
        batch = data_list_collater([data])
        out = radius_graph_pbc(
            batch,
            radius=radius,
            max_num_neighbors_threshold=max_neigh,
            pbc=[False, False, False],
        )

        assert check_features_match(
            data.edge_index, data.cell_offsets, out[0], out[1]
        )
