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
from pymatgen.io.ase import AseAtomsAdaptor

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
