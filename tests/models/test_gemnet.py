"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import os
import random

import numpy as np
import pytest
import torch
from ase.io import read
from torch_geometric.data import Data

from ocpmodels.common.transforms import RandomRotate
from ocpmodels.datasets import data_list_collater
from ocpmodels.models import GemNetT
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


@pytest.fixture(scope="class")
def load_model(request):
    torch.manual_seed(4)
    model = GemNetT(
        None,
        -1,
        1,
        cutoff=6.0,
        num_spherical=7,
        num_radial=128,
        num_blocks=3,
        emb_size_atom=16,
        emb_size_edge=16,
        emb_size_trip=16,
        emb_size_rbf=16,
        emb_size_cbf=16,
        emb_size_bil_trip=64,
        num_before_skip=1,
        num_after_skip=2,
        num_concat=1,
        num_atom=3,
        regress_forces=True,
        direct_forces=True,
        scale_file=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "gemnet-dT-scales.json"
        ),
    )
    request.cls.model = model


@pytest.mark.usefixtures("load_data")
@pytest.mark.usefixtures("load_model")
class TestGemNetT:
    def test_rotation_invariance(self):
        random.seed(1)
        data = self.data

        # Sampling a random rotation within [-180, 180] for all axes.
        transform = RandomRotate([-180, 180], [0, 1, 2])
        data_rotated, rot, inv_rot = transform(data.clone())
        assert not np.array_equal(data.pos, data_rotated.pos)

        # Pass it through the model.
        batch = data_list_collater([data, data_rotated])
        out = self.model(batch)

        # Compare predicted energies and forces (after inv-rotation).
        energies = out[0].detach()
        np.testing.assert_almost_equal(energies[0], energies[1], decimal=5)

        forces = out[1].detach()
        logging.info(forces)
        np.testing.assert_array_almost_equal(
            forces[: forces.shape[0] // 2],
            torch.matmul(forces[forces.shape[0] // 2 :], inv_rot),
            decimal=4,
        )

    def test_energy_force_shape(self):
        data = self.data

        # Pass it through the model.
        out = self.model(data_list_collater([data]))

        # Compare shape of predicted energies, forces.
        energy = out[0].detach()
        np.testing.assert_equal(energy.shape, (1, 1))

        forces = out[1].detach()
        np.testing.assert_equal(forces.shape, (data.pos.shape[0], 3))
