"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import random

import numpy as np
import pytest
import torch
from ase.io import read
from torch_geometric.data import Data

from ocpmodels.common.registry import registry
from ocpmodels.common.transforms import RandomRotate
from ocpmodels.common.utils import setup_imports
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


@pytest.fixture(scope="class")
def load_model(request):
    setup_imports()

    model = registry.get_model_class("forcenet")(
        None,
        32,
        1,
        cutoff=6.0,
    )
    request.cls.model = model


@pytest.mark.usefixtures("load_data")
@pytest.mark.usefixtures("load_model")
class TestForceNet:
    def test_energy_force_shape(self):
        data = self.data

        # Pass it through the model.
        out = self.model(data_list_collater([data]))

        # Compare shape of predicted energies, forces.
        energy = out[0].detach()
        np.testing.assert_equal(energy.shape, (1, 1))

        forces = out[1].detach()
        np.testing.assert_equal(forces.shape, (data.pos.shape[0], 3))
