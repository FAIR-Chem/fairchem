"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os

import numpy as np
import pytest
from ase.io import read

from ocpmodels.datasets import data_list_collater
from ocpmodels.models import ForceNet
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
    model = ForceNet(
        None,
        32,
        1,
        cutoff=6.0,
    )
    request.cls.model = model


@pytest.mark.usefixtures("load_data")
@pytest.mark.usefixtures("load_model")
class TestForceNet:
    def test_energy_force_shape(self, snapshot):
        # Recreate the Data object to only keep the necessary features.
        data = self.data

        # Pass it through the model.
        energy, forces = self.model(data_list_collater([data]))

        assert snapshot == energy.shape
        assert snapshot == pytest.approx(energy.detach(), rel=1e-5, abs=1e-8)

        assert snapshot == forces.shape
        assert snapshot == pytest.approx(forces.detach(), rel=1e-5, abs=1e-8)
