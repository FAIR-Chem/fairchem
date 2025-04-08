"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import requests
import torch
from ase.build import molecule

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import load_state_dict, setup_imports
from fairchem.core.datasets import data_list_collater
from fairchem.core.preprocessing import AtomsToGraphs


@pytest.fixture(scope="class")
def load_data(request) -> None:
    molecules = [
        molecule("bicyclobutane"),
        molecule("CH4"),
        molecule("H2O"),
        molecule("O2"),
    ]

    a2g = AtomsToGraphs(
        max_neigh=200,
        radius=6,
        r_energy=False,
        r_forces=False,
        molecule_cell_size=120,
    )
    data_list = a2g.convert_all(molecules)
    request.cls.data_list = data_list


@pytest.fixture(scope="class")
def load_model(request) -> None:
    torch.manual_seed(4)
    setup_imports()

    # download and load weights.
    checkpoint_url = "https://dl.fbaipublicfiles.com/opencatalystproject/models/2022_07/s2ef/gemnet_oc_base_s2ef_all.pt"

    # load buffer into memory as a stream
    # and then load it with torch.load
    r = requests.get(checkpoint_url, stream=True)
    r.raise_for_status()
    checkpoint = torch.load(io.BytesIO(r.content), map_location=torch.device("cpu"))

    model = registry.get_model_class("gemnet_oc")(
        num_spherical=7,
        num_radial=128,
        num_blocks=4,
        emb_size_atom=256,
        emb_size_edge=512,
        emb_size_trip_in=64,
        emb_size_trip_out=64,
        emb_size_quad_in=32,
        emb_size_quad_out=32,
        emb_size_aint_in=64,
        emb_size_aint_out=64,
        emb_size_rbf=16,
        emb_size_cbf=16,
        emb_size_sbf=32,
        num_before_skip=2,
        num_after_skip=2,
        num_concat=1,
        num_atom=3,
        num_output_afteratom=3,
        num_atom_emb_layers=2,
        num_global_out_layers=2,
        regress_forces=True,
        direct_forces=True,
        use_pbc=True,
        cutoff=12.0,
        cutoff_qint=12.0,
        cutoff_aeaint=12.0,
        cutoff_aint=12.0,
        max_neighbors=30,
        max_neighbors_qint=8,
        max_neighbors_aeaint=20,
        max_neighbors_aint=1000,
        rbf={"name": "gaussian"},
        envelope={"name": "polynomial", "exponent": 5},
        cbf={"name": "spherical_harmonics"},
        sbf={"name": "legendre_outer"},
        extensive=True,
        forces_coupled=False,
        output_init="HeOrthogonal",
        activation="silu",
        quad_interaction=True,
        atom_edge_interaction=True,
        edge_atom_interaction=True,
        atom_interaction=True,
        qint_tags=[0, 1, 2],
        scale_file=checkpoint["scale_dict"],
    )

    new_dict = {k[len("module.") * 2 :]: v for k, v in checkpoint["state_dict"].items()}
    load_state_dict(model, new_dict)

    request.cls.model = model


@pytest.mark.usefixtures("load_data")
@pytest.mark.usefixtures("load_model")
class TestGemNetOC:
    def test_batched_consistency(self, snapshot) -> None:
        # Recreate the Data object to only keep the necessary features.
        batch = data_list_collater(self.data_list)

        # Pass it through the model.
        batched_outputs = self.model(batch)
        batched_energy, batched_forces = (
            batched_outputs["energy"],
            batched_outputs["forces"],
        )

        single_energy = []
        single_forces = []
        for data in self.data_list:
            outputs = self.model(data_list_collater([data]))
            energy, forces = outputs["energy"], outputs["forces"]
            single_energy.append(energy)
            single_forces.append(forces)

        single_energy = torch.cat(single_energy)
        single_forces = torch.cat(single_forces)

        np.testing.assert_array_almost_equal(
            batched_energy.detach(), single_energy.detach(), decimal=3
        )
        np.testing.assert_array_almost_equal(
            batched_forces.detach(), single_forces.detach(), decimal=3
        )

    def test_quad_consistency(self, snapshot) -> None:
        # Recreate the Data object to only keep the necessary features.
        water_atoms = self.data_list[2]
        pair_atoms = self.data_list[3]
        batch = data_list_collater([water_atoms, pair_atoms])

        # Pass it through the model.
        outputs = self.model(batch)
        quad_energies, quad_forces = outputs["energy"], outputs["forces"]

        # Disable quad interactions in the model
        self.model.quad_interaction = False
        for idx in range(len(self.model.int_blocks)):
            self.model.int_blocks[idx].quad_interaction = None

        # Pass it through the model.
        outputs = self.model(batch)
        no_quad_energies, no_quad_forces = outputs["energy"], outputs["forces"]

        np.testing.assert_array_almost_equal(
            quad_energies.detach(), no_quad_energies.detach(), decimal=3
        )
        np.testing.assert_array_almost_equal(
            quad_forces.detach(), no_quad_forces.detach(), decimal=3
        )
