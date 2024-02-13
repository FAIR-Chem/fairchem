"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import io
import random

import pytest
import requests
import torch
from ase.build import add_adsorbate, fcc111
from ase.optimize import BFGS

from ocpmodels.common.relaxation.ase_utils import OCPCalculator


# Following the OCP tutorial: https://github.com/Open-Catalyst-Project/tutorial
@pytest.fixture(scope="class")
def load_data(request) -> None:
    atoms = fcc111("Pt", size=(2, 2, 5), vacuum=10.0)
    add_adsorbate(atoms, "O", height=1.2, position="fcc")
    request.cls.atoms = atoms


# First let's just make sure all checkpoints are being loaded without any
# errors as part of the ASE calculator setup.
@pytest.mark.parametrize(
    "model_url",
    [
        # SchNet
        "https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/s2ef/schnet_all_large.pt",
        # DieNet++
        "https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_02/s2ef/dimenetpp_all.pt",
        # GemNet-dT
        "https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_08/s2ef/gemnet_t_direct_h512_all.pt",
        # PaiNN
        "https://dl.fbaipublicfiles.com/opencatalystproject/models/2022_05/s2ef/painn_h512_s2ef_all.pt",
        # GemNet-OC
        "https://dl.fbaipublicfiles.com/opencatalystproject/models/2023_03/s2ef/gemnet_oc_base_s2ef_all_md.pt",
        # SCN
        "https://dl.fbaipublicfiles.com/opencatalystproject/models/2023_03/s2ef/scn_all_md_s2ef.pt",
        # Equiformer v2  # already tested in test_relaxation_final_energy
        # "https://dl.fbaipublicfiles.com/opencatalystproject/models/2023_06/oc20/s2ef/eq2_153M_ec4_allmd.pt",
        # eSCNm # already tested in test_random_seed_final_energy
        # "https://dl.fbaipublicfiles.com/opencatalystproject/models/2023_03/s2ef/escn_l6_m3_lay20_all_md_s2ef.pt",
    ],
)
def test_calculator_setup(model_url):
    r = requests.get(model_url)
    r.raise_for_status()
    _ = OCPCalculator(checkpoint_path=io.BytesIO(r.content), cpu=True)


@pytest.mark.usefixtures("load_data")
class TestCalculator:
    # Run an adslab relaxation using the ASE calculator and ase.optimize.BFGS
    # with one model and compare the final energy.
    def test_relaxation_final_energy(self, snapshot) -> None:
        random.seed(1)
        torch.manual_seed(1)

        equiformerv2_url = "https://dl.fbaipublicfiles.com/opencatalystproject/models/2023_06/oc20/s2ef/eq2_153M_ec4_allmd.pt"
        r = requests.get(equiformerv2_url, stream=True)
        r.raise_for_status()
        calc = OCPCalculator(checkpoint_path=io.BytesIO(r.content), cpu=True)

        atoms = self.atoms
        atoms.set_calculator(calc)
        opt = BFGS(atoms)
        opt.run(fmax=0.05, steps=100)

        assert snapshot == round(atoms.get_potential_energy(), 2)

    def test_random_seed_final_energy(self, snapshot):
        escn_url = "https://dl.fbaipublicfiles.com/opencatalystproject/models/2023_03/s2ef/escn_l6_m3_lay20_all_md_s2ef.pt"
        r = requests.get(escn_url, stream=True)
        r.raise_for_status()
        seeds = [100, 200, 100]
        results_by_seed = {}
        # compute the value for each seed , make sure repeated seeds have the exact same output
        for seed in seeds:
            calc = OCPCalculator(
                checkpoint_path=io.BytesIO(r.content), cpu=True, seed=seed
            )
            slab = fcc111("Pt", size=(2, 2, 5), vacuum=10.0)
            add_adsorbate(slab, "O", height=1.2, position="fcc")
            slab.set_calculator(calc)
            energy = slab.get_potential_energy()
            if seed in results_by_seed:
                assert results_by_seed[seed] == energy
            else:
                results_by_seed[seed] = energy
        # make sure different seeds give slightly different results , expected due to discretization error in grid
        for seed_a in set(seeds):
            for seed_b in set(seeds) - set([seed_a]):
                assert results_by_seed[seed_a] != results_by_seed[seed_b]
