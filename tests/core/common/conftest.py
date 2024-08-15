from __future__ import annotations

import pytest
from ase import build

from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from fairchem.core.datasets import data_list_collater
from fairchem.core.preprocessing.atoms_to_graphs import AtomsToGraphs


@pytest.fixture(scope="session")
def calculator(tmp_path_factory):
    dir = tmp_path_factory.mktemp("checkpoints")
    calc = OCPCalculator(
        model_name="EquiformerV2-31M-S2EF-OC20-All+MD", local_cache=dir, seed=0
    )
    # TODO debug this
    # removing amp so that we always get float32 predictions
    calc.trainer.scaler = None
    return calc


@pytest.fixture()
def atoms_list():
    atoms_list = [
        build.bulk("Cu", "fcc", a=3.8, cubic=True),
        build.bulk("NaCl", crystalstructure="rocksalt", a=5.8),
    ]
    for atoms in atoms_list:
        atoms.rattle(stdev=0.05, seed=0)
    return atoms_list


@pytest.fixture()
def batch(atoms_list):
    a2g = AtomsToGraphs(r_edges=False, r_pbc=True)
    return data_list_collater([a2g.convert(atoms) for atoms in atoms_list])
