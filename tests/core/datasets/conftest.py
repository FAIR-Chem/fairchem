import numpy as np
import pytest
from ase import build
from ase.calculators.singlepoint import SinglePointCalculator


@pytest.fixture(scope="module")
def structures():
    structures = [
        build.molecule("H2O", vacuum=4),
        build.bulk("Cu"),
        build.fcc111("Pt", size=[2, 2, 3], vacuum=8, periodic=True),
    ]
    for atoms in structures:
        calc = SinglePointCalculator(
            atoms,
            energy=1,
            forces=atoms.positions,
            # there is an issue with ASE db when writing a db with 3x3 stress if is flattened to (9,) and then
            # errors when trying to read it
            stress=np.random.random((6,)),
        )
        atoms.calc = calc
        atoms.info["extensive_property"] = 3 * len(atoms)
        atoms.info["tensor_property"] = np.random.random((6, 6))

    structures[2].set_pbc(True)
    return structures
