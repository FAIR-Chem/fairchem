import pytest

import numpy as np
import numpy.testing as npt

from ase import build
from ase.optimize import LBFGS, FIRE
from ase.filters import UnitCellFilter
from fairchem.core.datasets import data_list_collater
from fairchem.core.preprocessing.atoms_to_graphs import AtomsToGraphs
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from fairchem.core.common.relaxation import OptimizableBatch, OptimizableUnitCellBatch


@pytest.fixture
def calculator(tmpdir):
    calc = OCPCalculator(
        model_name="EquiformerV2-31M-S2EF-OC20-All+MD", local_cache=tmpdir, seed=0
    )
    # TODO debug this
    # removing amp so that we always get float32 predictions
    calc.trainer.scaler = None
    return calc


@pytest.fixture(scope="function")
def atoms_list():
    atoms_list = [
        build.bulk("Cu", "fcc", a=4.0, cubic=True),
        build.bulk("NaCl", crystalstructure="rocksalt", a=3.5),
    ]
    return atoms_list


@pytest.fixture(scope="function")
def batch(atoms_list):
    a2g = AtomsToGraphs(r_edges=False, r_pbc=True)
    return data_list_collater([a2g.convert(atoms) for atoms in atoms_list])


@pytest.fixture(params=[LBFGS, FIRE])
def optimizer_cls(request):
    return request.param


def test_ase_relaxation(atoms_list, batch, calculator, optimizer_cls):
    """Tests batch relaxation using ASE optimizers."""
    obatch = OptimizableBatch(batch, trainer=calculator.trainer, numpy=True)

    # optimize atoms in batch using ASE
    batch_optimizer = optimizer_cls(obatch)
    batch_optimizer.run(0.05, 50)

    # optimize atoms one-by-one
    for atoms in atoms_list:
        atoms.calc = calculator
        opt = optimizer_cls(atoms)
        opt.run(0.05, 50)

    # compare energy and atom positions
    for a1, a2 in zip(atoms_list, obatch.get_atoms_list()):
        assert a1.get_potential_energy() / len(a1) == pytest.approx(
            a2.get_potential_energy() / len(a2), abs=0.05
        )
        pnorm1 = np.linalg.norm(a1.positions, axis=1)
        pnorm2 = np.linalg.norm(a2.positions, axis=1)
        npt.assert_allclose(pnorm1, pnorm2, atol=0.05)


@pytest.mark.skip("Skip until we have a test model that can predict stress")
def test_ase_cell_relaxation(atoms_list, batch, calculator, optimizer_cls):
    """Tests batch relaxation using ASE optimizers."""

    cell_factor = batch.natoms.cpu().numpy().mean()
    obatch = OptimizableUnitCellBatch(
        batch, trainer=calculator.trainer, numpy=True, cell_factor=cell_factor
    )

    # optimize atoms in batch using ASE
    batch_optimizer = optimizer_cls(obatch)
    batch_optimizer.run(0.05, 50)

    # optimize atoms one-by-one
    for atoms in atoms_list:
        atoms.calc = calculator
        opt = optimizer_cls(UnitCellFilter(atoms, cell_factor=cell_factor))
        opt.run(0.05, 50)

    # compare energy, atom positions and cell
    for a1, a2 in zip(atoms_list, obatch.get_atoms_list()):
        assert a1.get_potential_energy() / len(a1) == pytest.approx(
            a2.get_potential_energy() / len(a2), abs=0.05
        )

        pnorm1 = np.linalg.norm(a1.positions, axis=1)
        pnorm2 = np.linalg.norm(a2.positions, axis=1)
        npt.assert_allclose(pnorm1, pnorm2, atol=0.05)

        cnorm1 = np.linalg.norm(a1.cell.array, axis=1)
        cnorm2 = np.linalg.norm(a2.cell.array, axis=1)
        npt.assert_allclose(cnorm1, cnorm2, atol=0.05)
        npt.assert_allclose(
            a1.cell.array.T / cnorm1, a2.cell.array.T / cnorm2, atol=0.01
        )
