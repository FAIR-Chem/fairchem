import pytest

import numpy as np
import numpy.testing as npt

from ase import build
from ase.optimize import LBFGS, FIRE2
from fairchem.core.datasets import data_list_collater
from fairchem.core.preprocessing.atoms_to_graphs import AtomsToGraphs
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from fairchem.core.common.relaxation import OptimizableBatch


@pytest.fixture
def calculator(tmpdir):
    # TODO load trainer directly - add a from_checkpoint class method to Trainer class
    # ckpt_path = Path(__file__).parent.resolve() / "s2efs_with_linref.pt"
    # calc = OCPCalculator(checkpoint_path=str(ckpt_path), trainer="eqV2_trainer_", seed=0, cpu=False)
    calc = OCPCalculator(
        model_name="EquiformerV2-31M-S2EF-OC20-All+MD", local_cache=tmpdir
    )
    # removing amp gives non-flakey results
    # calc.scaler = None
    return calc


@pytest.fixture(scope="function")
def atoms_list():
    atoms_list = [
        build.make_supercell(build.bulk("Cu", "fcc", a=4.0, cubic=True), 2 * np.eye(3)),
        build.bulk("NaCl", crystalstructure="rocksalt", a=3.5),
        # TODO check why relaxing a surface does not match
        # build.fcc111("Pt", size=[2, 2, 3], vacuum=8, periodic=True)
    ]
    return atoms_list


@pytest.fixture(scope="function")
def batch(atoms_list):
    a2g = AtomsToGraphs(r_edges=False, r_pbc=True)
    return data_list_collater([a2g.convert(atoms) for atoms in atoms_list])


@pytest.fixture(params=[LBFGS, FIRE2])
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

    for a1, a2 in zip(atoms_list, obatch.get_atoms_list()):
        assert np.round(a1.get_potential_energy(), 1) == np.round(
            a2.get_potential_energy(), 1
        )
        npt.assert_allclose(a1.positions, a2.positions)
