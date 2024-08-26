from __future__ import annotations

from itertools import combinations, product

import numpy as np
import numpy.testing as npt
import pytest
from ase.io import read
from ase.optimize import LBFGS as LBFGS_ASE

from fairchem.core.common.relaxation import OptimizableBatch
from fairchem.core.common.relaxation.optimizers import LBFGS
from fairchem.core.modules.evaluator import min_diff


def test_lbfgs_relaxation(atoms_list, batch, calculator):
    """Tests batch relaxation using fairchem LBFGS optimizer."""
    obatch = OptimizableBatch(batch, trainer=calculator.trainer, numpy=False)

    # optimize atoms one-by-one
    for atoms in atoms_list:
        atoms.calc = calculator
        opt = LBFGS_ASE(atoms, damping=0.8, alpha=70.0)
        opt.run(0.01, 20)

    # optimize atoms in batch using ASE
    batch_optimizer = LBFGS(obatch, damping=0.8, alpha=70.0)
    batch_optimizer.run(0.01, 20)

    # compare energy and atom positions, this needs pretty slack tols but that should be ok
    for a1, a2 in zip(atoms_list, obatch.get_atoms_list()):
        assert a1.get_potential_energy() / len(a1) == pytest.approx(
            a2.get_potential_energy() / len(a2), abs=0.05
        )
        diff = min_diff(a1.positions, a2.positions, a1.get_cell(), pbc=a1.pbc)
        npt.assert_allclose(diff, 0, atol=0.01)


@pytest.mark.parametrize(
    ("save_full_traj", "steps"), list(product((True, False), (0, 1, 5)))
)
def test_lbfgs_write_trajectory(save_full_traj, steps, batch, calculator, tmp_path):
    obatch = OptimizableBatch(batch, trainer=calculator.trainer, numpy=False)
    batch_optimizer = LBFGS(
        obatch,
        save_full_traj=save_full_traj,
        traj_dir=tmp_path,
        traj_names=[f"system-{i}" for i in range(len(batch))],
    )

    batch_optimizer.run(0.01, steps=steps)

    # check that trajectory files where written
    traj_files = list(tmp_path.glob("*.traj"))
    assert len(traj_files) == len(batch)

    traj_length = (
        0 if steps == 0 else steps + 1 if save_full_traj else 2
    )  # first and final frame
    for file in traj_files:
        traj = read(file, ":")
        assert len(traj) == traj_length

        # make sure all written frames are unique
        for a1, a2 in combinations(traj, r=2):
            assert not np.allclose(a1.positions, a2.positions, atol=1e-3)
