"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from pathlib import Path

import torch

from ocpmodels.common.registry import registry

from .optimizers.lbfgs_torch import LBFGS, TorchCalc
from .optimizers.lbfgs_stress_review import LBFGS_Stress, TorchCalcStress



def ml_relax(
    batch,
    model,
    steps,
    fmax,
    relax_opt,
    isotropic=False,
    stress=False,
    opt_forces=True,
    opt_stress=True,
    cell_factor=0.1,
    device="cuda:0",
    transform=None,
    early_stop_batch=False,
):
    """
    Runs ML-based relaxations.
    Args:
        batch: object
        model: object
        steps: int
            Max number of steps in the structure relaxation.
        fmax: float
            Structure relaxation terminates when the max force
            of the system is no bigger than fmax.
        relax_opt: str
            Optimizer and corresponding parameters to be used for structure relaxations.
    """
    batch = batch[0]
    ids = batch.sid
    if stress:
        calc = TorchCalcStress(model, isotropic=isotropic, opt_forces=opt_forces, opt_stress=opt_stress, cell_factor=cell_factor, transform=transform)
    else:
        calc = TorchCalc(model, transform)

    # Run ML-based relaxation
    traj_dir = relax_opt.get("traj_dir", None)
    if stress:
        optimizer = LBFGS_Stress(
            batch,
            calc,
            maxstep=relax_opt.get("maxstep", 0.04),
            memory=relax_opt["memory"],
            damping=relax_opt.get("damping", 1.0),
            alpha=relax_opt.get("alpha", 70.0),
            device=device,
            traj_dir=Path(traj_dir) if traj_dir is not None else None,
            traj_names=ids,
            early_stop_batch=early_stop_batch,
        )
    else:
        optimizer = LBFGS(
            batch,
            calc,
            maxstep=relax_opt.get("maxstep", 0.04),
            memory=relax_opt["memory"],
            damping=relax_opt.get("damping", 1.0),
            alpha=relax_opt.get("alpha", 70.0),
            device=device,
            traj_dir=Path(traj_dir) if traj_dir is not None else None,
            traj_names=ids,
            early_stop_batch=early_stop_batch,
        )
    relaxed_batch = optimizer.run(fmax=fmax, steps=steps)

    return relaxed_batch
