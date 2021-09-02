"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from pathlib import Path

import torch

from ocpmodels.common.registry import registry

from .optimizers.lbfgs_torch import LBFGS, TorchCalc


def ml_relax(
    batch,
    model,
    steps,
    fmax,
    relax_opt,
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
    calc = TorchCalc(model, transform)

    # Run ML-based relaxation
    traj_dir = relax_opt.get("traj_dir", None)
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
