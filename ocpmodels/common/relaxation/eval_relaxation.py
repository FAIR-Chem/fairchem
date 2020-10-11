from pathlib import Path

import torch

from ocpmodels.common.meter import mae, mae_ratio, mean_l2_distance
from ocpmodels.common.registry import registry

from .optimizers.lbfgs_torch import LBFGS, TorchCalc


def relax_eval(
    batch,
    model,
    steps,
    fmax,
    relax_opt,
    return_relaxed_pos,
    device="cuda:0",
    transform=None,
):
    """
    Evaluation of ML-based relaxations.
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
        return_relaxed_pos: bool
            Whether to return relaxed positions to be written for dft
            evaluation.
    """
    batch = batch[0]
    ids = batch.id
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
        traj_dir=Path(traj_dir),
        traj_names=ids,
    )
    relaxed_batch = optimizer.run(fmax=fmax, steps=steps)

    return relaxed_batch
