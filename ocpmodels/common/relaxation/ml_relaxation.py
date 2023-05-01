"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
from collections import deque
from pathlib import Path

import torch
from torch_geometric.data import Batch

from ocpmodels.common.registry import registry
from ocpmodels.datasets.lmdb_dataset import data_list_collater

from .optimizers.lbfgs_torch import LBFGS, TorchCalc
from .optimizers.lbfgs_stress import LBFGS_Stress, TorchCalcStress



def ml_relax(
    batch,
    model,
    steps,
    fmax,
    relax_opt,
    save_full_traj,
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
        save_full_traj: bool
            Whether to save out the full ASE trajectory. If False, only save out initial and final frames.
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
            save_full_traj=save_full_traj,
            traj_dir=Path(traj_dir) if traj_dir is not None else None,
            traj_names=ids,
            early_stop_batch=early_stop_batch,
        )
        
    batches = deque([batch[0]])
    relaxed_batches = []
    while batches:
        batch = batches.popleft()
        oom = False
        ids = batch.sid
        calc = TorchCalc(model, transform)

        # Run ML-based relaxation
        traj_dir = relax_opt.get("traj_dir", None)
        
        relaxed_batch = optimizer.run(fmax=fmax, steps=steps)
        try:
            relaxed_batch = optimizer.run(fmax=fmax, steps=steps)
            relaxed_batches.append(relaxed_batch)
        except RuntimeError as e:
            oom = True
            torch.cuda.empty_cache()

        if oom:
            # move OOM recovery code outside of except clause to allow tensors to be freed.
            data_list = batch.to_data_list()
            if len(data_list) == 1:
                raise e
            logging.info(
                f"Failed to relax batch with size: {len(data_list)}, splitting into two..."
            )
            mid = len(data_list) // 2
            batches.appendleft(data_list_collater(data_list[:mid]))
            batches.appendleft(data_list_collater(data_list[mid:]))

    relaxed_batch = Batch.from_data_list(relaxed_batches)
    return relaxed_batch
