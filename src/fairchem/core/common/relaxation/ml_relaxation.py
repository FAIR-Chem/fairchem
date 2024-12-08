"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch_geometric.data import Batch

from fairchem.core.common.typing import assert_is_instance
from fairchem.core.datasets.lmdb_dataset import data_list_collater

from .optimizable import OptimizableBatch, OptimizableUnitCellBatch
from .optimizers.lbfgs_torch import LBFGS

if TYPE_CHECKING:
    from fairchem.core.trainers import BaseTrainer


def ml_relax(
    batch: Batch,
    model: BaseTrainer,
    steps: int,
    fmax: float,
    relax_opt: dict[str] | None = None,
    relax_cell: bool = False,
    relax_volume: bool = False,
    save_full_traj: bool = True,
    transform: torch.nn.Module | None = None,
    mask_converged: bool = True,
):
    """Runs ML-based relaxations.

    Args:
        batch: a data batch object.
        model: a trainer object with model.
        steps: Max number of steps in the structure relaxation.
        fmax: Structure relaxation terminates when the max force of the system is no bigger than fmax.
        relax_opt: Optimizer parameters to be used for structure relaxations.
        relax_cell: if true will use stress predictions to relax crystallographic cell.
            The model given must predict stress
        relax_volume: if true will relax the cell isotropically. the given model must predict stress.
        save_full_traj: Whether to save out the full ASE trajectory. If False, only save out initial and final frames.
        mask_converged: whether to mask batches where all atoms are below convergence threshold
        cumulative_mask: if true, once system is masked then it remains masked even if new predictions give forces
                above threshold, ie. once masked always masked. Note if this is used make sure to check convergence with
                the same fmax always
    """
    relax_opt = relax_opt or {}
    # if not pbc is set, ignore it when comparing batches
    if not hasattr(batch, "pbc"):
        OptimizableBatch.ignored_changes = {"pbc"}

    batches = deque([batch])
    relaxed_batches = []
    while batches:
        batch = batches.popleft()
        oom = False
        ids = batch.sid

        # clone the batch otherwise you can not run batch.to_data_list
        # see https://github.com/pyg-team/pytorch_geometric/issues/8439#issuecomment-1826747915
        if relax_cell or relax_volume:
            optimizable = OptimizableUnitCellBatch(
                batch.clone(),
                trainer=model,
                transform=transform,
                mask_converged=mask_converged,
                hydrostatic_strain=relax_volume,
            )
        else:
            optimizable = OptimizableBatch(
                batch.clone(),
                trainer=model,
                transform=transform,
                mask_converged=mask_converged,
            )

        # Run ML-based relaxation
        traj_dir = relax_opt.get("traj_dir")
        relax_opt.update({"traj_dir": Path(traj_dir) if traj_dir is not None else None})

        optimizer = LBFGS(
            optimizable_batch=optimizable,
            save_full_traj=save_full_traj,
            traj_names=ids,
            **relax_opt,
        )

        e: RuntimeError | None = None
        try:
            optimizer.run(fmax=fmax, steps=steps)
            relaxed_batches.append(optimizable.batch)
        except RuntimeError as err:
            e = err
            oom = True
            torch.cuda.empty_cache()

        if oom:
            # move OOM recovery code outside off except clause to allow tensors to be freed.
            data_list = batch.to_data_list()
            if len(data_list) == 1:
                raise assert_is_instance(e, RuntimeError)
            logging.info(
                f"Failed to relax batch with size: {len(data_list)}, splitting into two..."
            )
            mid = len(data_list) // 2
            batches.appendleft(
                data_list_collater(data_list[:mid], otf_graph=optimizable.otf_graph)
            )
            batches.appendleft(
                data_list_collater(data_list[mid:], otf_graph=optimizable.otf_graph)
            )

    # reset for good measure
    OptimizableBatch.ignored_changes = {}

    relaxed_batch = Batch.from_data_list(relaxed_batches)

    # Batch.from_data_list is not intended to be used with a list of batches, so when sid is a list of str
    # it will be incorrectly collated as a list of lists for each batch.
    # but we can not use to_data_list in the relaxed batches (since they have been changed, see linked comment above).
    # So instead just manually fix it for now. Remove this once pyg dependency is removed
    if isinstance(relaxed_batch.sid, list):
        relaxed_batch.sid = [sid for sid_list in relaxed_batch.sid for sid in sid_list]

    return relaxed_batch
