core.common.relaxation.ml_relaxation
====================================

.. py:module:: core.common.relaxation.ml_relaxation

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Functions
---------

.. autoapisummary::

   core.common.relaxation.ml_relaxation.ml_relax


Module Contents
---------------

.. py:function:: ml_relax(batch: torch_geometric.data.Batch, model: fairchem.core.trainers.BaseTrainer, steps: int, fmax: float, relax_opt: dict[str] | None = None, relax_cell: bool = False, relax_volume: bool = False, save_full_traj: bool = True, transform: torch.nn.Module | None = None, mask_converged: bool = True)

   Runs ML-based relaxations.

   :param batch: a data batch object.
   :param model: a trainer object with model.
   :param steps: Max number of steps in the structure relaxation.
   :param fmax: Structure relaxation terminates when the max force of the system is no bigger than fmax.
   :param relax_opt: Optimizer parameters to be used for structure relaxations.
   :param relax_cell: if true will use stress predictions to relax crystallographic cell.
                      The model given must predict stress
   :param relax_volume: if true will relax the cell isotropically. the given model must predict stress.
   :param save_full_traj: Whether to save out the full ASE trajectory. If False, only save out initial and final frames.
   :param mask_converged: whether to mask batches where all atoms are below convergence threshold
   :param cumulative_mask: if true, once system is masked then it remains masked even if new predictions give forces
                           above threshold, ie. once masked always masked. Note if this is used make sure to check convergence with
                           the same fmax always


