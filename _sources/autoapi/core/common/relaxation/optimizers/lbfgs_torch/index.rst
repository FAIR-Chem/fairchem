core.common.relaxation.optimizers.lbfgs_torch
=============================================

.. py:module:: core.common.relaxation.optimizers.lbfgs_torch

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.common.relaxation.optimizers.lbfgs_torch.LBFGS
   core.common.relaxation.optimizers.lbfgs_torch.TorchCalc


Module Contents
---------------

.. py:class:: LBFGS(batch: torch_geometric.data.Batch, model: TorchCalc, maxstep: float = 0.01, memory: int = 100, damping: float = 0.25, alpha: float = 100.0, force_consistent=None, device: str = 'cuda:0', save_full_traj: bool = True, traj_dir: pathlib.Path | None = None, traj_names=None, early_stop_batch: bool = False)

   .. py:method:: get_energy_and_forces(apply_constraint: bool = True)


   .. py:method:: set_positions(update, update_mask) -> None


   .. py:method:: check_convergence(iteration, forces=None, energy=None)


   .. py:method:: run(fmax, steps)


   .. py:method:: step(iteration: int, forces: torch.Tensor | None, update_mask: torch.Tensor) -> None


   .. py:method:: write(energy, forces, update_mask) -> None


.. py:class:: TorchCalc(model, transform=None)

   .. py:method:: get_energy_and_forces(atoms, apply_constraint: bool = True)


   .. py:method:: update_graph(atoms)


