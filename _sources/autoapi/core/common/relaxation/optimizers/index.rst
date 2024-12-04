core.common.relaxation.optimizers
=================================

.. py:module:: core.common.relaxation.optimizers

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/core/common/relaxation/optimizers/lbfgs_torch/index


Classes
-------

.. autoapisummary::

   core.common.relaxation.optimizers.LBFGS


Package Contents
----------------

.. py:class:: LBFGS(optimizable_batch: core.common.relaxation.optimizers.optimizable.OptimizableBatch, maxstep: float = 0.02, memory: int = 100, damping: float = 1.2, alpha: float = 100.0, save_full_traj: bool = True, traj_dir: pathlib.Path | None = None, traj_names: list[str] | None = None)

   Limited memory BFGS optimizer for batch ML relaxations.


   .. py:attribute:: optimizable


   .. py:attribute:: maxstep


   .. py:attribute:: memory


   .. py:attribute:: damping


   .. py:attribute:: alpha


   .. py:attribute:: H0


   .. py:attribute:: save_full


   .. py:attribute:: traj_dir


   .. py:attribute:: traj_names


   .. py:attribute:: trajectories
      :value: None



   .. py:attribute:: fmax
      :value: None



   .. py:attribute:: steps
      :value: None



   .. py:attribute:: s


   .. py:attribute:: y


   .. py:attribute:: rho


   .. py:attribute:: r0
      :value: None



   .. py:attribute:: f0
      :value: None



   .. py:method:: run(fmax, steps)


   .. py:method:: determine_step(dr)


   .. py:method:: _batched_dot(x: torch.Tensor, y: torch.Tensor)


   .. py:method:: step(iteration: int) -> None


   .. py:method:: write() -> None


