:py:mod:`ocpmodels.common.relaxation.ml_relaxation`
===================================================

.. py:module:: ocpmodels.common.relaxation.ml_relaxation

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   ocpmodels.common.relaxation.ml_relaxation.ml_relax



.. py:function:: ml_relax(batch, model, steps: int, fmax: float, relax_opt, save_full_traj, device: str = 'cuda:0', transform=None, early_stop_batch: bool = False)

   Runs ML-based relaxations.
   :param batch: object
   :param model: object
   :param steps: int
                 Max number of steps in the structure relaxation.
   :param fmax: float
                Structure relaxation terminates when the max force
                of the system is no bigger than fmax.
   :param relax_opt: str
                     Optimizer and corresponding parameters to be used for structure relaxations.
   :param save_full_traj: bool
                          Whether to save out the full ASE trajectory. If False, only save out initial and final frames.


