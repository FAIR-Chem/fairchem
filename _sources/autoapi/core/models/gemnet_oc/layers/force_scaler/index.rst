core.models.gemnet_oc.layers.force_scaler
=========================================

.. py:module:: core.models.gemnet_oc.layers.force_scaler

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.
   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.models.gemnet_oc.layers.force_scaler.ForceScaler


Module Contents
---------------

.. py:class:: ForceScaler(init_scale: float = 2.0**8, growth_factor: float = 2.0, backoff_factor: float = 0.5, growth_interval: int = 2000, max_force_iters: int = 50, enabled: bool = True)

   Scales up the energy and then scales down the forces
   to prevent NaNs and infs in calculations using AMP.
   Inspired by torch.cuda.amp.GradScaler.


   .. py:attribute:: scale_factor


   .. py:attribute:: growth_factor


   .. py:attribute:: backoff_factor


   .. py:attribute:: growth_interval


   .. py:attribute:: max_force_iters


   .. py:attribute:: enabled


   .. py:attribute:: finite_force_results
      :value: 0



   .. py:method:: scale(energy)


   .. py:method:: unscale(forces)


   .. py:method:: calc_forces(energy, pos)


   .. py:method:: calc_forces_and_update(energy, pos)


   .. py:method:: update() -> None


