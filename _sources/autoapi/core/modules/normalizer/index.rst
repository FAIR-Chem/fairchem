core.modules.normalizer
=======================

.. py:module:: core.modules.normalizer

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.modules.normalizer.Normalizer


Module Contents
---------------

.. py:class:: Normalizer(tensor: torch.Tensor | None = None, mean=None, std=None, device=None)

   Normalize a Tensor and restore it later.


   .. py:method:: to(device) -> None


   .. py:method:: norm(tensor: torch.Tensor) -> torch.Tensor


   .. py:method:: denorm(normed_tensor: torch.Tensor) -> torch.Tensor


   .. py:method:: state_dict()


   .. py:method:: load_state_dict(state_dict) -> None


