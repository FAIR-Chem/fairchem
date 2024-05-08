:py:mod:`fairchem.core.modules.evaluator`
=========================================

.. py:module:: fairchem.core.modules.evaluator

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   fairchem.core.modules.evaluator.Evaluator



Functions
~~~~~~~~~

.. autoapisummary::

   fairchem.core.modules.evaluator.forcesx_mae
   fairchem.core.modules.evaluator.forcesx_mse
   fairchem.core.modules.evaluator.forcesy_mae
   fairchem.core.modules.evaluator.forcesy_mse
   fairchem.core.modules.evaluator.forcesz_mae
   fairchem.core.modules.evaluator.forcesz_mse
   fairchem.core.modules.evaluator.energy_forces_within_threshold
   fairchem.core.modules.evaluator.energy_within_threshold
   fairchem.core.modules.evaluator.average_distance_within_threshold
   fairchem.core.modules.evaluator.min_diff
   fairchem.core.modules.evaluator.cosine_similarity
   fairchem.core.modules.evaluator.mae
   fairchem.core.modules.evaluator.mse
   fairchem.core.modules.evaluator.magnitude_error



Attributes
~~~~~~~~~~

.. autoapisummary::

   fairchem.core.modules.evaluator.NONE


.. py:data:: NONE

   

.. py:class:: Evaluator(task: str | None = None, eval_metrics: dict | None = None)


   .. py:attribute:: task_metrics
      :type: ClassVar[dict[str, str]]

      

   .. py:attribute:: task_primary_metric
      :type: ClassVar[dict[str, str | None]]

      

   .. py:method:: eval(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], prev_metrics=None)


   .. py:method:: update(key, stat, metrics)



.. py:function:: forcesx_mae(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = NONE)


.. py:function:: forcesx_mse(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = NONE)


.. py:function:: forcesy_mae(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = None)


.. py:function:: forcesy_mse(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = None)


.. py:function:: forcesz_mae(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = None)


.. py:function:: forcesz_mse(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = None)


.. py:function:: energy_forces_within_threshold(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = None) -> dict[str, float | int]


.. py:function:: energy_within_threshold(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = None) -> dict[str, float | int]


.. py:function:: average_distance_within_threshold(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = None) -> dict[str, float | int]


.. py:function:: min_diff(pred_pos: torch.Tensor, dft_pos: torch.Tensor, cell: torch.Tensor, pbc: torch.Tensor)


.. py:function:: cosine_similarity(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = NONE)


.. py:function:: mae(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = NONE) -> dict[str, float | int]


.. py:function:: mse(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = NONE) -> dict[str, float | int]


.. py:function:: magnitude_error(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = NONE, p: int = 2) -> dict[str, float | int]


