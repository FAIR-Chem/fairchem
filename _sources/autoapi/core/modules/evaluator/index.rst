core.modules.evaluator
======================

.. py:module:: core.modules.evaluator

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Attributes
----------

.. autoapisummary::

   core.modules.evaluator.NONE_SLICE


Classes
-------

.. autoapisummary::

   core.modules.evaluator.Evaluator


Functions
---------

.. autoapisummary::

   core.modules.evaluator.metrics_dict
   core.modules.evaluator.cosine_similarity
   core.modules.evaluator.mae
   core.modules.evaluator.mse
   core.modules.evaluator.per_atom_mae
   core.modules.evaluator.per_atom_mse
   core.modules.evaluator.magnitude_error
   core.modules.evaluator.forcesx_mae
   core.modules.evaluator.forcesx_mse
   core.modules.evaluator.forcesy_mae
   core.modules.evaluator.forcesy_mse
   core.modules.evaluator.forcesz_mae
   core.modules.evaluator.forcesz_mse
   core.modules.evaluator.energy_forces_within_threshold
   core.modules.evaluator.energy_within_threshold
   core.modules.evaluator.average_distance_within_threshold
   core.modules.evaluator.min_diff
   core.modules.evaluator.rmse


Module Contents
---------------

.. py:data:: NONE_SLICE

.. py:class:: Evaluator(task: str | None = None, eval_metrics: dict | None = None)

   .. py:attribute:: task_metrics
      :type:  ClassVar[dict[str, str]]


   .. py:attribute:: task_primary_metric
      :type:  ClassVar[dict[str, str | None]]


   .. py:attribute:: task


   .. py:attribute:: target_metrics


   .. py:method:: eval(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], prev_metrics: dict | None = None)


   .. py:method:: update(key, stat, metrics)


.. py:function:: metrics_dict(metric_fun: Callable) -> Callable

   Wrap up the return of a metrics function


.. py:function:: cosine_similarity(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = NONE_SLICE)

.. py:function:: mae(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = NONE_SLICE) -> torch.Tensor

.. py:function:: mse(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = NONE_SLICE) -> torch.Tensor

.. py:function:: per_atom_mae(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = NONE_SLICE) -> torch.Tensor

.. py:function:: per_atom_mse(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = NONE_SLICE) -> torch.Tensor

.. py:function:: magnitude_error(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = NONE_SLICE, p: int = 2) -> torch.Tensor

.. py:function:: forcesx_mae(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = NONE_SLICE)

.. py:function:: forcesx_mse(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = NONE_SLICE)

.. py:function:: forcesy_mae(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = None)

.. py:function:: forcesy_mse(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = None)

.. py:function:: forcesz_mae(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = None)

.. py:function:: forcesz_mse(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = None)

.. py:function:: energy_forces_within_threshold(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = None) -> dict[str, float | int]

.. py:function:: energy_within_threshold(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = None) -> dict[str, float | int]

.. py:function:: average_distance_within_threshold(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = None) -> dict[str, float | int]

.. py:function:: min_diff(pred_pos: torch.Tensor, dft_pos: torch.Tensor, cell: torch.Tensor, pbc: torch.Tensor)

.. py:function:: rmse(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = None) -> dict[str, float | int]

