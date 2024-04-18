:py:mod:`ocpmodels.modules.evaluator`
=====================================

.. py:module:: ocpmodels.modules.evaluator

.. autoapi-nested-parse::

   Copyright (c) Facebook, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   ocpmodels.modules.evaluator.Evaluator



Functions
~~~~~~~~~

.. autoapisummary::

   ocpmodels.modules.evaluator.forcesx_mae
   ocpmodels.modules.evaluator.forcesx_mse
   ocpmodels.modules.evaluator.forcesy_mae
   ocpmodels.modules.evaluator.forcesy_mse
   ocpmodels.modules.evaluator.forcesz_mae
   ocpmodels.modules.evaluator.forcesz_mse
   ocpmodels.modules.evaluator.energy_forces_within_threshold
   ocpmodels.modules.evaluator.energy_within_threshold
   ocpmodels.modules.evaluator.average_distance_within_threshold
   ocpmodels.modules.evaluator.min_diff
   ocpmodels.modules.evaluator.cosine_similarity
   ocpmodels.modules.evaluator.mae
   ocpmodels.modules.evaluator.mse
   ocpmodels.modules.evaluator.magnitude_error



.. py:class:: Evaluator(task: str = None, eval_metrics: dict = {})


   .. py:attribute:: task_metrics

      

   .. py:attribute:: task_primary_metric

      

   .. py:method:: eval(prediction: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], prev_metrics={})


   .. py:method:: update(key, stat, metrics)



.. py:function:: forcesx_mae(prediction: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], key=None)


.. py:function:: forcesx_mse(prediction: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], key=None)


.. py:function:: forcesy_mae(prediction: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], key=None)


.. py:function:: forcesy_mse(prediction: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], key=None)


.. py:function:: forcesz_mae(prediction: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], key=None)


.. py:function:: forcesz_mse(prediction: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], key=None)


.. py:function:: energy_forces_within_threshold(prediction: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], key=None) -> Dict[str, Union[float, int]]


.. py:function:: energy_within_threshold(prediction: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], key=None) -> Dict[str, Union[float, int]]


.. py:function:: average_distance_within_threshold(prediction: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], key=None) -> Dict[str, Union[float, int]]


.. py:function:: min_diff(pred_pos: torch.Tensor, dft_pos: torch.Tensor, cell: torch.Tensor, pbc: torch.Tensor)


.. py:function:: cosine_similarity(prediction: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], key=slice(None))


.. py:function:: mae(prediction: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], key=slice(None)) -> Dict[str, Union[float, int]]


.. py:function:: mse(prediction: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], key=slice(None)) -> Dict[str, Union[float, int]]


.. py:function:: magnitude_error(prediction: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], key=slice(None), p: int = 2) -> Dict[str, Union[float, int]]


