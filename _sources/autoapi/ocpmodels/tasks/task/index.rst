:py:mod:`ocpmodels.tasks.task`
==============================

.. py:module:: ocpmodels.tasks.task

.. autoapi-nested-parse::

   Copyright (c) Facebook, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   ocpmodels.tasks.task.BaseTask
   ocpmodels.tasks.task.TrainTask
   ocpmodels.tasks.task.PredictTask
   ocpmodels.tasks.task.ValidateTask
   ocpmodels.tasks.task.RelxationTask




.. py:class:: BaseTask(config)


   .. py:method:: setup(trainer) -> None


   .. py:method:: run()
      :abstractmethod:



.. py:class:: TrainTask(config)


   Bases: :py:obj:`BaseTask`

   .. py:method:: _process_error(e: RuntimeError) -> None


   .. py:method:: run() -> None



.. py:class:: PredictTask(config)


   Bases: :py:obj:`BaseTask`

   .. py:method:: run() -> None



.. py:class:: ValidateTask(config)


   Bases: :py:obj:`BaseTask`

   .. py:method:: run() -> None



.. py:class:: RelxationTask(config)


   Bases: :py:obj:`BaseTask`

   .. py:method:: run() -> None



