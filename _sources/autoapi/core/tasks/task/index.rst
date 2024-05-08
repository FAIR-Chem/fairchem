:py:mod:`core.tasks.task`
=========================

.. py:module:: core.tasks.task

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   core.tasks.task.BaseTask
   core.tasks.task.TrainTask
   core.tasks.task.PredictTask
   core.tasks.task.ValidateTask
   core.tasks.task.RelaxationTask




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



.. py:class:: RelaxationTask(config)


   Bases: :py:obj:`BaseTask`

   .. py:method:: run() -> None



