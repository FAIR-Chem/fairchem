core.tasks
==========

.. py:module:: core.tasks


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/core/tasks/task/index


Classes
-------

.. autoapisummary::

   core.tasks.PredictTask
   core.tasks.RelaxationTask
   core.tasks.TrainTask
   core.tasks.ValidateTask


Package Contents
----------------

.. py:class:: PredictTask(config)

   Bases: :py:obj:`BaseTask`


   .. py:method:: run() -> None


.. py:class:: RelaxationTask(config)

   Bases: :py:obj:`BaseTask`


   .. py:method:: run() -> None


.. py:class:: TrainTask(config)

   Bases: :py:obj:`BaseTask`


   .. py:method:: _process_error(e: RuntimeError) -> None


   .. py:method:: run() -> None


.. py:class:: ValidateTask(config)

   Bases: :py:obj:`BaseTask`


   .. py:method:: run() -> None


