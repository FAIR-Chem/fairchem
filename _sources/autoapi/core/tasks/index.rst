:py:mod:`core.tasks`
====================

.. py:module:: core.tasks


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   task/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   core.tasks.PredictTask
   core.tasks.RelaxationTask
   core.tasks.TrainTask
   core.tasks.ValidateTask




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



