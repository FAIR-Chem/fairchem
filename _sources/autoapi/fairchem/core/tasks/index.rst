:py:mod:`fairchem.core.tasks`
=============================

.. py:module:: fairchem.core.tasks


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

   fairchem.core.tasks.PredictTask
   fairchem.core.tasks.RelaxationTask
   fairchem.core.tasks.TrainTask
   fairchem.core.tasks.ValidateTask




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



