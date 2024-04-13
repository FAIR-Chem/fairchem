:py:mod:`ocpmodels.tasks`
=========================

.. py:module:: ocpmodels.tasks


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

   ocpmodels.tasks.PredictTask
   ocpmodels.tasks.RelaxationTask
   ocpmodels.tasks.TrainTask
   ocpmodels.tasks.ValidateTask




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



