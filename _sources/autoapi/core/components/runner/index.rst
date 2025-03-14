core.components.runner
======================

.. py:module:: core.components.runner


Classes
-------

.. autoapisummary::

   core.components.runner.Runner
   core.components.runner.MockRunner


Module Contents
---------------

.. py:class:: Runner

   Represents an abstraction over things that run in a loop and can save/load state.
   ie: Trainers, Validators, Relaxation all fall in this category.
   This allows us to decouple away from a monolithic trainer class


   .. py:method:: initialize(job_config: omegaconf.DictConfig) -> None
      :abstractmethod:



   .. py:method:: run() -> Any
      :abstractmethod:



   .. py:method:: save_state(checkpoint_location: str, is_preemption: bool = False) -> bool
      :abstractmethod:



   .. py:method:: load_state(checkpoint_location: str | None) -> None
      :abstractmethod:



.. py:class:: MockRunner(x: int, y: int, z: int)

   Bases: :py:obj:`Runner`


   Represents an abstraction over things that run in a loop and can save/load state.
   ie: Trainers, Validators, Relaxation all fall in this category.
   This allows us to decouple away from a monolithic trainer class


   .. py:attribute:: x


   .. py:attribute:: y


   .. py:attribute:: z


   .. py:method:: run() -> Any


   .. py:method:: initialize(job_config: omegaconf.DictConfig) -> None


   .. py:method:: save_state(checkpoint_location: str, is_preemption: bool = False) -> bool


   .. py:method:: load_state(checkpoint_location: str | None) -> None


