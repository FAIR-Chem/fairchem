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


   .. py:property:: config
      :type: omegaconf.DictConfig



   .. py:method:: run() -> Any
      :abstractmethod:



   .. py:method:: save_state(checkpoint_location: str) -> None
      :abstractmethod:



   .. py:method:: load_state(checkpoint_location: str) -> None
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


   .. py:method:: save_state() -> None


   .. py:method:: load_state() -> None


