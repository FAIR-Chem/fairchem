core.components.runner
======================

.. py:module:: core.components.runner

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



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


   Used for testing


   .. py:attribute:: x


   .. py:attribute:: y


   .. py:attribute:: z


   .. py:method:: run() -> Any


   .. py:method:: initialize(job_config: omegaconf.DictConfig) -> None


   .. py:method:: save_state(checkpoint_location: str, is_preemption: bool = False) -> bool


   .. py:method:: load_state(checkpoint_location: str | None) -> None


