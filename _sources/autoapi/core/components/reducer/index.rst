core.components.reducer
=======================

.. py:module:: core.components.reducer

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.components.reducer.Reducer
   core.components.reducer.MockReducer


Module Contents
---------------

.. py:class:: Reducer

   Represents an abstraction over things reduce the results written by a runner.


   .. py:method:: initialize(job_config: omegaconf.DictConfig, runner_config: omegaconf.DictConfig) -> None
      :abstractmethod:


      Initialize takes both the job config and a runner config assumed to have been run beforehand



   .. py:method:: reduce() -> Any
      :abstractmethod:


      Use file pattern to reduce



   .. py:method:: save_state(checkpoint_location: str, is_preemption: bool = False) -> bool
      :abstractmethod:



   .. py:method:: load_state(checkpoint_location: str | None) -> None
      :abstractmethod:



.. py:class:: MockReducer

   Bases: :py:obj:`Reducer`


   Used for testing


   .. py:attribute:: calling_runner_config
      :value: None



   .. py:method:: initialize(job_config: omegaconf.DictConfig, runner_config: omegaconf.DictConfig) -> None

      Initialize takes both the job config and a runner config assumed to have been run beforehand



   .. py:method:: reduce() -> Any

      Use file pattern to reduce



   .. py:method:: save_state(checkpoint_location: str, is_preemption: bool = False) -> bool


   .. py:method:: load_state(checkpoint_location: str | None) -> None


