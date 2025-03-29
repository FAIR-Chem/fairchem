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

   Represents an abstraction over things that reduce the results written by a set of runner.

   .. note::

      When running with the `fairchemv2` cli, the `job_config` and `runner_config` attributes are set at
      runtime to those given in the config file.

   .. attribute:: job_config

      a managed attribute that gives access to the job config

      :type: DictConfig

   .. attribute:: runner_config

      a managed attributed that gives access to the calling runner config

      :type: DictConfig


   .. py:attribute:: job_config


   .. py:attribute:: runner_config


   .. py:property:: runner_type
      :type: type[fairchem.core.components.runner.Runner]

      :abstractmethod:


      The runner type this reducer is associated with.


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


   .. py:property:: runner_type

      The runner type this reducer is associated with.


   .. py:method:: reduce() -> Any

      Use file pattern to reduce



   .. py:method:: save_state(checkpoint_location: str, is_preemption: bool = False) -> bool


   .. py:method:: load_state(checkpoint_location: str | None) -> None


