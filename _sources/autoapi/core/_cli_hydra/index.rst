core._cli_hydra
===============

.. py:module:: core._cli_hydra

.. autoapi-nested-parse::

   Copyright (c) Facebook, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Attributes
----------

.. autoapisummary::

   core._cli_hydra.logger


Classes
-------

.. autoapisummary::

   core._cli_hydra.Submitit


Functions
---------

.. autoapisummary::

   core._cli_hydra.map_cli_args_to_dist_config
   core._cli_hydra.get_hydra_config_from_yaml
   core._cli_hydra.runner_wrapper
   core._cli_hydra.main


Module Contents
---------------

.. py:data:: logger

.. py:class:: Submitit

   Bases: :py:obj:`submitit.helpers.Checkpointable`


   Derived callable classes are requeued after timeout with their current
   state dumped at checkpoint.

   __call__ method must be implemented to make your class a callable.

   .. note::

      The following implementation of the checkpoint method resubmits the full current
      state of the callable (self) with the initial argument. You may want to replace the method to
      curate the state (dump a neural network to a standard format and remove it from
      the state so that not to pickle it) and change/remove the initial parameters.


   .. py:method:: __call__(dict_config: omegaconf.DictConfig, cli_args: argparse.Namespace) -> None


   .. py:method:: checkpoint(*args, **kwargs)

      Resubmits the same callable with the same arguments



.. py:function:: map_cli_args_to_dist_config(cli_args: argparse.Namespace) -> dict

.. py:function:: get_hydra_config_from_yaml(config_yml: str, overrides_args: list[str]) -> omegaconf.DictConfig

.. py:function:: runner_wrapper(config: omegaconf.DictConfig, cli_args: argparse.Namespace)

.. py:function:: main(args: argparse.Namespace | None = None, override_args: list[str] | None = None)

