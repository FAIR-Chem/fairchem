core._cli
=========

.. py:module:: core._cli

.. autoapi-nested-parse::

   Copyright (c) Facebook, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core._cli.Runner


Functions
---------

.. autoapisummary::

   core._cli.runner_wrapper
   core._cli.main


Module Contents
---------------

.. py:class:: Runner

   Bases: :py:obj:`submitit.helpers.Checkpointable`


   Derived callable classes are requeued after timeout with their current
   state dumped at checkpoint.

   __call__ method must be implemented to make your class a callable.

   .. note::

      The following implementation of the checkpoint method resubmits the full current
      state of the callable (self) with the initial argument. You may want to replace the method to
      curate the state (dump a neural network to a standard format and remove it from
      the state so that not to pickle it) and change/remove the initial parameters.


   .. py:attribute:: config
      :value: None



   .. py:method:: __call__(config: dict) -> None


   .. py:method:: checkpoint(*args, **kwargs)

      Resubmits the same callable with the same arguments



.. py:function:: runner_wrapper(config: dict)

.. py:function:: main(args: argparse.Namespace | None = None, override_args: list[str] | None = None)

   Run the main fairchem program.


