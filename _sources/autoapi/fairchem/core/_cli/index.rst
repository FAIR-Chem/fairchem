:py:mod:`fairchem.core._cli`
============================

.. py:module:: fairchem.core._cli

.. autoapi-nested-parse::

   Copyright (c) Facebook, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   fairchem.core._cli.Runner



Functions
~~~~~~~~~

.. autoapisummary::

   fairchem.core._cli.main



.. py:class:: Runner(distributed: bool = False)


   Bases: :py:obj:`submitit.helpers.Checkpointable`

   Derived callable classes are requeued after timeout with their current
   state dumped at checkpoint.

   __call__ method must be implemented to make your class a callable.

   .. note::

      The following implementation of the checkpoint method resubmits the full current
      state of the callable (self) with the initial argument. You may want to replace the method to
      curate the state (dump a neural network to a standard format and remove it from
      the state so that not to pickle it) and change/remove the initial parameters.

   .. py:method:: __call__(config: dict) -> None


   .. py:method:: checkpoint(*args, **kwargs)

      Resubmits the same callable with the same arguments



.. py:function:: main()

   Run the main ocp-models program.


