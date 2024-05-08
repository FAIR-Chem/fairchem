:py:mod:`fairchem.core.models.gemnet_gp.initializers`
=====================================================

.. py:module:: fairchem.core.models.gemnet_gp.initializers

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   fairchem.core.models.gemnet_gp.initializers._standardize
   fairchem.core.models.gemnet_gp.initializers.he_orthogonal_init



.. py:function:: _standardize(kernel)

   Makes sure that N*Var(W) = 1 and E[W] = 0


.. py:function:: he_orthogonal_init(tensor: torch.Tensor) -> torch.Tensor

   Generate a weight matrix with variance according to He (Kaiming) initialization.
   Based on a random (semi-)orthogonal matrix neural networks
   are expected to learn better when features are decorrelated
   (stated by eg. "Reducing overfitting in deep networks by decorrelating representations",
   "Dropout: a simple way to prevent neural networks from overfitting",
   "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks")


