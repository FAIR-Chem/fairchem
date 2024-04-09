:py:mod:`ocpmodels.models.gemnet_oc.initializers`
=================================================

.. py:module:: ocpmodels.models.gemnet_oc.initializers

.. autoapi-nested-parse::

   Copyright (c) Facebook, Inc. and its affiliates.
   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   ocpmodels.models.gemnet_oc.initializers._standardize
   ocpmodels.models.gemnet_oc.initializers.he_orthogonal_init
   ocpmodels.models.gemnet_oc.initializers.grid_init
   ocpmodels.models.gemnet_oc.initializers.log_grid_init
   ocpmodels.models.gemnet_oc.initializers.get_initializer



.. py:function:: _standardize(kernel)

   Makes sure that N*Var(W) = 1 and E[W] = 0


.. py:function:: he_orthogonal_init(tensor: torch.Tensor) -> torch.Tensor

   Generate a weight matrix with variance according to He (Kaiming) initialization.
   Based on a random (semi-)orthogonal matrix neural networks
   are expected to learn better when features are decorrelated
   (stated by eg. "Reducing overfitting in deep networks by decorrelating representations",
   "Dropout: a simple way to prevent neural networks from overfitting",
   "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks")


.. py:function:: grid_init(tensor: torch.Tensor, start: int = -1, end: int = 1) -> torch.Tensor

   Generate a weight matrix so that each input value corresponds to one value on a regular grid between start and end.


.. py:function:: log_grid_init(tensor: torch.Tensor, start: int = -4, end: int = 0) -> torch.Tensor

   Generate a weight matrix so that each input value corresponds to one value on a regular logarithmic grid between 10^start and 10^end.


.. py:function:: get_initializer(name, **init_kwargs)


