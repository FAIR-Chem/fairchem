:py:mod:`ocpmodels.models.gemnet.layers.base_layers`
====================================================

.. py:module:: ocpmodels.models.gemnet.layers.base_layers

.. autoapi-nested-parse::

   Copyright (c) Facebook, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   ocpmodels.models.gemnet.layers.base_layers.Dense
   ocpmodels.models.gemnet.layers.base_layers.ScaledSiLU
   ocpmodels.models.gemnet.layers.base_layers.SiQU
   ocpmodels.models.gemnet.layers.base_layers.ResidualLayer




.. py:class:: Dense(in_features, out_features, bias: bool = False, activation=None)


   Bases: :py:obj:`torch.nn.Module`

   Combines dense layer with scaling for swish activation.

   :param units: Output embedding size.
   :type units: int
   :param activation: Name of the activation function to use.
   :type activation: str
   :param bias: True if use bias.
   :type bias: bool

   .. py:method:: reset_parameters(initializer=he_orthogonal_init) -> None


   .. py:method:: forward(x)



.. py:class:: ScaledSiLU


   Bases: :py:obj:`torch.nn.Module`

   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing to nest them in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self):
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will have their
   parameters converted too when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool

   .. py:method:: forward(x)



.. py:class:: SiQU


   Bases: :py:obj:`torch.nn.Module`

   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing to nest them in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self):
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will have their
   parameters converted too when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool

   .. py:method:: forward(x)



.. py:class:: ResidualLayer(units: int, nLayers: int = 2, layer=Dense, **layer_kwargs)


   Bases: :py:obj:`torch.nn.Module`

   Residual block with output scaled by 1/sqrt(2).

   :param units: Output embedding size.
   :type units: int
   :param nLayers: Number of dense layers.
   :type nLayers: int
   :param layer_kwargs: Keyword arguments for initializing the layers.
   :type layer_kwargs: str

   .. py:method:: forward(input)



