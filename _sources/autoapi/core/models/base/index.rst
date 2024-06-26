core.models.base
================

.. py:module:: core.models.base

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.models.base.BaseModel


Module Contents
---------------

.. py:class:: BaseModel(num_atoms=None, bond_feat_dim=None, num_targets=None)

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


   .. py:method:: forward(data)
      :abstractmethod:



   .. py:method:: generate_graph(data, cutoff=None, max_neighbors=None, use_pbc=None, otf_graph=None, enforce_max_neighbors_strictly=None)


   .. py:property:: num_params
      :type: int



   .. py:method:: no_weight_decay() -> list

      Returns a list of parameters with no weight decay.



