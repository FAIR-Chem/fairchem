:py:mod:`fairchem.core.models.dimenet_plus_plus`
================================================

.. py:module:: fairchem.core.models.dimenet_plus_plus

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.

   ---

   This code borrows heavily from the DimeNet implementation as part of
   pytorch-geometric: https://github.com/rusty1s/pytorch_geometric. License:

   ---

   Copyright (c) 2020 Matthias Fey <matthias.fey@tu-dortmund.de>

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   fairchem.core.models.dimenet_plus_plus.InteractionPPBlock
   fairchem.core.models.dimenet_plus_plus.OutputPPBlock
   fairchem.core.models.dimenet_plus_plus.DimeNetPlusPlus
   fairchem.core.models.dimenet_plus_plus.DimeNetPlusPlusWrap




Attributes
~~~~~~~~~~

.. autoapisummary::

   fairchem.core.models.dimenet_plus_plus.sym


.. py:data:: sym

   

.. py:class:: InteractionPPBlock(hidden_channels: int, int_emb_size: int, basis_emb_size: int, num_spherical: int, num_radial: int, num_before_skip: int, num_after_skip: int, act='silu')


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

   .. py:method:: reset_parameters() -> None


   .. py:method:: forward(x, rbf, sbf, idx_kj, idx_ji)



.. py:class:: OutputPPBlock(num_radial: int, hidden_channels: int, out_emb_channels: int, out_channels: int, num_layers: int, act: str = 'silu')


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

   .. py:method:: reset_parameters() -> None


   .. py:method:: forward(x, rbf, i, num_nodes: int | None = None)



.. py:class:: DimeNetPlusPlus(hidden_channels: int, out_channels: int, num_blocks: int, int_emb_size: int, basis_emb_size: int, out_emb_channels: int, num_spherical: int, num_radial: int, cutoff: float = 5.0, envelope_exponent: int = 5, num_before_skip: int = 1, num_after_skip: int = 2, num_output_layers: int = 3, act: str = 'silu')


   Bases: :py:obj:`torch.nn.Module`

   DimeNet++ implementation based on https://github.com/klicperajo/dimenet.

   :param hidden_channels: Hidden embedding size.
   :type hidden_channels: int
   :param out_channels: Size of each output sample.
   :type out_channels: int
   :param num_blocks: Number of building blocks.
   :type num_blocks: int
   :param int_emb_size: Embedding size used for interaction triplets
   :type int_emb_size: int
   :param basis_emb_size: Embedding size used in the basis transformation
   :type basis_emb_size: int
   :param out_emb_channels: Embedding size used for atoms in the output block
   :type out_emb_channels: int
   :param num_spherical: Number of spherical harmonics.
   :type num_spherical: int
   :param num_radial: Number of radial basis functions.
   :type num_radial: int
   :param cutoff: (float, optional): Cutoff distance for interatomic
                  interactions. (default: :obj:`5.0`)
   :param envelope_exponent: Shape of the smooth cutoff.
                             (default: :obj:`5`)
   :type envelope_exponent: int, optional
   :param num_before_skip: (int, optional): Number of residual layers in the
                           interaction blocks before the skip connection. (default: :obj:`1`)
   :param num_after_skip: (int, optional): Number of residual layers in the
                          interaction blocks after the skip connection. (default: :obj:`2`)
   :param num_output_layers: (int, optional): Number of linear layers for the
                             output blocks. (default: :obj:`3`)
   :param act: (function, optional): The activation funtion.
               (default: :obj:`silu`)

   .. py:attribute:: url
      :value: 'https://github.com/klicperajo/dimenet/raw/master/pretrained'

      

   .. py:method:: reset_parameters() -> None


   .. py:method:: triplets(edge_index, cell_offsets, num_nodes: int)


   .. py:method:: forward(z, pos, batch=None)
      :abstractmethod:



.. py:class:: DimeNetPlusPlusWrap(num_atoms: int, bond_feat_dim: int, num_targets: int, use_pbc: bool = True, regress_forces: bool = True, hidden_channels: int = 128, num_blocks: int = 4, int_emb_size: int = 64, basis_emb_size: int = 8, out_emb_channels: int = 256, num_spherical: int = 7, num_radial: int = 6, otf_graph: bool = False, cutoff: float = 10.0, envelope_exponent: int = 5, num_before_skip: int = 1, num_after_skip: int = 2, num_output_layers: int = 3)


   Bases: :py:obj:`DimeNetPlusPlus`, :py:obj:`fairchem.core.models.base.BaseModel`

   DimeNet++ implementation based on https://github.com/klicperajo/dimenet.

   :param hidden_channels: Hidden embedding size.
   :type hidden_channels: int
   :param out_channels: Size of each output sample.
   :type out_channels: int
   :param num_blocks: Number of building blocks.
   :type num_blocks: int
   :param int_emb_size: Embedding size used for interaction triplets
   :type int_emb_size: int
   :param basis_emb_size: Embedding size used in the basis transformation
   :type basis_emb_size: int
   :param out_emb_channels: Embedding size used for atoms in the output block
   :type out_emb_channels: int
   :param num_spherical: Number of spherical harmonics.
   :type num_spherical: int
   :param num_radial: Number of radial basis functions.
   :type num_radial: int
   :param cutoff: (float, optional): Cutoff distance for interatomic
                  interactions. (default: :obj:`5.0`)
   :param envelope_exponent: Shape of the smooth cutoff.
                             (default: :obj:`5`)
   :type envelope_exponent: int, optional
   :param num_before_skip: (int, optional): Number of residual layers in the
                           interaction blocks before the skip connection. (default: :obj:`1`)
   :param num_after_skip: (int, optional): Number of residual layers in the
                          interaction blocks after the skip connection. (default: :obj:`2`)
   :param num_output_layers: (int, optional): Number of linear layers for the
                             output blocks. (default: :obj:`3`)
   :param act: (function, optional): The activation funtion.
               (default: :obj:`silu`)

   .. py:property:: num_params
      :type: int


   .. py:method:: _forward(data)


   .. py:method:: forward(data)



