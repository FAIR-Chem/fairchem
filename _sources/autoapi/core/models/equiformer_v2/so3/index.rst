core.models.equiformer_v2.so3
=============================

.. py:module:: core.models.equiformer_v2.so3

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.


   .. todo::

      1. Simplify the case when `num_resolutions` == 1.
      2. Remove indexing when the shape is the same.
      3. Move some functions outside classes and to separate files.



Classes
-------

.. autoapisummary::

   core.models.equiformer_v2.so3.CoefficientMappingModule
   core.models.equiformer_v2.so3.SO3_Embedding
   core.models.equiformer_v2.so3.SO3_Rotation
   core.models.equiformer_v2.so3.SO3_Grid
   core.models.equiformer_v2.so3.SO3_Linear
   core.models.equiformer_v2.so3.SO3_LinearV2


Module Contents
---------------

.. py:class:: CoefficientMappingModule(lmax_list: list[int], mmax_list: list[int])

   Bases: :py:obj:`torch.nn.Module`


   Helper module for coefficients used to reshape lval <--> m and to get coefficients of specific degree or order

   :param lmax_list (list: int):   List of maximum degree of the spherical harmonics
   :param mmax_list (list: int):   List of maximum order of the spherical harmonics


   .. py:attribute:: lmax_list


   .. py:attribute:: mmax_list


   .. py:attribute:: num_resolutions


   .. py:attribute:: device
      :value: 'cpu'



   .. py:attribute:: mask_indices_cache
      :value: None



   .. py:attribute:: rotate_inv_rescale_cache
      :value: None



   .. py:method:: complex_idx(m: int, lmax: int, m_complex, l_harmonic)

      Add `m_complex` and `l_harmonic` to the input arguments
      since we cannot use `self.m_complex`.



   .. py:method:: coefficient_idx(lmax: int, mmax: int)


   .. py:method:: get_rotate_inv_rescale(lmax: int, mmax: int)


   .. py:method:: __repr__() -> str


.. py:class:: SO3_Embedding(length: int, lmax_list: list[int], num_channels: int, device: torch.device, dtype: torch.dtype)

   Helper functions for performing operations on irreps embedding

   :param length: Batch size
   :type length: int
   :param lmax_list (list: int):   List of maximum degree of the spherical harmonics
   :param num_channels: Number of channels
   :type num_channels: int
   :param device: Device of the output
   :param dtype: type of the output tensors


   .. py:attribute:: num_channels


   .. py:attribute:: device


   .. py:attribute:: dtype


   .. py:attribute:: num_resolutions


   .. py:attribute:: num_coefficients
      :value: 0



   .. py:method:: clone() -> SO3_Embedding


   .. py:method:: set_embedding(embedding) -> None


   .. py:method:: set_lmax_mmax(lmax_list: list[int], mmax_list: list[int]) -> None


   .. py:method:: _expand_edge(edge_index: torch.Tensor) -> None


   .. py:method:: expand_edge(edge_index: torch.Tensor)


   .. py:method:: _reduce_edge(edge_index: torch.Tensor, num_nodes: int)


   .. py:method:: _m_primary(mapping)


   .. py:method:: _l_primary(mapping)


   .. py:method:: _rotate(SO3_rotation, lmax_list: list[int], mmax_list: list[int])


   .. py:method:: _rotate_inv(SO3_rotation, mappingReduced)


   .. py:method:: _grid_act(SO3_grid, act, mappingReduced)


   .. py:method:: to_grid(SO3_grid, lmax=-1)


   .. py:method:: _from_grid(x_grid, SO3_grid, lmax: int = -1)


.. py:class:: SO3_Rotation(lmax: int)

   Bases: :py:obj:`torch.nn.Module`


   Helper functions for Wigner-D rotations

   :param lmax_list (list: int):   List of maximum degree of the spherical harmonics


   .. py:attribute:: lmax


   .. py:attribute:: mapping


   .. py:method:: set_wigner(rot_mat3x3)


   .. py:method:: rotate(embedding, out_lmax: int, out_mmax: int)


   .. py:method:: rotate_inv(embedding, in_lmax: int, in_mmax: int)


   .. py:method:: RotationToWignerDMatrix(edge_rot_mat, start_lmax: int, end_lmax: int) -> torch.Tensor


.. py:class:: SO3_Grid(lmax: int, mmax: int, normalization: str = 'integral', resolution: int | None = None)

   Bases: :py:obj:`torch.nn.Module`


   Helper functions for grid representation of the irreps

   :param lmax: Maximum degree of the spherical harmonics
   :type lmax: int
   :param mmax: Maximum order of the spherical harmonics
   :type mmax: int


   .. py:attribute:: lmax


   .. py:attribute:: mmax


   .. py:attribute:: lat_resolution


   .. py:attribute:: mapping


   .. py:method:: get_to_grid_mat(device)


   .. py:method:: get_from_grid_mat(device)


   .. py:method:: to_grid(embedding, lmax: int, mmax: int)


   .. py:method:: from_grid(grid, lmax: int, mmax: int)


.. py:class:: SO3_Linear(in_features: int, out_features: int, lmax: int, bias: bool = True)

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


   .. py:attribute:: in_features


   .. py:attribute:: out_features


   .. py:attribute:: lmax


   .. py:attribute:: linear_list


   .. py:method:: forward(input_embedding, output_scale=None)


   .. py:method:: __repr__() -> str


.. py:class:: SO3_LinearV2(in_features: int, out_features: int, lmax: int, bias: bool = True)

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


   .. py:attribute:: in_features


   .. py:attribute:: out_features


   .. py:attribute:: lmax


   .. py:attribute:: weight


   .. py:attribute:: bias


   .. py:method:: forward(input_embedding)


   .. py:method:: __repr__() -> str


