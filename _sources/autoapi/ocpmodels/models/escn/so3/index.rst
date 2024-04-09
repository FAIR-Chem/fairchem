:py:mod:`ocpmodels.models.escn.so3`
===================================

.. py:module:: ocpmodels.models.escn.so3

.. autoapi-nested-parse::

   Copyright (c) Facebook, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   ocpmodels.models.escn.so3.CoefficientMapping
   ocpmodels.models.escn.so3.SO3_Embedding
   ocpmodels.models.escn.so3.SO3_Rotation
   ocpmodels.models.escn.so3.SO3_Grid




Attributes
~~~~~~~~~~

.. autoapisummary::

   ocpmodels.models.escn.so3._Jd


.. py:data:: _Jd

   

.. py:class:: CoefficientMapping(lmax_list: List[int], mmax_list: List[int], device)


   Helper functions for coefficients used to reshape l<-->m and to get coefficients of specific degree or order

   :param lmax_list (list: int):   List of maximum degree of the spherical harmonics
   :param mmax_list (list: int):   List of maximum order of the spherical harmonics
   :param device: Device of the output

   .. py:method:: complex_idx(m, lmax: int = -1)


   .. py:method:: coefficient_idx(lmax: int, mmax: int) -> torch.Tensor



.. py:class:: SO3_Embedding(length: int, lmax_list: List[int], num_channels: int, device: torch.device, dtype: torch.dtype)


   Bases: :py:obj:`torch.nn.Module`

   Helper functions for irreps embedding

   :param length: Batch size
   :type length: int
   :param lmax_list (list: int):   List of maximum degree of the spherical harmonics
   :param num_channels: Number of channels
   :type num_channels: int
   :param device: Device of the output
   :param dtype: type of the output tensors

   .. py:method:: clone() -> SO3_Embedding


   .. py:method:: set_embedding(embedding) -> None


   .. py:method:: set_lmax_mmax(lmax_list, mmax_list) -> None


   .. py:method:: _expand_edge(edge_index) -> None


   .. py:method:: expand_edge(edge_index) -> SO3_Embedding


   .. py:method:: _reduce_edge(edge_index, num_nodes: int) -> None


   .. py:method:: _m_primary(mapping) -> None


   .. py:method:: _l_primary(mapping) -> None


   .. py:method:: _rotate(SO3_rotation, lmax_list, mmax_list) -> None


   .. py:method:: _rotate_inv(SO3_rotation, mappingReduced) -> None


   .. py:method:: _grid_act(SO3_grid, act, mappingReduced) -> None


   .. py:method:: to_grid(SO3_grid, lmax: int = -1) -> torch.Tensor


   .. py:method:: _from_grid(x_grid, SO3_grid, lmax: int = -1) -> None



.. py:class:: SO3_Rotation(rot_mat3x3: torch.Tensor, lmax: List[int])


   Bases: :py:obj:`torch.nn.Module`

   Helper functions for Wigner-D rotations

   :param rot_mat3x3: Rotation matrix
   :type rot_mat3x3: tensor
   :param lmax_list (list: int):   List of maximum degree of the spherical harmonics

   .. py:method:: set_lmax(lmax) -> None


   .. py:method:: rotate(embedding, out_lmax, out_mmax) -> torch.Tensor


   .. py:method:: rotate_inv(embedding, in_lmax, in_mmax) -> torch.Tensor


   .. py:method:: RotationToWignerDMatrix(edge_rot_mat: torch.Tensor, start_lmax: int, end_lmax: int) -> torch.Tensor


   .. py:method:: wigner_D(lval, alpha, beta, gamma)


   .. py:method:: _z_rot_mat(angle: torch.Tensor, lv: int) -> torch.Tensor



.. py:class:: SO3_Grid(lmax: int, mmax: int)


   Bases: :py:obj:`torch.nn.Module`

   Helper functions for grid representation of the irreps

   :param lmax: Maximum degree of the spherical harmonics
   :type lmax: int
   :param mmax: Maximum order of the spherical harmonics
   :type mmax: int

   .. py:method:: _initialize(device: torch.device) -> None


   .. py:method:: get_to_grid_mat(device: torch.device)


   .. py:method:: get_from_grid_mat(device: torch.device)


   .. py:method:: to_grid(embedding: torch.Tensor, lmax: int, mmax: int) -> torch.Tensor


   .. py:method:: from_grid(grid: torch.Tensor, lmax: int, mmax: int) -> torch.Tensor



