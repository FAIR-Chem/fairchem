:py:mod:`ocpmodels.models.scn.spherical_harmonics`
==================================================

.. py:module:: ocpmodels.models.scn.spherical_harmonics

.. autoapi-nested-parse::

   Copyright (c) Facebook, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   ocpmodels.models.scn.spherical_harmonics.SphericalHarmonicsHelper



Functions
~~~~~~~~~

.. autoapisummary::

   ocpmodels.models.scn.spherical_harmonics.wigner_D
   ocpmodels.models.scn.spherical_harmonics._z_rot_mat



Attributes
~~~~~~~~~~

.. autoapisummary::

   ocpmodels.models.scn.spherical_harmonics._Jd


.. py:data:: _Jd

   

.. py:class:: SphericalHarmonicsHelper(lmax: int, mmax: int, num_taps: int, num_bands: int)


   Helper functions for spherical harmonics calculations and representations

   :param lmax: Maximum degree of the spherical harmonics
   :type lmax: int
   :param mmax: Maximum order of the spherical harmonics
   :type mmax: int
   :param num_taps: Number of taps or rotations (1 or otherwise set automatically based on mmax)
   :type num_taps: int
   :param num_bands: Number of bands used during message aggregation for the 1x1 pointwise convolution (1 or 2)
   :type num_bands: int

   .. py:method:: InitWignerDMatrix(edge_rot_mat) -> None


   .. py:method:: InitYRotMapping()


   .. py:method:: ToGrid(x, channels) -> torch.Tensor


   .. py:method:: FromGrid(x_grid, channels) -> torch.Tensor


   .. py:method:: CombineYRotations(x) -> torch.Tensor


   .. py:method:: Rotate(x) -> torch.Tensor


   .. py:method:: FlipGrid(grid, num_channels: int) -> torch.Tensor


   .. py:method:: RotateInv(x) -> torch.Tensor


   .. py:method:: RotateWigner(x, wigner) -> torch.Tensor


   .. py:method:: RotationMatrix(rot_x: float, rot_y: float, rot_z: float) -> torch.Tensor


   .. py:method:: RotationToWignerDMatrix(edge_rot_mat, start_lmax, end_lmax)



.. py:function:: wigner_D(l, alpha, beta, gamma)


.. py:function:: _z_rot_mat(angle, l)


