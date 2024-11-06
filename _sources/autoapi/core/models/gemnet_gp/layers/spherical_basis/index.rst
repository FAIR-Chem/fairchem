core.models.gemnet_gp.layers.spherical_basis
============================================

.. py:module:: core.models.gemnet_gp.layers.spherical_basis

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.models.gemnet_gp.layers.spherical_basis.CircularBasisLayer


Module Contents
---------------

.. py:class:: CircularBasisLayer(num_spherical: int, radial_basis: core.models.gemnet_gp.layers.radial_basis.RadialBasis, cbf, efficient: bool = False)

   Bases: :py:obj:`torch.nn.Module`


   2D Fourier Bessel Basis

   :param num_spherical: Controls maximum frequency.
   :type num_spherical: int
   :param radial_basis: Radial basis functions
   :type radial_basis: RadialBasis
   :param cbf: Name and hyperparameters of the cosine basis function
   :type cbf: dict
   :param efficient: Whether to use the "efficient" summation order
   :type efficient: bool


   .. py:attribute:: radial_basis


   .. py:attribute:: efficient


   .. py:method:: forward(D_ca, cosφ_cab, id3_ca)


