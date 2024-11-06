core.models.gemnet_oc.layers.spherical_basis
============================================

.. py:module:: core.models.gemnet_oc.layers.spherical_basis

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.
   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.models.gemnet_oc.layers.spherical_basis.CircularBasisLayer
   core.models.gemnet_oc.layers.spherical_basis.SphericalBasisLayer


Module Contents
---------------

.. py:class:: CircularBasisLayer(num_spherical: int, radial_basis: core.models.gemnet_oc.layers.radial_basis.RadialBasis, cbf: dict, scale_basis: bool = False)

   Bases: :py:obj:`torch.nn.Module`


   2D Fourier Bessel Basis

   :param num_spherical: Number of basis functions. Controls the maximum frequency.
   :type num_spherical: int
   :param radial_basis: Radial basis function.
   :type radial_basis: RadialBasis
   :param cbf: Name and hyperparameters of the circular basis function.
   :type cbf: dict
   :param scale_basis: Whether to scale the basis values for better numerical stability.
   :type scale_basis: bool


   .. py:attribute:: radial_basis


   .. py:attribute:: scale_basis


   .. py:method:: forward(D_ca, cosφ_cab)


.. py:class:: SphericalBasisLayer(num_spherical: int, radial_basis: core.models.gemnet_oc.layers.radial_basis.RadialBasis, sbf: dict, scale_basis: bool = False)

   Bases: :py:obj:`torch.nn.Module`


   3D Fourier Bessel Basis

   :param num_spherical: Number of basis functions. Controls the maximum frequency.
   :type num_spherical: int
   :param radial_basis: Radial basis functions.
   :type radial_basis: RadialBasis
   :param sbf: Name and hyperparameters of the spherical basis function.
   :type sbf: dict
   :param scale_basis: Whether to scale the basis values for better numerical stability.
   :type scale_basis: bool


   .. py:attribute:: num_spherical


   .. py:attribute:: radial_basis


   .. py:attribute:: scale_basis


   .. py:method:: forward(D_ca, cosφ_cab, θ_cabd)


