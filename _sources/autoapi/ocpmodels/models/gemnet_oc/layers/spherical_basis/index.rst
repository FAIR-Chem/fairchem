:py:mod:`ocpmodels.models.gemnet_oc.layers.spherical_basis`
===========================================================

.. py:module:: ocpmodels.models.gemnet_oc.layers.spherical_basis

.. autoapi-nested-parse::

   Copyright (c) Facebook, Inc. and its affiliates.
   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   ocpmodels.models.gemnet_oc.layers.spherical_basis.CircularBasisLayer
   ocpmodels.models.gemnet_oc.layers.spherical_basis.SphericalBasisLayer




.. py:class:: CircularBasisLayer(num_spherical: int, radial_basis: ocpmodels.models.gemnet_oc.layers.radial_basis.RadialBasis, cbf: dict, scale_basis: bool = False)


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

   .. py:method:: forward(D_ca, cosφ_cab)



.. py:class:: SphericalBasisLayer(num_spherical: int, radial_basis: ocpmodels.models.gemnet_oc.layers.radial_basis.RadialBasis, sbf: dict, scale_basis: bool = False)


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

   .. py:method:: forward(D_ca, cosφ_cab, θ_cabd)



