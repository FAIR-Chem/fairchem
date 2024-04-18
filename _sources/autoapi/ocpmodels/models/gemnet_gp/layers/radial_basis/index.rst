:py:mod:`ocpmodels.models.gemnet_gp.layers.radial_basis`
========================================================

.. py:module:: ocpmodels.models.gemnet_gp.layers.radial_basis

.. autoapi-nested-parse::

   Copyright (c) Facebook, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   ocpmodels.models.gemnet_gp.layers.radial_basis.PolynomialEnvelope
   ocpmodels.models.gemnet_gp.layers.radial_basis.ExponentialEnvelope
   ocpmodels.models.gemnet_gp.layers.radial_basis.SphericalBesselBasis
   ocpmodels.models.gemnet_gp.layers.radial_basis.BernsteinBasis
   ocpmodels.models.gemnet_gp.layers.radial_basis.RadialBasis




.. py:class:: PolynomialEnvelope(exponent: int)


   Bases: :py:obj:`torch.nn.Module`

   Polynomial envelope function that ensures a smooth cutoff.

   :param exponent: Exponent of the envelope function.
   :type exponent: int

   .. py:method:: forward(d_scaled: torch.Tensor) -> torch.Tensor



.. py:class:: ExponentialEnvelope


   Bases: :py:obj:`torch.nn.Module`

   Exponential envelope function that ensures a smooth cutoff,
   as proposed in Unke, Chmiela, Gastegger, Schütt, Sauceda, Müller 2021.
   SpookyNet: Learning Force Fields with Electronic Degrees of Freedom
   and Nonlocal Effects

   .. py:method:: forward(d_scaled) -> torch.Tensor



.. py:class:: SphericalBesselBasis(num_radial: int, cutoff: float)


   Bases: :py:obj:`torch.nn.Module`

   1D spherical Bessel basis

   :param num_radial: Controls maximum frequency.
   :type num_radial: int
   :param cutoff: Cutoff distance in Angstrom.
   :type cutoff: float

   .. py:method:: forward(d_scaled)



.. py:class:: BernsteinBasis(num_radial: int, pregamma_initial: float = 0.45264)


   Bases: :py:obj:`torch.nn.Module`

   Bernstein polynomial basis,
   as proposed in Unke, Chmiela, Gastegger, Schütt, Sauceda, Müller 2021.
   SpookyNet: Learning Force Fields with Electronic Degrees of Freedom
   and Nonlocal Effects

   :param num_radial: Controls maximum frequency.
   :type num_radial: int
   :param pregamma_initial: Initial value of exponential coefficient gamma.
                            Default: gamma = 0.5 * a_0**-1 = 0.94486,
                            inverse softplus -> pregamma = log e**gamma - 1 = 0.45264
   :type pregamma_initial: float

   .. py:method:: forward(d_scaled) -> torch.Tensor



.. py:class:: RadialBasis(num_radial: int, cutoff: float, rbf: Dict[str, str] = {'name': 'gaussian'}, envelope: Dict[str, Union[str, int]] = {'name': 'polynomial', 'exponent': 5})


   Bases: :py:obj:`torch.nn.Module`

   :param num_radial: Controls maximum frequency.
   :type num_radial: int
   :param cutoff: Cutoff distance in Angstrom.
   :type cutoff: float
   :param rbf: Basis function and its hyperparameters.
   :type rbf: dict = {"name": "gaussian"}
   :param envelope: Envelope function and its hyperparameters.
   :type envelope: dict = {"name": "polynomial", "exponent": 5}

   .. py:method:: forward(d)



