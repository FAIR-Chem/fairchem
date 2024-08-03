core.models.gemnet_gp.layers.radial_basis
=========================================

.. py:module:: core.models.gemnet_gp.layers.radial_basis

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.models.gemnet_gp.layers.radial_basis.PolynomialEnvelope
   core.models.gemnet_gp.layers.radial_basis.ExponentialEnvelope
   core.models.gemnet_gp.layers.radial_basis.SphericalBesselBasis
   core.models.gemnet_gp.layers.radial_basis.BernsteinBasis
   core.models.gemnet_gp.layers.radial_basis.RadialBasis


Module Contents
---------------

.. py:class:: PolynomialEnvelope(exponent: int)

   Bases: :py:obj:`torch.nn.Module`


   Polynomial envelope function that ensures a smooth cutoff.

   :param exponent: Exponent of the envelope function.
   :type exponent: int


   .. py:attribute:: p


   .. py:attribute:: a
      :type:  float


   .. py:attribute:: b
      :type:  float


   .. py:attribute:: c
      :type:  float


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


   .. py:attribute:: norm_const


   .. py:attribute:: frequencies


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


   .. py:attribute:: prefactor


   .. py:attribute:: pregamma


   .. py:attribute:: softplus


   .. py:attribute:: exp1


   .. py:attribute:: exp2


   .. py:method:: forward(d_scaled) -> torch.Tensor


.. py:class:: RadialBasis(num_radial: int, cutoff: float, rbf: dict[str, str] | None = None, envelope: dict[str, str | int] | None = None)

   Bases: :py:obj:`torch.nn.Module`


   :param num_radial: Controls maximum frequency.
   :type num_radial: int
   :param cutoff: Cutoff distance in Angstrom.
   :type cutoff: float
   :param rbf: Basis function and its hyperparameters.
   :type rbf: dict = {"name": "gaussian"}
   :param envelope: Envelope function and its hyperparameters.
   :type envelope: dict = {"name": "polynomial", "exponent": 5}


   .. py:attribute:: inv_cutoff


   .. py:attribute:: env_name


   .. py:attribute:: env_hparams


   .. py:attribute:: rbf_name


   .. py:attribute:: rbf_hparams


   .. py:method:: forward(d)


