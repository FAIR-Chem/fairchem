core.models.gemnet_oc.layers.radial_basis
=========================================

.. py:module:: core.models.gemnet_oc.layers.radial_basis

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.
   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.models.gemnet_oc.layers.radial_basis.PolynomialEnvelope
   core.models.gemnet_oc.layers.radial_basis.ExponentialEnvelope
   core.models.gemnet_oc.layers.radial_basis.GaussianBasis
   core.models.gemnet_oc.layers.radial_basis.SphericalBesselBasis
   core.models.gemnet_oc.layers.radial_basis.BernsteinBasis
   core.models.gemnet_oc.layers.radial_basis.RadialBasis


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


   .. py:method:: forward(d_scaled: torch.Tensor) -> torch.Tensor


.. py:class:: GaussianBasis(start: float = 0.0, stop: float = 5.0, num_gaussians: int = 50, trainable: bool = False)

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


   .. py:attribute:: coeff


   .. py:method:: forward(dist: torch.Tensor) -> torch.Tensor


.. py:class:: SphericalBesselBasis(num_radial: int, cutoff: float)

   Bases: :py:obj:`torch.nn.Module`


   First-order spherical Bessel basis

   :param num_radial: Number of basis functions. Controls the maximum frequency.
   :type num_radial: int
   :param cutoff: Cutoff distance in Angstrom.
   :type cutoff: float


   .. py:attribute:: norm_const


   .. py:attribute:: frequencies


   .. py:method:: forward(d_scaled: torch.Tensor) -> torch.Tensor


.. py:class:: BernsteinBasis(num_radial: int, pregamma_initial: float = 0.45264)

   Bases: :py:obj:`torch.nn.Module`


   Bernstein polynomial basis,
   as proposed in Unke, Chmiela, Gastegger, Schütt, Sauceda, Müller 2021.
   SpookyNet: Learning Force Fields with Electronic Degrees of Freedom
   and Nonlocal Effects

   :param num_radial: Number of basis functions. Controls the maximum frequency.
   :type num_radial: int
   :param pregamma_initial: Initial value of exponential coefficient gamma.
                            Default: gamma = 0.5 * a_0**-1 = 0.94486,
                            inverse softplus -> pregamma = log e**gamma - 1 = 0.45264
   :type pregamma_initial: float


   .. py:attribute:: pregamma


   .. py:attribute:: softplus


   .. py:method:: forward(d_scaled: torch.Tensor) -> torch.Tensor


.. py:class:: RadialBasis(num_radial: int, cutoff: float, rbf: dict[str, str] | None = None, envelope: dict[str, str | int] | None = None, scale_basis: bool = False)

   Bases: :py:obj:`torch.nn.Module`


   :param num_radial: Number of basis functions. Controls the maximum frequency.
   :type num_radial: int
   :param cutoff: Cutoff distance in Angstrom.
   :type cutoff: float
   :param rbf: Basis function and its hyperparameters.
   :type rbf: dict = {"name": "gaussian"}
   :param envelope: Envelope function and its hyperparameters.
   :type envelope: dict = {"name": "polynomial", "exponent": 5}
   :param scale_basis: Whether to scale the basis values for better numerical stability.
   :type scale_basis: bool


   .. py:attribute:: inv_cutoff


   .. py:attribute:: scale_basis


   .. py:method:: forward(d: torch.Tensor) -> torch.Tensor


