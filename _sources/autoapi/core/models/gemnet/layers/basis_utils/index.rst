:py:mod:`core.models.gemnet.layers.basis_utils`
===============================================

.. py:module:: core.models.gemnet.layers.basis_utils

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   core.models.gemnet.layers.basis_utils.Jn
   core.models.gemnet.layers.basis_utils.Jn_zeros
   core.models.gemnet.layers.basis_utils.spherical_bessel_formulas
   core.models.gemnet.layers.basis_utils.bessel_basis
   core.models.gemnet.layers.basis_utils.sph_harm_prefactor
   core.models.gemnet.layers.basis_utils.associated_legendre_polynomials
   core.models.gemnet.layers.basis_utils.real_sph_harm



.. py:function:: Jn(r: float, n: int)

   numerical spherical bessel functions of order n


.. py:function:: Jn_zeros(n: int, k: int)

   Compute the first k zeros of the spherical bessel functions up to order n (excluded)


.. py:function:: spherical_bessel_formulas(n: int)

   Computes the sympy formulas for the spherical bessel functions up to order n (excluded)


.. py:function:: bessel_basis(n: int, k: int)

   Compute the sympy formulas for the normalized and rescaled spherical bessel functions up to
   order n (excluded) and maximum frequency k (excluded).

   :returns:

             list
                 Bessel basis formulas taking in a single argument x.
                 Has length n where each element has length k. -> In total n*k many.
   :rtype: bess_basis


.. py:function:: sph_harm_prefactor(l_degree: int, m_order: int)

   Computes the constant pre-factor for the spherical harmonic of degree l and order m.

   :param l_degree: Degree of the spherical harmonic. l >= 0
   :type l_degree: int
   :param m_order: Order of the spherical harmonic. -l <= m <= l
   :type m_order: int

   :returns: **factor**
   :rtype: float


.. py:function:: associated_legendre_polynomials(L_maxdegree: int, zero_m_only: bool = True, pos_m_only: bool = True)

   Computes string formulas of the associated legendre polynomials up to degree L (excluded).

   :param L_maxdegree: Degree up to which to calculate the associated legendre polynomials (degree L is excluded).
   :type L_maxdegree: int
   :param zero_m_only: If True only calculate the polynomials for the polynomials where m=0.
   :type zero_m_only: bool
   :param pos_m_only: If True only calculate the polynomials for the polynomials where m>=0. Overwritten by zero_m_only.
   :type pos_m_only: bool

   :returns: **polynomials** -- Contains the sympy functions of the polynomials (in total L many if zero_m_only is True else L^2 many).
   :rtype: list


.. py:function:: real_sph_harm(L_maxdegree: int, use_theta: bool, use_phi: bool = True, zero_m_only: bool = True)

   Computes formula strings of the the real part of the spherical harmonics up to degree L (excluded).
   Variables are either spherical coordinates phi and theta (or cartesian coordinates x,y,z) on the UNIT SPHERE.

   :param L_maxdegree: Degree up to which to calculate the spherical harmonics (degree L is excluded).
   :type L_maxdegree: int
   :param use_theta:
                     - True: Expects the input of the formula strings to contain theta.
                     - False: Expects the input of the formula strings to contain z.
   :type use_theta: bool
   :param use_phi:
                   - True: Expects the input of the formula strings to contain phi.
                   - False: Expects the input of the formula strings to contain x and y.
                   Does nothing if zero_m_only is True
   :type use_phi: bool
   :param zero_m_only: If True only calculate the harmonics where m=0.
   :type zero_m_only: bool

   :returns: **Y_lm_real** -- Computes formula strings of the the real part of the spherical harmonics up
             to degree L (where degree L is not excluded).
             In total L^2 many sph harm exist up to degree L (excluded). However, if zero_m_only only is True then
             the total count is reduced to be only L many.
   :rtype: list


