"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math

import numpy as np
import sympy as sym
import torch
from scipy import special as sp
from scipy.optimize import brentq


def Jn(r: float, n: int):
    """
    numerical spherical bessel functions of order n
    """
    return sp.spherical_jn(n, r)


def Jn_zeros(n: int, k: int):
    """
    Compute the first k zeros of the spherical bessel functions
    up to order n (excluded)
    """
    zerosj = np.zeros((n, k), dtype="float32")
    zerosj[0] = np.arange(1, k + 1) * np.pi
    points = np.arange(1, k + n) * np.pi
    racines = np.zeros(k + n - 1, dtype="float32")
    for i in range(1, n):
        for j in range(k + n - 1 - i):
            racines[j] = brentq(Jn, points[j], points[j + 1], (i,))
        points = racines
        zerosj[i][:k] = racines[:k]

    return zerosj


def spherical_bessel_formulas(n: int):
    """
    Computes the sympy formulas for the spherical bessel functions
    up to order n (excluded)
    """
    x = sym.symbols("x", real=True)
    # j_i = (-x)^i * (1/x * d/dx)^î * sin(x)/x
    j = [sym.sin(x) / x]  # j_0
    a = sym.sin(x) / x
    for i in range(1, n):
        b = sym.diff(a, x) / x
        j += [sym.simplify(b * (-x) ** i)]
        a = sym.simplify(b)
    return j


def bessel_basis(n: int, k: int):
    """
    Compute the sympy formulas for the normalized and rescaled spherical bessel
    functions up to order n (excluded) and maximum frequency k (excluded).

    Returns
    -------
    bess_basis: list
        Bessel basis formulas taking in a single argument x.
        Has length n where each element has length k. -> In total n*k many.
    """
    zeros = Jn_zeros(n, k)
    normalizer = []
    for order in range(n):
        normalizer_tmp = []
        for i in range(k):
            normalizer_tmp += [0.5 * Jn(zeros[order, i], order + 1) ** 2]
        normalizer_tmp = (
            1 / np.array(normalizer_tmp) ** 0.5
        )  # sqrt(2/(j_l+1)**2) , sqrt(1/c**3) not taken into account yet
        normalizer += [normalizer_tmp]

    f = spherical_bessel_formulas(n)
    x = sym.symbols("x", real=True)
    bess_basis = []
    for order in range(n):
        bess_basis_tmp = []
        for i in range(k):
            bess_basis_tmp += [
                sym.simplify(
                    normalizer[order][i]
                    * f[order].subs(x, zeros[order, i] * x)
                )
            ]
        bess_basis += [bess_basis_tmp]
    return bess_basis


def sph_harm_prefactor(l_degree: int, m_order: int):
    """
    Computes the constant pre-factor for the spherical harmonic
    of degree l and order m.

    Arguments
    ---------
    l_degree: int
        Degree of the spherical harmonic. l >= 0
    m_order: int
        Order of the spherical harmonic. -l <= m <= l

    Returns
    -------
    factor: float

    """
    # sqrt((2*l+1)/4*pi * (l-m)!/(l+m)! )
    return (
        (2 * l_degree + 1)
        / (4 * np.pi)
        * math.factorial(l_degree - abs(m_order))
        / math.factorial(l_degree + abs(m_order))
    ) ** 0.5


def associated_legendre_polynomials(
    L_maxdegree: int, zero_m_only: bool = True, pos_m_only: bool = True
):
    """
    Computes string formulas of the associated legendre polynomials
    up to degree L (excluded).

    Arguments
    ---------
    L_maxdegree: int
        Degree up to which to calculate the associated legendre polynomials
        (degree L is excluded).
    zero_m_only: bool
        If True only calculate the polynomials for the polynomials where m=0.
    pos_m_only: bool
        If True only calculate the polynomials for the polynomials where m>=0.
        Overwritten by zero_m_only.

    Returns
    -------
    polynomials: list
        Contains the sympy functions of the polynomials
        (in total L many if zero_m_only is True else L^2 many).
    """
    # calculations from http://web.cmb.usc.edu/people/alber/Software/tomominer/docs/cpp/group__legendre__polynomials.html
    z = sym.symbols("z", real=True)
    P_l_m = [
        [0] * (2 * l_degree + 1) for l_degree in range(L_maxdegree)
    ]  # for order l: -l <= m <= l

    P_l_m[0][0] = 1
    if L_maxdegree > 1:
        if zero_m_only:
            # m = 0
            P_l_m[1][0] = z
            for l_degree in range(2, L_maxdegree):
                P_l_m[l_degree][0] = sym.simplify(
                    (
                        (2 * l_degree - 1) * z * P_l_m[l_degree - 1][0]
                        - (l_degree - 1) * P_l_m[l_degree - 2][0]
                    )
                    / l_degree
                )
            return P_l_m
        else:
            # for m >= 0
            for l_degree in range(1, L_maxdegree):
                P_l_m[l_degree][l_degree] = sym.simplify(
                    (1 - 2 * l_degree)
                    * (1 - z**2) ** 0.5
                    * P_l_m[l_degree - 1][l_degree - 1]
                )  # P_00, P_11, P_22, P_33

            for m_order in range(0, L_maxdegree - 1):
                P_l_m[m_order + 1][m_order] = sym.simplify(
                    (2 * m_order + 1) * z * P_l_m[m_order][m_order]
                )  # P_10, P_21, P_32, P_43

            for l_degree in range(2, L_maxdegree):
                for m_order in range(l_degree - 1):  # P_20, P_30, P_31
                    P_l_m[l_degree][m_order] = sym.simplify(
                        (
                            (2 * l_degree - 1)
                            * z
                            * P_l_m[l_degree - 1][m_order]
                            - (l_degree + m_order - 1)
                            * P_l_m[l_degree - 2][m_order]
                        )
                        / (l_degree - m_order)
                    )

            if not pos_m_only:
                # for m < 0: P_l(-m) = (-1)^m * (l-m)!/(l+m)! * P_lm
                for l_degree in range(1, L_maxdegree):
                    for m_order in range(
                        1, l_degree + 1
                    ):  # P_1(-1), P_2(-1) P_2(-2)
                        P_l_m[l_degree][-m_order] = sym.simplify(
                            (-1) ** m_order
                            * math.factorial(l_degree - m_order)
                            / math.factorial(l_degree + m_order)
                            * P_l_m[l_degree][m_order]
                        )

            return P_l_m


def real_sph_harm(
    L_maxdegree: int,
    use_theta: bool,
    use_phi: bool = True,
    zero_m_only: bool = True,
) -> None:
    """
    Computes formula strings of the the real part of the spherical harmonics
    up to degree L (excluded). Variables are either spherical coordinates phi
    and theta (or cartesian coordinates x,y,z) on the UNIT SPHERE.

    Arguments
    ---------
    L_maxdegree: int
        Degree up to which to calculate the spherical harmonics
        (degree L is excluded).
    use_theta: bool
        - True: Expects the input of the formula strings to contain theta.
        - False: Expects the input of the formula strings to contain z.
    use_phi: bool
        - True: Expects the input of the formula strings to contain phi.
        - False: Expects the input of the formula strings to contain x and y.
        Does nothing if zero_m_only is True
    zero_m_only: bool
        If True only calculate the harmonics where m=0.

    Returns
    -------
    Y_lm_real: list
        Computes formula strings of the the real part of the spherical
        harmonics up to degree L (where degree L is not excluded).
        In total L^2 many sph harm exist up to degree L (excluded).
        However, if zero_m_only only is True then the total count
        is reduced to L.
    """
    z = sym.symbols("z", real=True)
    P_l_m = associated_legendre_polynomials(L_maxdegree, zero_m_only)
    if zero_m_only:
        # for all m != 0: Y_lm = 0
        Y_l_m = [[0] for l_degree in range(L_maxdegree)]
    else:
        Y_l_m = [
            [0] * (2 * l_degree + 1) for l_degree in range(L_maxdegree)
        ]  # for order l: -l <= m <= l

    # convert expressions to spherical coordiantes
    if use_theta:
        # replace z by cos(theta)
        theta = sym.symbols("theta", real=True)
        for l_degree in range(L_maxdegree):
            for m_order in range(len(P_l_m[l_degree])):
                if not isinstance(P_l_m[l_degree][m_order], int):
                    P_l_m[l_degree][m_order] = P_l_m[l_degree][m_order].subs(
                        z, sym.cos(theta)
                    )

    ## calculate Y_lm
    # Y_lm = N * P_lm(cos(theta)) * exp(i*m*phi)
    #             { sqrt(2) * (-1)^m * N * P_l|m| * sin(|m|*phi)   if m < 0
    # Y_lm_real = { Y_lm                                           if m = 0
    #             { sqrt(2) * (-1)^m * N * P_lm * cos(m*phi)       if m > 0

    for l_degree in range(L_maxdegree):
        Y_l_m[l_degree][0] = sym.simplify(
            sph_harm_prefactor(l_degree, 0) * P_l_m[l_degree][0]
        )  # Y_l0

    if not zero_m_only:
        phi = sym.symbols("phi", real=True)
        for l_degree in range(1, L_maxdegree):
            # m > 0
            for m_order in range(1, l_degree + 1):
                Y_l_m[l_degree][m_order] = sym.simplify(
                    2**0.5
                    * (-1) ** m_order
                    * sph_harm_prefactor(l_degree, m_order)
                    * P_l_m[l_degree][m_order]
                    * sym.cos(m_order * phi)
                )
            # m < 0
            for m_order in range(1, l_degree + 1):
                Y_l_m[l_degree][-m_order] = sym.simplify(
                    2**0.5
                    * (-1) ** m_order
                    * sph_harm_prefactor(l_degree, -m_order)
                    * P_l_m[l_degree][m_order]
                    * sym.sin(m_order * phi)
                )

        # convert expressions to cartesian coordinates
        if not use_phi:
            # replace phi by atan2(y,x)
            x, y = sym.symbols("x y", real=True)
            for l_degree in range(L_maxdegree):
                for m_order in range(len(Y_l_m[l_degree])):
                    Y_l_m[l_degree][m_order] = sym.simplify(
                        Y_l_m[l_degree][m_order].subs(phi, sym.atan2(y, x))
                    )
    return Y_l_m


def get_sph_harm_basis(L_maxdegree: int, zero_m_only: bool = True):
    """Get a function calculating the spherical harmonics basis from z and phi."""
    # retrieve equations
    Y_lm = real_sph_harm(
        L_maxdegree, use_theta=False, use_phi=True, zero_m_only=zero_m_only
    )
    Y_lm_flat = [Y for Y_l in Y_lm for Y in Y_l]

    # convert to pytorch functions
    z = sym.symbols("z", real=True)
    variables = [z]
    if not zero_m_only:
        variables.append(sym.symbols("phi", real=True))

    modules = {"sin": torch.sin, "cos": torch.cos, "sqrt": torch.sqrt}
    sph_funcs = sym.lambdify(variables, Y_lm_flat, modules)

    # Return as a single function
    # args are either [cosφ] or [cosφ, ϑ]
    def basis_fn(*args) -> torch.Tensor:
        basis = sph_funcs(*args)
        basis[0] = args[0].new_tensor(basis[0]).expand_as(args[0])
        return torch.stack(basis, dim=1)

    return basis_fn
