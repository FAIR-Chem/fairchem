import torch
import sympy as sym
import numpy as np
from scipy.optimize import brentq
from scipy import special as sp

def Jn(r, n):
    """
    numerical spherical bessel functions of order n
    """
    return sp.spherical_jn(n, r)


def Jn_zeros(n, k):
    """
    Compute the first k zeros of the spherical bessel functions up to order n (excluded)
    """
    zerosj = np.zeros((n, k), dtype="float32")
    zerosj[0] = np.arange(1, k + 1) * np.pi
    points = np.arange(1, k + n) * np.pi
    racines = np.zeros(k + n - 1, dtype="float32")
    for i in range(1, n):
        for j in range(k + n - 1 - i):
            foo = brentq(Jn, points[j], points[j + 1], (i,))
            racines[j] = foo
        points = racines
        zerosj[i][:k] = racines[:k]

    return zerosj


def spherical_bessel_formulas(n):
    """
    Computes the sympy formulas for the spherical bessel functions up to order n (excluded)
    """
    x = sym.symbols("x")
    # j_i = (-x)^i * (1/x * d/dx)^î * sin(x)/x
    j = [sym.sin(x) / x]  # j_0
    a = sym.sin(x) / x
    for i in range(1, n):
        b = sym.diff(a, x) / x
        j += [sym.simplify(b * (-x) ** i)]
        a = sym.simplify(b)
    return j


def bessel_basis(n, k):
    """
    Compute the sympy formulas for the normalized and rescaled spherical bessel functions up to
    order n (excluded) and maximum frequency k (excluded).
    Returns:
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
    x = sym.symbols("x")
    bess_basis = []
    for order in range(n):
        bess_basis_tmp = []
        for i in range(k):
            bess_basis_tmp += [
                sym.simplify(
                    normalizer[order][i] * f[order].subs(x, zeros[order, i] * x)
                )
            ]
        bess_basis += [bess_basis_tmp]
    return bess_basis


def sph_harm_prefactor(l, m):
    """Computes the constant pre-factor for the spherical harmonic of degree l and order m.
    Parameters
    ----------
        l: int
            Degree of the spherical harmonic. l >= 0
        m: int
            Order of the spherical harmonic. -l <= m <= l
    Returns
    -------
        factor: float
    """
    # sqrt((2*l+1)/4*pi * (l-m)!/(l+m)! )
    return (
        (2 * l + 1)
        / (4 * np.pi)
        * np.math.factorial(l - abs(m))
        / np.math.factorial(l + abs(m))
    ) ** 0.5


def associated_legendre_polynomials(L, zero_m_only=True, pos_m_only=True):
    """Computes string formulas of the associated legendre polynomials up to degree L (excluded).
    Parameters
    ----------
        L: int
            Degree up to which to calculate the associated legendre polynomials (degree L is excluded).
        zero_m_only: bool
            If True only calculate the polynomials for the polynomials where m=0.
        pos_m_only: bool
            If True only calculate the polynomials for the polynomials where m>=0. Overwritten by zero_m_only.
    Returns
    -------
        polynomials: list
            Contains the sympy functions of the polynomials (in total L many if zero_m_only is True else L^2 many).
    """
    # calculations from http://web.cmb.usc.edu/people/alber/Software/tomominer/docs/cpp/group__legendre__polynomials.html
    z = sym.symbols("z")
    P_l_m = [[0] * (2 * l + 1) for l in range(L)]  # for order l: -l <= m <= l

    P_l_m[0][0] = 1
    if L > 0:
        if zero_m_only:
            # m = 0
            P_l_m[1][0] = z
            for l in range(2, L):
                P_l_m[l][0] = sym.simplify(
                    ((2 * l - 1) * z * P_l_m[l - 1][0] - (l - 1) * P_l_m[l - 2][0]) / l
                )
            return P_l_m
        else:
            # for m >= 0
            for l in range(1, L):
                P_l_m[l][l] = sym.simplify(
                    (1 - 2 * l) * (1 - z ** 2) ** 0.5 * P_l_m[l - 1][l - 1]
                )  # P_00, P_11, P_22, P_33

            for m in range(0, L - 1):
                P_l_m[m + 1][m] = sym.simplify(
                    (2 * m + 1) * z * P_l_m[m][m]
                )  # P_10, P_21, P_32, P_43

            for l in range(2, L):
                for m in range(l - 1):  # P_20, P_30, P_31
                    P_l_m[l][m] = sym.simplify(
                        (
                            (2 * l - 1) * z * P_l_m[l - 1][m]
                            - (l + m - 1) * P_l_m[l - 2][m]
                        )
                        / (l - m)
                    )

            if not pos_m_only:
                # for m < 0: P_l(-m) = (-1)^m * (l-m)!/(l+m)! * P_lm
                for l in range(1, L):
                    for m in range(1, l + 1):  # P_1(-1), P_2(-1) P_2(-2)
                        P_l_m[l][-m] = sym.simplify(
                            (-1) ** m
                            * np.math.factorial(l - m)
                            / np.math.factorial(l + m)
                            * P_l_m[l][m]
                        )

            return P_l_m


def real_sph_harm(L, spherical_coordinates, zero_m_only=True):
    """
    Computes formula strings of the the real part of the spherical harmonics up to degree L (excluded).
    Variables are either spherical coordinates phi and theta (or cartesian coordinates x,y,z) on the UNIT SPHERE.
    Parameters
    ----------
        L: int
            Degree up to which to calculate the spherical harmonics (degree L is excluded).
        spherical_coordinates: bool
            - True: Expects the input of the formula strings to be phi and theta.
            - False: Expects the input of the formula strings to be x, y and z.
        zero_m_only: bool
            If True only calculate the harmonics where m=0.
    Returns
    -------
        Y_lm_real: list
            Computes formula strings of the the real part of the spherical harmonics up
            to degree L (where degree L is not excluded).
            In total L^2 many sph harm exist up to degree L (excluded). However, if zero_m_only only is True then
            the total count is reduced to be only L many.
    """
    z = sym.symbols("z")
    P_l_m = associated_legendre_polynomials(L, zero_m_only)
    if zero_m_only:
        # for all m != 0: Y_lm = 0
        Y_l_m = [[0] for l in range(L)]
    else:
        Y_l_m = [[0] * (2 * l + 1) for l in range(L)]  # for order l: -l <= m <= l

    # convert expressions to spherical coordiantes
    if spherical_coordinates:
        # replace z by cos(theta)
        theta = sym.symbols("theta")
        for l in range(L):
            for m in range(len(P_l_m[l])):
                if not isinstance(P_l_m[l][m], int):
                    P_l_m[l][m] = P_l_m[l][m].subs(z, sym.cos(theta))

    ## calculate Y_lm
    # Y_lm = N * P_lm(cos(theta)) * exp(i*m*phi)
    #             { sqrt(2) * (-1)^m * N * P_l|m| * sin(|m|*phi)   if m < 0
    # Y_lm_real = { Y_lm                                           if m = 0
    #             { sqrt(2) * (-1)^m * N * P_lm * cos(m*phi)       if m > 0

    for l in range(L):
        Y_l_m[l][0] = sym.simplify(sph_harm_prefactor(l, 0) * P_l_m[l][0])  # Y_l0

    if not zero_m_only:
        phi = sym.symbols("phi")
        for l in range(1, L):
            # m > 0
            for m in range(1, l + 1):
                Y_l_m[l][m] = sym.simplify(
                    2 ** 0.5
                    * (-1) ** m
                    * sph_harm_prefactor(l, m)
                    * P_l_m[l][m]
                    * sym.cos(m * phi)
                )
            # m < 0
            for m in range(1, l + 1):
                Y_l_m[l][-m] = sym.simplify(
                    2 ** 0.5
                    * (-1) ** m
                    * sph_harm_prefactor(l, -m)
                    * P_l_m[l][m]
                    * sym.sin(m * phi)
                )

        # convert expressions to cartesian coordinates
        if not spherical_coordinates:
            # replace phi by atan2(y,x)
            x = sym.symbols("x")
            y = sym.symbols("y")
            for l in range(L):
                for m in range(len(Y_l_m[l])):
                    Y_l_m[l][m] = sym.simplify(Y_l_m[l][m].subs(phi, sym.atan2(y, x)))
    return Y_l_m


class angle_emb(torch.nn.Module):
    def __init__(self, num_radial, num_spherical, cutoff=8.0):
        super(angle_emb, self).__init__()
        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff

        bessel_formulas = bessel_basis(num_spherical, num_radial)
        Y_lm = real_sph_harm(
            num_spherical, spherical_coordinates=True, zero_m_only=True
        )
        self.sph_funcs = []  # (num_spherical,)
        self.bessel_funcs = []  # (num_spherical * num_radial,)

        x = sym.symbols("x")
        theta = sym.symbols("theta")
        modules = {"sin": torch.sin, "cos": torch.cos, "sqrt": torch.sqrt}
        m = 0  # only single angle
        for l in range(len(Y_lm)):  # num_spherical
            if l == 0:
                # Y_00 is only a constant -> function returns value and not tensor
                first_sph = sym.lambdify([theta], Y_lm[l][m], modules)
                self.sph_funcs.append(
                    lambda theta: torch.zeros_like(theta) + first_sph(theta)
                )
            else:
                self.sph_funcs.append(sym.lambdify([theta], Y_lm[l][m], modules))
            for n in range(num_radial):
                self.bessel_funcs.append(
                    sym.lambdify([x], bessel_formulas[l][n], modules)
                )

    def forward(self, dist, angle):
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        sbf = torch.stack([f(angle) for f in self.sph_funcs], dim=1)
        n, k = self.num_spherical, self.num_radial
        out = (rbf.view(-1, n, k) * sbf.view(-1, n, 1)).view(-1, n * k)
        return out


class torsion_emb(torch.nn.Module):
    def __init__(self, num_radial, num_spherical, cutoff=8.0):
        super(torsion_emb, self).__init__()
        assert num_radial <= 64
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.cutoff = cutoff

        bessel_formulas = bessel_basis(num_spherical, num_radial)
        Y_lm = real_sph_harm(
            num_spherical, spherical_coordinates=True, zero_m_only=False
        )
        self.sph_funcs = []  # (num_spherical**2,)
        self.bessel_funcs = []  # (num_spherical * num_radial,)

        # convert to tensorflow functions
        x = sym.symbols("x")
        theta = sym.symbols("theta")
        phi = sym.symbols("phi")
        modules = {"sin": torch.sin, "cos": torch.cos, "sqrt": torch.sqrt}
        for l in range(len(Y_lm)):  # num_spherical
            for m in range(len(Y_lm[l])):
                if (
                        l == 0
                ):  # Y_00 is only a constant -> function returns value and not tensor
                    first_sph = sym.lambdify([theta, phi], Y_lm[l][m], modules)
                    self.sph_funcs.append(
                        lambda theta, phi: torch.zeros_like(theta)
                                           + first_sph(theta, phi)
                    )
                else:
                    self.sph_funcs.append(
                        sym.lambdify([theta, phi], Y_lm[l][m], modules)
                    )
            for j in range(num_radial):
                self.bessel_funcs.append(
                    sym.lambdify([x], bessel_formulas[l][j], modules)
                )

        self.register_buffer(
            "degreeInOrder", torch.arange(num_spherical) * 2 + 1, persistent=False
        )

    def forward(self, dist, theta, phi):
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        sbf = torch.stack([f(theta, phi) for f in self.sph_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        rbf = rbf.view((-1, n, k)).repeat_interleave(self.degreeInOrder, dim=1).view((-1, n ** 2 * k))
        sbf = sbf.repeat_interleave(k, dim=1)
        out = rbf * sbf
        return out
