"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

---

This code borrows heavily from the SphereNet implementation as part of DIG:
Dive into Graphs: https://github.com/divelab/DIG. License: GPL-3.0.

# Based on the code from: https://github.com/klicperajo/dimenet,
# https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/models/dimenet_utils.py
"""

from math import pi as PI
from math import sqrt

import numpy as np
import torch
from scipy import special as sp
from scipy.optimize import brentq
from torch_geometric.nn.models.dimenet_utils import (
    Jn,
    Jn_zeros,
    associated_legendre_polynomials,
    bessel_basis,
    sph_harm_prefactor,
    spherical_bessel_formulas,
)
from torch_scatter import scatter
from torch_sparse import SparseTensor

try:
    import sympy as sym
except ImportError:
    sym = None


def xyztodat(pos, out, num_nodes, use_pbc=True, torsional=True):
    edge_index = out["edge_index"]
    j, i = edge_index  # j->i
    if use_pbc:
        offsets = out["offsets"]
        dist = out["distances"]
    else:
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

    # necessary for training stability
    with torch.no_grad():
        value = torch.arange(j.size(0), device=j.device)
        adj_t = SparseTensor(
            row=i, col=j, value=value, sparse_sizes=(num_nodes, num_nodes)
        )
        adj_t_row = adj_t[j]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        # Node indices (k->j->i) for triplets.
        idx_i = i.repeat_interleave(num_triplets)
        idx_j = j.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()
        mask = idx_i != idx_k
        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

        # Edge indices (k-j, j->i) for triplets.
        idx_kj = adj_t_row.storage.value()[mask]
        idx_ji = adj_t_row.storage.row()[mask]

        # Calculate angles. 0 to pi
        if use_pbc:
            pos_ji, pos_jk = (
                pos[idx_i] - pos[idx_j] + offsets[idx_ji],
                pos[idx_k] - pos[idx_j] + offsets[idx_kj],
            )
        else:
            pos_ji = pos[idx_i] - pos[idx_j]
            pos_jk = pos[idx_k] - pos[idx_j]

        a = (pos_ji * pos_jk).sum(dim=-1)  # cos_angle * |pos_ji| * |pos_jk|
        b = torch.cross(pos_ji, pos_jk).norm(
            dim=-1
        )  # sin_angle * |pos_ji| * |pos_jk|
        angle = torch.atan2(b, a)

        if not torsional:
            # return only distance+angle information
            return dist, angle, None, i, j, idx_kj, idx_ji
        else:
            # additionally compute torsional information
            idx_batch = torch.arange(len(idx_i), device=j.device)
            idx_k_n = adj_t[idx_j].storage.col()
            idx_nj_t = adj_t[idx_j].storage.value()
            repeat = num_triplets
            num_triplets_t = num_triplets.repeat_interleave(repeat)[mask]
            idx_i_t = idx_i.repeat_interleave(num_triplets_t)
            idx_j_t = idx_j.repeat_interleave(num_triplets_t)
            idx_k_t = idx_k.repeat_interleave(num_triplets_t)
            idx_kj_t = idx_kj.repeat_interleave(num_triplets_t)
            idx_ji_t = idx_ji.repeat_interleave(num_triplets_t)
            idx_batch_t = idx_batch.repeat_interleave(num_triplets_t)
            mask = idx_i_t != idx_k_n
            (
                idx_i_t,
                idx_j_t,
                idx_k_t,
                idx_k_n,
                idx_batch_t,
                idx_kj_t,
                idx_ji_t,
                idx_nj_t,
            ) = (
                idx_i_t[mask],
                idx_j_t[mask],
                idx_k_t[mask],
                idx_k_n[mask],
                idx_batch_t[mask],
                idx_kj_t[mask],
                idx_ji_t[mask],
                idx_nj_t[mask],
            )

            # Calculate torsions.
            pos_j0 = pos[idx_k_t] - pos[idx_j_t]
            pos_ji = pos[idx_i_t] - pos[idx_j_t]
            pos_jk = pos[idx_k_n] - pos[idx_j_t]
            if use_pbc:
                pos_j0 += offsets[idx_nj_t]
                pos_ji += offsets[idx_ji_t]
                pos_jk += offsets[idx_kj_t]
            dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
            plane1 = torch.cross(pos_ji, pos_j0)
            plane2 = torch.cross(pos_ji, pos_jk)
            a = (plane1 * plane2).sum(
                dim=-1
            )  # cos_angle * |plane1| * |plane2|
            b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
            torsion1 = torch.atan2(b, a)  # -pi to pi
            torsion1[torsion1 <= 0] += 2 * PI  # 0 to 2pi
            torsion = scatter(torsion1, idx_batch_t, reduce="min")

        return dist, angle, torsion, i, j, idx_kj, idx_ji


def real_sph_harm(lmax, zero_m_only=False, spherical_coordinates=True):
    """
    Computes formula strings of the the real part of the spherical harmonics up to order lmax (excluded).
    Variables are either cartesian coordinates x,y,z on the unit sphere or spherical coordinates phi and theta.
    """
    if not zero_m_only:
        x = sym.symbols("x")
        y = sym.symbols("y")
        S_m = [x * 0]
        C_m = [1 + 0 * x]
        # S_m = [0]
        # C_m = [1]
        for i in range(1, lmax):
            x = sym.symbols("x")
            y = sym.symbols("y")
            S_m += [x * S_m[i - 1] + y * C_m[i - 1]]
            C_m += [x * C_m[i - 1] - y * S_m[i - 1]]

    P_l_m = associated_legendre_polynomials(lmax, zero_m_only)
    if spherical_coordinates:
        theta = sym.symbols("theta")
        z = sym.symbols("z")
        for i in range(len(P_l_m)):
            for j in range(len(P_l_m[i])):
                if type(P_l_m[i][j]) != int:
                    P_l_m[i][j] = P_l_m[i][j].subs(z, sym.cos(theta))
        if not zero_m_only:
            phi = sym.symbols("phi")
            for i in range(len(S_m)):
                S_m[i] = (
                    S_m[i]
                    .subs(x, sym.sin(theta) * sym.cos(phi))
                    .subs(y, sym.sin(theta) * sym.sin(phi))
                )
            for i in range(len(C_m)):
                C_m[i] = (
                    C_m[i]
                    .subs(x, sym.sin(theta) * sym.cos(phi))
                    .subs(y, sym.sin(theta) * sym.sin(phi))
                )

    Y_func_l_m = [["0"] * (2 * j + 1) for j in range(lmax)]
    for i in range(lmax):
        Y_func_l_m[i][0] = sym.simplify(sph_harm_prefactor(i, 0) * P_l_m[i][0])

    if not zero_m_only:
        for i in range(1, lmax):
            for j in range(1, i + 1):
                Y_func_l_m[i][j] = sym.simplify(
                    2 ** 0.5 * sph_harm_prefactor(i, j) * C_m[j] * P_l_m[i][j]
                )
        for i in range(1, lmax):
            for j in range(1, i + 1):
                Y_func_l_m[i][-j] = sym.simplify(
                    2 ** 0.5 * sph_harm_prefactor(i, -j) * S_m[j] * P_l_m[i][j]
                )

    return Y_func_l_m


class Envelope(torch.nn.Module):
    def __init__(self, exponent):
        super(Envelope, self).__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x):
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return 1.0 / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2


class dist_emb(torch.nn.Module):
    def __init__(self, num_radial, cutoff=5.0, envelope_exponent=5):
        super(dist_emb, self).__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        self.freq = torch.nn.Parameter(torch.Tensor(num_radial))

        self.reset_parameters()

    def reset_parameters(self):
        torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(PI)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) / self.cutoff
        return self.envelope(dist) * (self.freq * dist).sin()


class angle_emb(torch.nn.Module):
    def __init__(
        self, num_spherical, num_radial, cutoff=5.0, envelope_exponent=5
    ):
        super(angle_emb, self).__init__()
        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        # self.envelope = Envelope(envelope_exponent)

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []

        x, theta = sym.symbols("x theta")
        modules = {"sin": torch.sin, "cos": torch.cos}
        for i in range(num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                self.sph_funcs.append(lambda x: torch.zeros_like(x) + sph1)
            else:
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)
            for j in range(num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def forward(self, dist, angle, idx_kj):
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        # rbf = self.envelope(dist).unsqueeze(-1) * rbf

        cbf = torch.stack([f(angle) for f in self.sph_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)
        return out


class torsion_emb(torch.nn.Module):
    def __init__(
        self, num_spherical, num_radial, cutoff=5.0, envelope_exponent=5
    ):
        super(torsion_emb, self).__init__()
        assert num_radial <= 64
        self.num_spherical = num_spherical  #
        self.num_radial = num_radial
        self.cutoff = cutoff
        # self.envelope = Envelope(envelope_exponent)

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical, zero_m_only=False)
        self.sph_funcs = []
        self.bessel_funcs = []

        x = sym.symbols("x")
        theta = sym.symbols("theta")
        phi = sym.symbols("phi")
        modules = {"sin": torch.sin, "cos": torch.cos}
        for i in range(self.num_spherical):
            if i == 0:
                sph1 = sym.lambdify(
                    [theta, phi], sph_harm_forms[i][0], modules
                )
                self.sph_funcs.append(
                    lambda x, y: torch.zeros_like(x)
                    + torch.zeros_like(y)
                    + sph1(0, 0)
                )  # torch.zeros_like(x) + torch.zeros_like(y)
            else:
                for k in range(-i, i + 1):
                    sph = sym.lambdify(
                        [theta, phi], sph_harm_forms[i][k + i], modules
                    )
                    self.sph_funcs.append(sph)
            for j in range(self.num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def forward(self, dist, angle, phi, idx_kj):
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        cbf = torch.stack([f(angle, phi) for f in self.sph_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].view(-1, 1, n, k) * cbf.view(-1, n, n, 1)).view(
            -1, n * n * k
        )
        return out
