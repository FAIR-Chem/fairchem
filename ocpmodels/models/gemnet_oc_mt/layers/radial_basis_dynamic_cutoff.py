"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
from logging import getLogger
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from scipy.special import binom
from torch_geometric.data import Batch

from ocpmodels.modules.scaling import ScaleFactor

from ..config import BackboneConfig

log = getLogger(__name__)


class PolynomialEnvelope(nn.Module):
    """
    Polynomial envelope function that ensures a smooth cutoff.

    Arguments
    ---------
        exponent: int
            Exponent of the envelope function.
    """

    def __init__(self, exponent):
        super().__init__()
        assert exponent > 0
        self.p = exponent
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, d_scaled):
        env_val = (
            1
            + self.a * d_scaled**self.p
            + self.b * d_scaled ** (self.p + 1)
            + self.c * d_scaled ** (self.p + 2)
        )
        return torch.where(d_scaled < 1, env_val, torch.zeros_like(d_scaled))


class ExponentialEnvelope(nn.Module):
    """
    Exponential envelope function that ensures a smooth cutoff,
    as proposed in Unke, Chmiela, Gastegger, Schütt, Sauceda, Müller 2021.
    SpookyNet: Learning Force Fields with Electronic Degrees of Freedom
    and Nonlocal Effects
    """

    def __init__(self):
        super().__init__()

    def forward(self, d_scaled):
        env_val = torch.exp(
            -(d_scaled**2) / ((1 - d_scaled) * (1 + d_scaled))
        )
        return torch.where(d_scaled < 1, env_val, torch.zeros_like(d_scaled))


class GaussianBasisOld(nn.Module):
    def __init__(
        self, start=0.0, stop=5.0, num_gaussians=50, trainable=False, **kwargs
    ):
        super().__init__()

        self.trainable = trainable
        offset = torch.linspace(start, stop, num_gaussians)
        if self.trainable:
            _ = nn.init.uniform_(offset, start, stop)
            self.offset = nn.Parameter(offset, requires_grad=True)
        else:
            self.register_buffer("offset", offset)
        self.coeff = -0.5 / ((stop - start) / (num_gaussians - 1)) ** 2

    def forward(self, dist):
        dist = dist[:, None] - self.offset[None, :]
        return torch.exp(self.coeff * torch.pow(dist, 2))


class GaussianBasis(nn.Module):
    def __init__(
        self,
        start=0.0,
        stop=5.0,
        num_gaussians=50,
        trainable=False,
        trainable_stds=False,
    ):
        super().__init__()

        self.trainable = trainable
        self.trainable_stds = trainable_stds

        offset = torch.linspace(start, stop, num_gaussians)
        if self.trainable:
            _ = nn.init.uniform_(offset, start, stop)
            self.offset = nn.Parameter(offset, requires_grad=True)
        else:
            self.register_buffer("offset", offset)

        if self.trainable_stds:
            self.temps = nn.Parameter(
                torch.empty(num_gaussians), requires_grad=True
            )
            temp_mean = -0.5 / ((stop - start) / (num_gaussians - 1)) ** 2
            _ = nn.init.normal_(
                self.temps, mean=temp_mean, std=abs(temp_mean) * 2.0
            )
        else:
            temp = 0.5 / ((stop - start) / (num_gaussians - 1)) ** 2
            self.register_buffer("temps", torch.full_like(offset, temp))

    def forward(self, dist):
        dist = dist[:, None] - self.offset[None, :]
        return torch.exp(dist.square() * -(self.temps.abs()))


class SphericalBesselBasis(nn.Module):
    """
    First-order spherical Bessel basis

    Arguments
    ---------
    num_radial: int
        Number of basis functions. Controls the maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    """

    def __init__(
        self,
        num_radial: int,
        cutoff: float,
    ):
        super().__init__()
        self.norm_const = math.sqrt(2 / (cutoff**3))
        # cutoff ** 3 to counteract dividing by d_scaled = d / cutoff

        # Initialize frequencies at canonical positions
        self.frequencies = nn.Parameter(
            data=torch.tensor(
                np.pi * np.arange(1, num_radial + 1, dtype=np.float32)
            ),
            requires_grad=True,
        )

    def forward(self, d_scaled):
        return (
            self.norm_const
            / d_scaled[:, None]
            * torch.sin(self.frequencies * d_scaled[:, None])
        )  # (num_edges, num_radial)


class BernsteinBasis(nn.Module):
    """
    Bernstein polynomial basis,
    as proposed in Unke, Chmiela, Gastegger, Schütt, Sauceda, Müller 2021.
    SpookyNet: Learning Force Fields with Electronic Degrees of Freedom
    and Nonlocal Effects

    Arguments
    ---------
    num_radial: int
        Number of basis functions. Controls the maximum frequency.
    pregamma_initial: float
        Initial value of exponential coefficient gamma.
        Default: gamma = 0.5 * a_0**-1 = 0.94486,
        inverse softplus -> pregamma = log e**gamma - 1 = 0.45264
    """

    def __init__(
        self,
        num_radial: int,
        pregamma_initial: float = 0.45264,
    ):
        super().__init__()
        prefactor = binom(num_radial - 1, np.arange(num_radial))
        self.register_buffer(
            "prefactor",
            torch.tensor(prefactor, dtype=torch.float),
            persistent=False,
        )

        self.pregamma = nn.Parameter(
            data=torch.tensor(pregamma_initial, dtype=torch.float),
            requires_grad=True,
        )
        self.softplus = nn.Softplus()

        exp1 = torch.arange(num_radial)
        self.register_buffer("exp1", exp1[None, :], persistent=False)
        exp2 = num_radial - 1 - exp1
        self.register_buffer("exp2", exp2[None, :], persistent=False)

    def forward(self, d_scaled):
        gamma = self.softplus(self.pregamma)  # constrain to positive
        exp_d = torch.exp(-gamma * d_scaled)[:, None]
        return (
            self.prefactor * (exp_d**self.exp1) * ((1 - exp_d) ** self.exp2)
        )


class RadialBasis(nn.Module):
    """

    Arguments
    ---------
    num_radial: int
        Number of basis functions. Controls the maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    rbf: dict = {"name": "gaussian"}
        Basis function and its hyperparameters.
    envelope: dict = {"name": "polynomial", "exponent": 5}
        Envelope function and its hyperparameters.
    scale_basis: bool
        Whether to scale the basis values for better numerical stability.
    """

    def __init__(
        self,
        num_radial: int,
        graph_type: str,
        rbf: dict = {"name": "gaussian"},
        envelope: dict = {"name": "polynomial", "exponent": 5},
        scale_basis: bool = False,
        *,
        absolute_cutoff: float | None,
    ):
        super().__init__()

        self.absolute_cutoff = absolute_cutoff
        if self.absolute_cutoff:
            log.info(
                f"[{self.__class__.__qualname__}] Using absolute cutoff of {self.absolute_cutoff} Angstroms."
            )
        else:
            log.info(f"[{self.__class__.__qualname__}] Using relative cutoff.")

        # self.inv_cutoff = 1 / cutoff
        self.graph_type = graph_type

        self.scale_basis = scale_basis
        if self.scale_basis:
            self.scale_rbf = ScaleFactor(num_radial)

        env_name = envelope["name"].lower()
        env_hparams = envelope.copy()
        del env_hparams["name"]

        if env_name == "polynomial":
            self.envelope = PolynomialEnvelope(**env_hparams)
        elif env_name == "exponential":
            self.envelope = ExponentialEnvelope(**env_hparams)
        else:
            raise ValueError(f"Unknown envelope function '{env_name}'.")

        rbf_name = rbf["name"].lower()
        rbf_hparams = rbf.copy()
        del rbf_hparams["name"]

        # RBFs get distances scaled to be in [0, 1]
        if rbf_name == "gaussian":
            cls = (
                GaussianBasisOld
                if BackboneConfig.instance().old_gaussian_implementation
                else GaussianBasis
            )
            self.rbf = cls(
                start=0, stop=1, num_gaussians=num_radial, **rbf_hparams
            )
        elif rbf_name == "spherical_bessel":
            assert (
                absolute_cutoff is not None
            ), "Spherical Bessel basis requires absolute cutoff."
            self.rbf = SphericalBesselBasis(
                num_radial=num_radial, cutoff=absolute_cutoff, **rbf_hparams
            )
        elif rbf_name == "bernstein":
            self.rbf = BernsteinBasis(num_radial=num_radial, **rbf_hparams)
        else:
            raise ValueError(f"Unknown radial basis function '{rbf_name}'.")

    @staticmethod
    def _get_tensor(data: Batch, key: str):
        value = getattr(data, key, None)
        if value is None:
            raise ValueError(f"Batch does not contain key '{key}'.")
        if not torch.is_tensor(value):
            raise ValueError(f"Key '{key}' must be a tensor.")

        value = cast(torch.Tensor, value)
        return value

    def forward(self, d, *, data: Batch):
        if not (cutoff := self.absolute_cutoff):
            edge_index = self._get_tensor(
                data, f"{self.graph_type}_edge_index"
            )
            cutoff = self._get_tensor(
                data, f"{self.graph_type}_cutoff"
            )  # (b,)
            cutoff = cutoff[data.batch]  # (n_nodes,)
            cutoff = cutoff[edge_index[0]]  # (n_edges,)

        d_scaled = d / cutoff

        env = self.envelope(d_scaled)
        res = env[:, None] * self.rbf(d_scaled)

        if self.scale_basis:
            res = self.scale_rbf(res)

        return res
        # (num_edges, num_radial) or (num_edges, num_orders * num_radial)
