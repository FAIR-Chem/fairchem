"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import sympy as sym
import torch
from torch_geometric.nn.models.schnet import GaussianSmearing

from ocpmodels.common.typing import assert_is_instance

from .basis_utils import real_sph_harm
from .radial_basis import RadialBasis


class CircularBasisLayer(torch.nn.Module):
    """
    2D Fourier Bessel Basis

    Parameters
    ----------
    num_spherical: int
        Controls maximum frequency.
    radial_basis: RadialBasis
        Radial basis functions
    cbf: dict
        Name and hyperparameters of the cosine basis function
    efficient: bool
        Whether to use the "efficient" summation order
    """

    def __init__(
        self,
        num_spherical: int,
        radial_basis: RadialBasis,
        cbf,
        efficient: bool = False,
    ) -> None:
        super().__init__()

        self.radial_basis = radial_basis
        self.efficient = efficient

        cbf_name = assert_is_instance(cbf["name"], str).lower()
        cbf_hparams = cbf.copy()
        del cbf_hparams["name"]

        if cbf_name == "gaussian":
            self.cosφ_basis = GaussianSmearing(
                start=-1, stop=1, num_gaussians=num_spherical, **cbf_hparams
            )
        elif cbf_name == "spherical_harmonics":
            Y_lm = real_sph_harm(
                num_spherical, use_theta=False, zero_m_only=True
            )
            sph_funcs = []  # (num_spherical,)

            # convert to tensorflow functions
            z = sym.symbols("z")
            modules = {"sin": torch.sin, "cos": torch.cos, "sqrt": torch.sqrt}
            m_order = 0  # only single angle
            for l_degree in range(len(Y_lm)):  # num_spherical
                if (
                    l_degree == 0
                ):  # Y_00 is only a constant -> function returns value and not tensor
                    first_sph = sym.lambdify(
                        [z], Y_lm[l_degree][m_order], modules
                    )
                    sph_funcs.append(
                        lambda z: torch.zeros_like(z) + first_sph(z)
                    )
                else:
                    sph_funcs.append(
                        sym.lambdify([z], Y_lm[l_degree][m_order], modules)
                    )
            self.cosφ_basis = lambda cosφ: torch.stack(
                [f(cosφ) for f in sph_funcs], dim=1
            )
        else:
            raise ValueError(f"Unknown cosine basis function '{cbf_name}'.")

    def forward(self, D_ca, cosφ_cab, id3_ca):
        rbf = self.radial_basis(D_ca)  # (num_edges, num_radial)
        cbf = self.cosφ_basis(cosφ_cab)  # (num_triplets, num_spherical)

        if not self.efficient:
            rbf = rbf[id3_ca]  # (num_triplets, num_radial)
            out = (rbf[:, None, :] * cbf[:, :, None]).view(
                -1, rbf.shape[-1] * cbf.shape[-1]
            )
            return (out,)
            # (num_triplets, num_radial * num_spherical)
        else:
            return (rbf[None, :, :], cbf)
            # (1, num_edges, num_radial), (num_edges, num_spherical)
