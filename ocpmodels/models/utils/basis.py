"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
from typing import List, Optional, Union

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from scipy.special import sph_harm

from .activations import Act


class Sine(nn.Module):
    def __init__(self, w0: float = 30.0) -> None:
        super(Sine, self).__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class SIREN(nn.Module):
    def __init__(
        self,
        layers: List[int],
        num_in_features: int,
        out_features: int,
        w0: float = 30.0,
        initializer: Optional[str] = "siren",
        c: float = 6,
    ) -> None:
        super(SIREN, self).__init__()
        self.layers = [nn.Linear(num_in_features, layers[0]), Sine(w0=w0)]

        for index in range(len(layers) - 1):
            self.layers.extend(
                [nn.Linear(layers[index], layers[index + 1]), Sine(w0=1)]
            )

        self.layers.append(nn.Linear(layers[-1], out_features))
        self.network = nn.Sequential(*self.layers)

        if initializer is not None and initializer == "siren":
            for m in self.network:
                if isinstance(m, nn.Linear):
                    num_input = float(m.weight.size(-1))
                    with torch.no_grad():
                        m.weight.uniform_(
                            -math.sqrt(6.0 / num_input),
                            math.sqrt(6.0 / num_input),
                        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.network(X)


class SINESmearing(nn.Module):
    def __init__(
        self,
        num_in_features: int,
        num_freqs: int = 40,
        use_cosine: bool = False,
    ) -> None:
        super(SINESmearing, self).__init__()

        self.num_freqs = num_freqs
        self.out_dim: int = num_in_features * self.num_freqs
        self.use_cosine = use_cosine

        freq = torch.arange(num_freqs).float()
        freq = torch.pow(torch.ones_like(freq) * 1.1, freq)
        self.freq_filter = nn.Parameter(
            freq.view(-1, 1).repeat(1, num_in_features).view(1, -1),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat(1, self.num_freqs)
        x = x * self.freq_filter

        if self.use_cosine:
            return torch.cos(x)
        else:
            return torch.sin(x)


class GaussianSmearing(nn.Module):
    def __init__(
        self,
        num_in_features: int,
        start: int = 0,
        end: int = 1,
        num_freqs: int = 50,
    ) -> None:
        super(GaussianSmearing, self).__init__()
        self.num_freqs = num_freqs
        offset = torch.linspace(start, end, num_freqs)
        self.coeff: float = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.offset = nn.Parameter(
            offset.view(-1, 1).repeat(1, num_in_features).view(1, -1),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat(1, self.num_freqs)
        x = x - self.offset
        return torch.exp(self.coeff * torch.pow(x, 2))


class FourierSmearing(nn.Module):
    def __init__(
        self,
        num_in_features: int,
        num_freqs: int = 40,
        use_cosine: bool = False,
    ) -> None:
        super(FourierSmearing, self).__init__()

        self.num_freqs = num_freqs
        self.out_dim: int = num_in_features * self.num_freqs
        self.use_cosine = use_cosine

        freq = torch.arange(num_freqs).to(torch.float32)
        self.freq_filter = nn.Parameter(
            freq.view(-1, 1).repeat(1, num_in_features).view(1, -1),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat(1, self.num_freqs)
        x = x * self.freq_filter

        if self.use_cosine:
            return torch.cos(x)
        else:
            return torch.sin(x)


class Basis(nn.Module):
    smearing: Union[
        SINESmearing,
        SINESmearing,
        FourierSmearing,
        GaussianSmearing,
        torch.nn.Sequential,
    ]

    def __init__(
        self,
        num_in_features: int,
        num_freqs: int = 50,
        basis_type: str = "powersine",
        act: str = "ssp",
        sph: Optional["SphericalSmearing"] = None,
    ) -> None:
        super(Basis, self).__init__()

        self.num_freqs = num_freqs
        self.basis_type = basis_type

        if basis_type == "powersine":
            self.smearing = SINESmearing(num_in_features, num_freqs)
            self.out_dim = num_in_features * num_freqs
        elif basis_type == "powercosine":
            self.smearing = SINESmearing(
                num_in_features, num_freqs, use_cosine=True
            )
            self.out_dim = num_in_features * num_freqs
        elif basis_type == "fouriersine":
            self.smearing = FourierSmearing(num_in_features, num_freqs)
            self.out_dim = num_in_features * num_freqs
        elif basis_type == "gauss":
            self.smearing = GaussianSmearing(
                num_in_features, start=0, end=1, num_freqs=num_freqs
            )
            self.out_dim = num_in_features * num_freqs
        elif basis_type == "linact":
            self.smearing = torch.nn.Sequential(
                torch.nn.Linear(num_in_features, num_freqs * num_in_features),
                Act(act),
            )
            self.out_dim = num_in_features * num_freqs
        elif basis_type == "raw" or basis_type == "rawcat":
            self.out_dim = num_in_features
        elif "sph" in basis_type:
            # by default, we use sine function to encode distance
            # sph must be given here
            assert sph is not None
            # assumes the first three columns are normalizaed xyz
            # the rest of the columns are distances
            if "cat" in basis_type:
                # concatenate
                self.smearing_sine = SINESmearing(
                    num_in_features - 3, num_freqs
                )
                self.out_dim = sph.out_dim + (num_in_features - 3) * num_freqs
            elif "mul" in basis_type:
                self.smearing_sine = SINESmearing(
                    num_in_features - 3, num_freqs
                )
                self.lin = torch.nn.Linear(
                    self.smearing_sine.out_dim, num_in_features - 3
                )
                self.out_dim = (num_in_features - 3) * sph.out_dim
            elif "m40" in basis_type:
                dim = 40
                self.smearing_sine = SINESmearing(
                    num_in_features - 3, num_freqs
                )
                self.lin = torch.nn.Linear(
                    self.smearing_sine.out_dim, dim
                )  # make the output dimensionality comparable.
                self.out_dim = dim * sph.out_dim
            elif "nosine" in basis_type:
                # does not use sine smearing for encoding distance
                self.out_dim = (num_in_features - 3) * sph.out_dim
            else:
                raise ValueError(
                    "cat or mul not specified for spherical harnomics."
                )
        else:
            raise RuntimeError("Undefined basis type.")

    def forward(
        self, x: torch.Tensor, edge_attr_sph: Optional[torch.Tensor] = None
    ):
        if "sph" in self.basis_type:
            if "nosine" not in self.basis_type:
                x_sine = self.smearing_sine(
                    x[:, 3:]
                )  # the first three features correspond to edge_vec_normalized, so we ignore
                if "cat" in self.basis_type:
                    # just concatenate spherical edge feature and sined node features
                    assert isinstance(edge_attr_sph, torch.Tensor)
                    return torch.cat([edge_attr_sph, x_sine], dim=1)
                elif "mul" in self.basis_type or "m40" in self.basis_type:
                    # multiply sined node features into spherical edge feature (inspired by theory in spherical harmonics)
                    r = self.lin(x_sine)
                    outer = torch.einsum("ik,ij->ikj", edge_attr_sph, r)
                    return torch.flatten(outer, start_dim=1)
                else:
                    raise RuntimeError(
                        f"Unknown basis type called {self.basis_type}"
                    )
            else:
                outer = torch.einsum("ik,ij->ikj", edge_attr_sph, x[:, 3:])
                return torch.flatten(outer, start_dim=1)

        elif "raw" in self.basis_type:
            # do nothing, just return node features
            pass
        else:
            x = self.smearing(x)
        return x


class SphericalSmearing(nn.Module):
    m: npt.NDArray[np.int_]
    n: npt.NDArray[np.int_]

    def __init__(self, max_n: int = 10, option: str = "all") -> None:
        super(SphericalSmearing, self).__init__()

        self.max_n = max_n

        m_list: List[int] = []
        n_list: List[int] = []
        for i in range(max_n):
            for j in range(0, i + 1):
                m_list.append(j)
                n_list.append(i)

        m = np.array(m_list)
        n = np.array(n_list)

        if option == "all":
            self.m = m
            self.n = n
        elif option == "sine":
            self.m = m[n % 2 == 1]
            self.n = n[n % 2 == 1]
        elif option == "cosine":
            self.m = m[n % 2 == 0]
            self.n = n[n % 2 == 0]

        self.out_dim = int(np.sum(self.m == 0) + 2 * np.sum(self.m != 0))

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        # assuming input is already normalized
        assert xyz.size(1) == 3

        xyz = xyz / xyz.norm(dim=-1).view(-1, 1)

        phi = torch.acos(xyz[:, 2])
        theta = torch.atan2(-xyz[:, 1], -xyz[:, 0]) + math.pi

        phi = phi.cpu().numpy()
        theta = theta.cpu().numpy()

        m_tile = np.tile(self.m, (len(xyz), 1))
        n_tile = np.tile(self.n, (len(xyz), 1))
        theta_tile = np.tile(theta.reshape(len(xyz), 1), (1, len(self.m)))
        phi_tile = np.tile(phi.reshape(len(xyz), 1), (1, len(self.m)))

        harm = sph_harm(m_tile, n_tile, theta_tile, phi_tile)

        harm_mzero = harm[:, self.m == 0]
        harm_mnonzero = harm[:, self.m != 0]

        harm_real = np.concatenate(
            [harm_mzero.real, harm_mnonzero.real, harm_mnonzero.imag], axis=1
        )

        return torch.from_numpy(harm_real).to(torch.float32).to(xyz.device)
