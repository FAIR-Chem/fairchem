"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
from torch.nn import Linear

from .radial import RadialMLP


class SO2_m_Conv(torch.nn.Module):
    """
    SO(2) Conv: Perform an SO(2) convolution on features corresponding to +- m

    Args:
        m (int):                    Order of the spherical harmonic coefficients
        sphere_channels (int):      Number of spherical channels
        m_output_channels (int):    Number of output channels used during the SO(2) conv
        lmax (int):                 degrees (l)
        mmax (int):                 orders (m)
    """

    def __init__(
        self,
        m: int,
        sphere_channels: int,
        m_output_channels: int,
        lmax: int,
        mmax: int,
    ) -> None:
        super().__init__()

        self.m = m
        self.sphere_channels = sphere_channels
        self.m_output_channels = m_output_channels
        self.lmax = lmax
        self.mmax = mmax

        assert self.mmax >= m
        num_coefficents = self.lmax - m + 1
        num_channels = num_coefficents * self.sphere_channels

        self.out_channels_half = self.m_output_channels * (
            num_channels // self.sphere_channels
        )
        self.fc = Linear(
            num_channels,
            2 * self.out_channels_half,
            bias=False,
        )
        self.fc.weight.data.mul_(1 / math.sqrt(2))

    def forward(self, x_m):
        x_m = self.fc(x_m)
        x_r = x_m.narrow(2, 0, self.out_channels_half)
        x_i = x_m.narrow(2, self.out_channels_half, self.out_channels_half)
        x_m_r = x_r.narrow(1, 0, 1) - x_i.narrow(1, 1, 1)  # x_r[:, 0] - x_i[:, 1]
        x_m_i = x_r.narrow(1, 1, 1) + x_i.narrow(1, 0, 1)  # x_r[:, 1] + x_i[:, 0]
        return torch.cat((x_m_r, x_m_i), dim=1)


class SO2_Convolution(torch.nn.Module):
    """
    SO(2) Block: Perform SO(2) convolutions for all m (orders)

    Args:
        sphere_channels (int):      Number of spherical channels
        m_output_channels (int):    Number of output channels used during the SO(2) conv
        lmax (int):                 degrees (l)
        mmax (int):                 orders (m)
        mappingReduced (CoefficientMappingModule): Used to extract a subset of m components
        internal_weights (bool):    If True, not using radial function to multiply inputs features
        edge_channels_list (list:int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
        extra_m0_output_channels (int): If not None, return `out_embedding` and `extra_m0_features` (Tensor).
    """

    def __init__(
        self,
        sphere_channels: int,
        m_output_channels: int,
        lmax: int,
        mmax: int,
        mappingReduced,
        internal_weights: bool = True,
        edge_channels_list: list[int] | None = None,
        extra_m0_output_channels: int | None = None,
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.m_output_channels = m_output_channels
        self.lmax = lmax
        self.mmax = mmax
        self.mappingReduced = mappingReduced
        self.internal_weights = internal_weights
        self.edge_channels_list = copy.deepcopy(edge_channels_list)
        self.extra_m0_output_channels = extra_m0_output_channels

        num_channels_m0 = (self.lmax + 1) * self.sphere_channels

        # SO(2) convolution for m = 0
        m0_output_channels = self.m_output_channels * (
            num_channels_m0 // self.sphere_channels
        )
        if self.extra_m0_output_channels is not None:
            m0_output_channels = m0_output_channels + self.extra_m0_output_channels
        self.fc_m0 = Linear(num_channels_m0, m0_output_channels)
        num_channels_rad = self.fc_m0.in_features

        # SO(2) convolution for non-zero m
        self.so2_m_conv = nn.ModuleList()
        for m in range(1, self.mmax + 1):
            self.so2_m_conv.append(
                SO2_m_Conv(
                    m,
                    self.sphere_channels,
                    self.m_output_channels,
                    self.lmax,
                    self.mmax,
                )
            )
            num_channels_rad = num_channels_rad + self.so2_m_conv[-1].fc.in_features

        # Embedding function of distance
        self.rad_func = None
        if not self.internal_weights:
            assert self.edge_channels_list is not None
            self.edge_channels_list.append(int(num_channels_rad))
            self.rad_func = RadialMLP(self.edge_channels_list)

    def forward(
        self,
        x: torch.Tensor,
        x_edge: torch.Tensor,
    ):
        num_edges = len(x_edge)
        out = []

        # Reshape the spherical harmonics based on m (order)
        x = torch.einsum("nac,ba->nbc", x, self.mappingReduced.to_m)

        # radial function
        if self.rad_func is not None:
            x_edge = self.rad_func(x_edge)
        offset_rad = 0

        # Compute m=0 coefficients separately since they only have real values (no imaginary)
        x_0 = x.narrow(1, 0, self.mappingReduced.m_size[0])
        x_0 = x_0.reshape(num_edges, -1)
        if self.rad_func is not None:
            x_edge_0 = x_edge.narrow(1, 0, self.fc_m0.in_features)
            x_0 = x_0 * x_edge_0
        x_0 = self.fc_m0(x_0)

        x_0_extra = None
        # extract extra m0 features
        if self.extra_m0_output_channels is not None:
            x_0_extra = x_0.narrow(-1, 0, self.extra_m0_output_channels)
            x_0 = x_0.narrow(
                -1,
                self.extra_m0_output_channels,
                (self.fc_m0.out_features - self.extra_m0_output_channels),
            )

        x_0 = x_0.view(num_edges, -1, self.m_output_channels)
        # x.embedding[:, 0 : self.mappingReduced.m_size[0]] = x_0
        out.append(x_0)
        offset_rad = offset_rad + self.fc_m0.in_features

        # Compute the values for the m > 0 coefficients
        offset = self.mappingReduced.m_size[0]
        for m in range(1, self.mmax + 1):
            # Get the m order coefficients
            x_m = x.narrow(1, offset, 2 * self.mappingReduced.m_size[m])
            x_m = x_m.reshape(num_edges, 2, -1)

            # Perform SO(2) convolution
            if self.rad_func is not None:
                x_edge_m = x_edge.narrow(
                    1, offset_rad, self.so2_m_conv[m - 1].fc.in_features
                )
                x_edge_m = x_edge_m.reshape(
                    num_edges, 1, self.so2_m_conv[m - 1].fc.in_features
                )
                x_m = x_m * x_edge_m
            x_m = self.so2_m_conv[m - 1](x_m)
            x_m = x_m.view(num_edges, -1, self.m_output_channels)
            # x.embedding[:, offset : offset + 2 * self.mappingReduced.m_size[m]] = x_m
            out.append(x_m)
            offset = offset + 2 * self.mappingReduced.m_size[m]
            offset_rad = offset_rad + self.so2_m_conv[m - 1].fc.in_features

        out = torch.cat(out, dim=1)
        # Reshape the spherical harmonics based on l (degree)
        out = torch.einsum("nac,ab->nbc", out, self.mappingReduced.to_m)

        if self.extra_m0_output_channels is not None:
            return out, x_0_extra
        else:
            return out


class SO2_Linear(torch.nn.Module):
    """
    SO(2) Linear: Perform SO(2) linear for all m (orders).

    Args:
        sphere_channels (int):      Number of spherical channels
        m_output_channels (int):    Number of output channels used during the SO(2) conv
        lmax (int):                 degrees (l)
        mmax (int):                 orders (m)
        mappingReduced (CoefficientMappingModule): Used to extract a subset of m components
        internal_weights (bool):    If True, not using radial function to multiply inputs features
        edge_channels_list (list:int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
    """

    def __init__(
        self,
        sphere_channels: int,
        m_output_channels: int,
        lmax: int,
        mmax: int,
        mappingReduced,
        internal_weights: bool = False,
        edge_channels_list: list[int] | None = None,
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.m_output_channels = m_output_channels
        self.lmax = lmax
        self.mmax = mmax
        self.mappingReduced = mappingReduced
        self.internal_weights = internal_weights
        self.edge_channels_list = copy.deepcopy(edge_channels_list)

        num_channels_m0 = (self.lmax + 1) * self.sphere_channels

        # SO(2) linear for m = 0
        self.fc_m0 = Linear(
            num_channels_m0,
            self.m_output_channels * (num_channels_m0 // self.sphere_channels),
        )
        num_channels_rad = self.fc_m0.in_features

        # SO(2) linear for non-zero m
        self.so2_m_fc = nn.ModuleList()
        for m in range(1, self.mmax + 1):
            num_coefficents = self.lmax - m + 1
            num_in_channels = num_coefficents * self.sphere_channels
            fc = Linear(
                num_in_channels,
                self.m_output_channels * (num_in_channels // self.sphere_channels),
                bias=False,
            )
            num_channels_rad = num_channels_rad + fc.in_features
            self.so2_m_fc.append(fc)

        # Embedding function of distance
        self.rad_func = None
        if not self.internal_weights:
            assert self.edge_channels_list is not None
            self.edge_channels_list.append(int(num_channels_rad))
            self.rad_func = RadialMLP(self.edge_channels_list)

    def forward(self, x, x_edge):
        batch_size = x.shape[0]
        out = []

        # Reshape the spherical harmonics based on m (order)
        x = torch.einsum("nac,ba->nbc", x, self.mappingReduced.to_m)

        # radial function
        if self.rad_func is not None:
            x_edge = self.rad_func(x_edge)
        offset_rad = 0

        # Compute m=0 coefficients separately since they only have real values (no imaginary)
        x_0 = x.narrow(1, 0, self.mappingReduced.m_size[0])
        x_0 = x_0.reshape(batch_size, -1)
        if self.rad_func is not None:
            x_edge_0 = x_edge.narrow(1, 0, self.fc_m0.in_features)
            x_0 = x_0 * x_edge_0
        x_0 = self.fc_m0(x_0)
        x_0 = x_0.view(batch_size, -1, self.m_output_channels)
        out.append(x_0)
        offset_rad = offset_rad + self.fc_m0.in_features

        # Compute the values for the m > 0 coefficients
        offset = self.mappingReduced.m_size[0]
        for m in range(1, self.mmax + 1):
            # Get the m order coefficients
            x_m = x.narrow(1, offset, 2 * self.mappingReduced.m_size[m])
            x_m = x_m.reshape(batch_size, 2, -1)
            if self.rad_func is not None:
                x_edge_m = x_edge.narrow(
                    1, offset_rad, self.so2_m_fc[m - 1].in_features
                )
                x_edge_m = x_edge_m.reshape(
                    batch_size, 1, self.so2_m_fc[m - 1].in_features
                )
                x_m = x_m * x_edge_m

            # Perform SO(2) linear
            x_m = self.so2_m_fc[m - 1](x_m)
            x_m = x_m.view(batch_size, -1, self.m_output_channels)
            out.append(x_m)

            offset = offset + 2 * self.mappingReduced.m_size[m]
            offset_rad = offset_rad + self.so2_m_fc[m - 1].in_features

        out = torch.cat(out, dim=1)

        # Reshape the spherical harmonics based on l (degree)
        return torch.einsum("nac,ab->nbc", out, self.mappingReduced.to_m)
