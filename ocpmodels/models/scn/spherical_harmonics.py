"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import math
import os

import torch

try:
    from e3nn import o3
    from e3nn.o3 import FromS2Grid, ToS2Grid

    # Borrowed from e3nn @ 0.4.0:
    # https://github.com/e3nn/e3nn/blob/0.4.0/e3nn/o3/_wigner.py#L10
    # _Jd is a list of tensors of shape (2l+1, 2l+1)
    _Jd = torch.load(os.path.join(os.path.dirname(__file__), "Jd.pt"))
except (ImportError, FileNotFoundError):
    logging.error(
        "Invalid setup for SCN. Either the e3nn library or Jd.pt is missing."
    )


class SphericalHarmonicsHelper:
    """
    Helper functions for spherical harmonics calculations and representations

    Args:
        lmax (int):             Maximum degree of the spherical harmonics
        mmax (int):             Maximum order of the spherical harmonics
        num_taps (int):         Number of taps or rotations (1 or otherwise set automatically based on mmax)
        num_bands (int):        Number of bands used during message aggregation for the 1x1 pointwise convolution (1 or 2)
    """

    def __init__(
        self,
        lmax: int,
        mmax: int,
        num_taps: int,
        num_bands: int,
    ) -> None:
        import sys

        if "e3nn" not in sys.modules:
            logging.error(
                "You need to install the e3nn library to use Spherical Harmonics"
            )
            raise ImportError

        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.num_taps = num_taps
        self.num_bands = num_bands

        # Make sure lmax is large enough to support the num_bands
        assert self.lmax - (self.num_bands - 1) >= 0

        self.sphere_basis = (self.lmax + 1) ** 2
        self.sphere_basis = int(self.sphere_basis)

        self.sphere_basis_reduce = self.lmax + 1
        for i in range(1, self.mmax + 1):
            self.sphere_basis_reduce = self.sphere_basis_reduce + 2 * (
                self.lmax + 1 - i
            )
        self.sphere_basis_reduce = int(self.sphere_basis_reduce)

    def InitWignerDMatrix(self, edge_rot_mat) -> None:
        self.device = edge_rot_mat.device

        # Initialize matrix to combine the y-axis rotations during message passing
        self.mapping_y_rot, self.y_rotations = self.InitYRotMapping()
        self.num_y_rotations = len(self.y_rotations)

        # Conversion from basis to grid respresentations
        self.grid_res = (self.lmax + 1) * 2
        self.to_grid_shb = torch.tensor([], device=self.device)
        self.to_grid_sha = torch.tensor([], device=self.device)

        for b in range(self.num_bands):
            l = self.lmax - b  # noqa: E741
            togrid = ToS2Grid(
                l,
                (self.grid_res, self.grid_res + 1),
                normalization="integral",
                device=self.device,
            )
            shb = togrid.shb
            sha = togrid.sha

            padding = torch.zeros(
                shb.size()[0],
                shb.size()[1],
                self.sphere_basis - shb.size()[2],
                device=self.device,
            )
            shb = torch.cat([shb, padding], dim=2)
            self.to_grid_shb = torch.cat([self.to_grid_shb, shb], dim=0)
            if b == 0:
                self.to_grid_sha = sha
            else:
                self.to_grid_sha = torch.block_diag(self.to_grid_sha, sha)

        self.to_grid_sha = self.to_grid_sha.view(
            self.num_bands, self.grid_res + 1, -1
        )
        self.to_grid_sha = torch.transpose(self.to_grid_sha, 0, 1).contiguous()
        self.to_grid_sha = self.to_grid_sha.view(
            (self.grid_res + 1) * self.num_bands, -1
        )

        self.to_grid_shb = self.to_grid_shb.detach()
        self.to_grid_sha = self.to_grid_sha.detach()

        self.from_grid = FromS2Grid(
            (self.grid_res, self.grid_res + 1),
            self.lmax,
            normalization="integral",
            device=self.device,
        )
        for p in self.from_grid.parameters():
            p.detach()

        # Compute subsets of Wigner matrices to use for messages
        wigner = torch.tensor([], device=self.device)
        wigner_inv = torch.tensor([], device=self.device)

        for y_rot in self.y_rotations:
            # Compute rotation about y-axis
            y_rot_mat = self.RotationMatrix(0, y_rot, 0)
            y_rot_mat = y_rot_mat.repeat(len(edge_rot_mat), 1, 1)
            # Add additional rotation about y-axis
            rot_mat = torch.bmm(y_rot_mat, edge_rot_mat)

            # Compute Wigner matrices corresponding to the 3x3 rotation matrices
            wignerD = self.RotationToWignerDMatrix(rot_mat, 0, self.lmax)

            basis_in = torch.tensor([], device=self.device)
            basis_out = torch.tensor([], device=self.device)
            start_l = 0
            end_l = self.lmax + 1
            for l in range(start_l, end_l):  # noqa: E741
                offset = l**2
                basis_in = torch.cat(
                    [
                        basis_in,
                        torch.arange(2 * l + 1, device=self.device) + offset,
                    ],
                    dim=0,
                )
                m_max = min(l, self.mmax)
                basis_out = torch.cat(
                    [
                        basis_out,
                        torch.arange(-m_max, m_max + 1, device=self.device)
                        + offset
                        + l,
                    ],
                    dim=0,
                )

            # Only keep the rows/columns of the matrices used given lmax and mmax
            wignerD_reduce = wignerD[:, basis_out.long(), :]
            wignerD_reduce = wignerD_reduce[:, :, basis_in.long()]

            if y_rot == 0.0:
                wigner_inv = (
                    torch.transpose(wignerD_reduce, 1, 2).contiguous().detach()
                )

            wigner = torch.cat([wigner, wignerD_reduce.unsqueeze(1)], dim=1)

        wigner = wigner.view(-1, self.sphere_basis_reduce, self.sphere_basis)

        self.wigner = wigner.detach()
        self.wigner_inv = wigner_inv.detach()

    # If num_taps is greater than 1, calculate how to combine the different samples.
    # Note the e3nn code flips the y-axis with the z-axis in the SCN paper description.
    def InitYRotMapping(self):
        if self.mmax == 0:
            y_rotations = torch.tensor([0.0], device=self.device)
            num_y_rotations = 1
            mapping_y_rot = torch.eye(
                self.sphere_basis_reduce, device=self.device
            )

        if self.mmax == 1:
            if self.num_taps == 1:
                y_rotations = torch.tensor([0.0], device=self.device)
                num_y_rotations = len(y_rotations)
                mapping_y_rot = torch.eye(
                    len(y_rotations) * self.sphere_basis_reduce,
                    self.sphere_basis_reduce,
                    device=self.device,
                )
            else:
                y_rotations = torch.tensor(
                    [0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi],
                    device=self.device,
                )
                num_y_rotations = len(y_rotations)
                mapping_y_rot = torch.zeros(
                    len(y_rotations) * self.sphere_basis_reduce,
                    self.sphere_basis_reduce,
                    device=self.device,
                )

                # m = 0
                for l in range(0, self.lmax + 1):  # noqa: E741
                    offset = (l - 1) * 3 + 2
                    if l == 0:  # noqa: E741
                        offset = 0
                    for y in range(num_y_rotations):
                        mapping_y_rot[
                            offset + y * self.sphere_basis_reduce, offset
                        ] = (1.0 / num_y_rotations)

                # m = -1
                for l in range(1, self.lmax + 1):  # noqa: E741
                    offset = (l - 1) * 3 + 1
                    for y in range(num_y_rotations):
                        mapping_y_rot[
                            offset + y * self.sphere_basis_reduce, offset
                        ] = (math.cos(y_rotations[y]) / num_y_rotations)
                        mapping_y_rot[
                            (offset + 2) + y * self.sphere_basis_reduce, offset
                        ] = (math.sin(y_rotations[y]) / num_y_rotations)

                # m = 1
                for l in range(1, self.lmax + 1):  # noqa: E741
                    offset = (l - 1) * 3 + 3
                    for y in range(num_y_rotations):
                        mapping_y_rot[
                            offset + y * self.sphere_basis_reduce, offset
                        ] = (math.cos(y_rotations[y]) / num_y_rotations)
                        mapping_y_rot[
                            offset - 2 + y * self.sphere_basis_reduce, offset
                        ] = (-math.sin(y_rotations[y]) / num_y_rotations)

        return mapping_y_rot.detach(), y_rotations

    # Simplified version of function from e3nn
    def ToGrid(self, x, channels) -> torch.Tensor:
        x = x.view(-1, self.sphere_basis, channels)
        x_grid = torch.einsum("mbi,zic->zbmc", self.to_grid_shb, x)
        x_grid = torch.einsum(
            "am,zbmc->zbac", self.to_grid_sha, x_grid
        ).contiguous()
        x_grid = x_grid.view(-1, self.num_bands * channels)
        return x_grid

    # Simplified version of function from e3nn
    def FromGrid(self, x_grid, channels) -> torch.Tensor:
        x_grid = x_grid.view(-1, self.grid_res, (self.grid_res + 1), channels)
        x = torch.einsum("am,zbac->zbmc", self.from_grid.sha, x_grid)
        x = torch.einsum("mbi,zbmc->zic", self.from_grid.shb, x).contiguous()
        x = x.view(-1, channels)
        return x

    def CombineYRotations(self, x) -> torch.Tensor:
        num_channels = x.size()[-1]
        x = x.view(
            -1, self.num_y_rotations * self.sphere_basis_reduce, num_channels
        )
        x = torch.einsum("abc, bd->adc", x, self.mapping_y_rot).contiguous()
        return x

    def Rotate(self, x) -> torch.Tensor:
        num_channels = x.size()[2]
        x = x.view(-1, 1, self.sphere_basis, num_channels).repeat(
            1, self.num_y_rotations, 1, 1
        )
        x = x.view(-1, self.sphere_basis, num_channels)
        # print('{} {}'.format(self.wigner.size(), x.size()))
        x_rot = torch.bmm(self.wigner, x)
        x_rot = x_rot.view(-1, self.sphere_basis_reduce * num_channels)
        return x_rot

    def FlipGrid(self, grid, num_channels: int) -> torch.Tensor:
        # lat long
        long_res = self.grid_res
        grid = grid.view(-1, self.grid_res, self.grid_res, num_channels)
        grid = torch.roll(grid, int(long_res // 2), 2)
        flip_grid = torch.flip(grid, [1])
        return flip_grid.view(-1, num_channels)

    def RotateInv(self, x) -> torch.Tensor:
        x_rot = torch.bmm(self.wigner_inv, x)
        return x_rot

    def RotateWigner(self, x, wigner) -> torch.Tensor:
        x_rot = torch.bmm(wigner, x)
        return x_rot

    def RotationMatrix(
        self, rot_x: float, rot_y: float, rot_z: float
    ) -> torch.Tensor:
        m1, m2, m3 = (
            torch.eye(3, device=self.device),
            torch.eye(3, device=self.device),
            torch.eye(3, device=self.device),
        )
        if rot_x:
            degree = rot_x
            sin, cos = math.sin(degree), math.cos(degree)
            m1 = torch.tensor(
                [[1, 0, 0], [0, cos, sin], [0, -sin, cos]], device=self.device
            )
        if rot_y:
            degree = rot_y
            sin, cos = math.sin(degree), math.cos(degree)
            m2 = torch.tensor(
                [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]], device=self.device
            )
        if rot_z:
            degree = rot_z
            sin, cos = math.sin(degree), math.cos(degree)
            m3 = torch.tensor(
                [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]], device=self.device
            )

        matrix = torch.mm(torch.mm(m1, m2), m3)
        matrix = matrix.view(1, 3, 3)

        return matrix

    def RotationToWignerDMatrix(self, edge_rot_mat, start_lmax, end_lmax):
        x = edge_rot_mat @ edge_rot_mat.new_tensor([0.0, 1.0, 0.0])
        alpha, beta = o3.xyz_to_angles(x)
        R = (
            o3.angles_to_matrix(
                alpha, beta, torch.zeros_like(alpha)
            ).transpose(-1, -2)
            @ edge_rot_mat
        )
        gamma = torch.atan2(R[..., 0, 2], R[..., 0, 0])

        size = (end_lmax + 1) ** 2 - (start_lmax) ** 2
        wigner = torch.zeros(len(alpha), size, size, device=self.device)
        start = 0
        for lmax in range(start_lmax, end_lmax + 1):
            block = wigner_D(lmax, alpha, beta, gamma)
            end = start + block.size()[1]
            wigner[:, start:end, start:end] = block
            start = end

        return wigner.detach()


# Borrowed from e3nn @ 0.4.0:
# https://github.com/e3nn/e3nn/blob/0.4.0/e3nn/o3/_wigner.py#L37
#
# In 0.5.0, e3nn shifted to torch.matrix_exp which is significantly slower:
# https://github.com/e3nn/e3nn/blob/0.5.0/e3nn/o3/_wigner.py#L92
def wigner_D(l, alpha, beta, gamma):
    if not l < len(_Jd):
        raise NotImplementedError(
            f"wigner D maximum l implemented is {len(_Jd) - 1}, send us an email to ask for more"
        )

    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    J = _Jd[l].to(dtype=alpha.dtype, device=alpha.device)
    Xa = _z_rot_mat(alpha, l)
    Xb = _z_rot_mat(beta, l)
    Xc = _z_rot_mat(gamma, l)
    return Xa @ J @ Xb @ J @ Xc


def _z_rot_mat(angle, l):
    shape, device, dtype = angle.shape, angle.device, angle.dtype
    M = angle.new_zeros((*shape, 2 * l + 1, 2 * l + 1))
    inds = torch.arange(0, 2 * l + 1, 1, device=device)
    reversed_inds = torch.arange(2 * l, -1, -1, device=device)
    frequencies = torch.arange(l, -l - 1, -1, dtype=dtype, device=device)
    M[..., inds, reversed_inds] = torch.sin(frequencies * angle[..., None])
    M[..., inds, inds] = torch.cos(frequencies * angle[..., None])
    return M
