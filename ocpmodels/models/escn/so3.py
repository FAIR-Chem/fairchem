"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
from typing import List

import torch

try:
    from e3nn import o3
    from e3nn.o3 import FromS2Grid, ToS2Grid
except ImportError:
    pass

# Borrowed from e3nn @ 0.4.0:
# https://github.com/e3nn/e3nn/blob/0.4.0/e3nn/o3/_wigner.py#L10
# _Jd is a list of tensors of shape (2l+1, 2l+1)
_Jd = torch.load(os.path.join(os.path.dirname(__file__), "Jd.pt"))


class CoefficientMapping:
    """
    Helper functions for coefficients used to reshape l<-->m and to get coefficients of specific degree or order

    Args:
        lmax_list (list:int):   List of maximum degree of the spherical harmonics
        mmax_list (list:int):   List of maximum order of the spherical harmonics
        device:                 Device of the output
    """

    def __init__(
        self,
        lmax_list: List[int],
        mmax_list: List[int],
        device,
    ) -> None:
        super().__init__()

        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(lmax_list)
        self.device = device

        # Compute the degree (l) and order (m) for each
        # entry of the embedding

        self.l_harmonic = torch.tensor([], device=self.device).long()
        self.m_harmonic = torch.tensor([], device=self.device).long()
        self.m_complex = torch.tensor([], device=self.device).long()

        self.res_size = torch.zeros(
            [self.num_resolutions], device=self.device
        ).long()
        offset = 0
        for i in range(self.num_resolutions):
            for lval in range(0, self.lmax_list[i] + 1):
                mmax = min(self.mmax_list[i], lval)
                m = torch.arange(-mmax, mmax + 1, device=self.device).long()
                self.m_complex = torch.cat([self.m_complex, m], dim=0)
                self.m_harmonic = torch.cat(
                    [self.m_harmonic, torch.abs(m).long()], dim=0
                )
                self.l_harmonic = torch.cat(
                    [self.l_harmonic, m.fill_(lval).long()], dim=0
                )
            self.res_size[i] = len(self.l_harmonic) - offset
            offset = len(self.l_harmonic)

        num_coefficients = len(self.l_harmonic)
        self.to_m = torch.zeros(
            [num_coefficients, num_coefficients], device=self.device
        )
        self.m_size = torch.zeros(
            [max(self.mmax_list) + 1], device=self.device
        ).long()

        # The following is implemented poorly - very slow. It only gets called
        # a few times so haven't optimized.
        offset = 0
        for m in range(max(self.mmax_list) + 1):
            idx_r, idx_i = self.complex_idx(m)

            for idx_out, idx_in in enumerate(idx_r):
                self.to_m[idx_out + offset, idx_in] = 1.0
            offset = offset + len(idx_r)
            self.m_size[m] = int(len(idx_r))

            for idx_out, idx_in in enumerate(idx_i):
                self.to_m[idx_out + offset, idx_in] = 1.0
            offset = offset + len(idx_i)

        self.to_m = self.to_m.detach()

    # Return mask containing coefficients of order m (real and imaginary parts)
    def complex_idx(self, m, lmax: int = -1):
        if lmax == -1:
            lmax = max(self.lmax_list)

        indices = torch.arange(len(self.l_harmonic), device=self.device)
        # Real part
        mask_r = torch.bitwise_and(
            self.l_harmonic.le(lmax), self.m_complex.eq(m)
        )
        mask_idx_r = torch.masked_select(indices, mask_r)

        mask_idx_i = torch.tensor([], device=self.device).long()
        # Imaginary part
        if m != 0:
            mask_i = torch.bitwise_and(
                self.l_harmonic.le(lmax), self.m_complex.eq(-m)
            )
            mask_idx_i = torch.masked_select(indices, mask_i)

        return mask_idx_r, mask_idx_i

    # Return mask containing coefficients less than or equal to degree (l) and order (m)
    def coefficient_idx(self, lmax: int, mmax: int) -> torch.Tensor:
        mask = torch.bitwise_and(
            self.l_harmonic.le(lmax), self.m_harmonic.le(mmax)
        )
        indices = torch.arange(len(mask), device=self.device)

        return torch.masked_select(indices, mask)


class SO3_Embedding(torch.nn.Module):
    """
    Helper functions for irreps embedding

    Args:
        length (int):           Batch size
        lmax_list (list:int):   List of maximum degree of the spherical harmonics
        num_channels (int):     Number of channels
        device:                 Device of the output
        dtype:                  type of the output tensors
    """

    def __init__(
        self,
        length: int,
        lmax_list: List[int],
        num_channels: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.device = device
        self.dtype = dtype
        self.num_resolutions = len(lmax_list)

        self.num_coefficients = 0
        for i in range(self.num_resolutions):
            self.num_coefficients = self.num_coefficients + int(
                (lmax_list[i] + 1) ** 2
            )

        embedding = torch.zeros(
            length,
            self.num_coefficients,
            self.num_channels,
            device=self.device,
            dtype=self.dtype,
        )

        self.set_embedding(embedding)
        self.set_lmax_mmax(lmax_list, lmax_list.copy())

    # Clone an embedding of irreps
    def clone(self) -> "SO3_Embedding":
        clone = SO3_Embedding(
            0,
            self.lmax_list.copy(),
            self.num_channels,
            self.device,
            self.dtype,
        )

        clone.set_embedding(self.embedding.clone())

        return clone

    # Initialize an embedding of irreps
    def set_embedding(self, embedding) -> None:
        self.length = len(embedding)
        self.embedding = embedding

    # Set the maximum order to be the maximum degree
    def set_lmax_mmax(self, lmax_list, mmax_list) -> None:
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list

    # Expand the node embeddings to the number of edges
    def _expand_edge(self, edge_index) -> None:
        embedding = self.embedding[edge_index]
        self.set_embedding(embedding)

    # Initialize an embedding of irreps of a neighborhood
    def expand_edge(self, edge_index) -> "SO3_Embedding":
        x_expand = SO3_Embedding(
            0,
            self.lmax_list.copy(),
            self.num_channels,
            self.device,
            self.dtype,
        )
        x_expand.set_embedding(self.embedding[edge_index])
        return x_expand

    # Compute the sum of the embeddings of the neighborhood
    def _reduce_edge(self, edge_index, num_nodes: int) -> None:
        new_embedding = torch.zeros(
            num_nodes,
            self.num_coefficients,
            self.num_channels,
            device=self.embedding.device,
            dtype=self.embedding.dtype,
        )
        new_embedding.index_add_(0, edge_index, self.embedding)
        self.set_embedding(new_embedding)

    # Reshape the embedding l-->m
    def _m_primary(self, mapping) -> None:
        self.embedding = torch.einsum(
            "nac,ba->nbc", self.embedding, mapping.to_m
        )

    # Reshape the embedding m-->l
    def _l_primary(self, mapping) -> None:
        self.embedding = torch.einsum(
            "nac,ab->nbc", self.embedding, mapping.to_m
        )

    # Rotate the embedding
    def _rotate(self, SO3_rotation, lmax_list, mmax_list) -> None:
        embedding_rotate = torch.tensor(
            [], device=self.device, dtype=self.dtype
        )

        offset = 0
        for i in range(self.num_resolutions):
            num_coefficients = int((self.lmax_list[i] + 1) ** 2)
            embedding_i = self.embedding[:, offset : offset + num_coefficients]
            embedding_rotate = torch.cat(
                [
                    embedding_rotate,
                    SO3_rotation[i].rotate(
                        embedding_i, lmax_list[i], mmax_list[i]
                    ),
                ],
                dim=1,
            )
            offset = offset + num_coefficients

        self.embedding = embedding_rotate
        self.set_lmax_mmax(lmax_list.copy(), mmax_list.copy())

    # Rotate the embedding by the inverse of the rotation matrix
    def _rotate_inv(self, SO3_rotation, mappingReduced) -> None:
        embedding_rotate = torch.tensor(
            [], device=self.device, dtype=self.dtype
        )

        offset = 0
        for i in range(self.num_resolutions):
            num_coefficients = mappingReduced.res_size[i]
            embedding_i = self.embedding[:, offset : offset + num_coefficients]
            embedding_rotate = torch.cat(
                [
                    embedding_rotate,
                    SO3_rotation[i].rotate_inv(
                        embedding_i, self.lmax_list[i], self.mmax_list[i]
                    ),
                ],
                dim=1,
            )
            offset = offset + num_coefficients

        self.embedding = embedding_rotate

        # Assume mmax = lmax when rotating back
        for i in range(self.num_resolutions):
            self.mmax_list[i] = int(self.lmax_list[i])

        self.set_lmax_mmax(self.lmax_list, self.mmax_list)

    # Compute point-wise spherical non-linearity
    def _grid_act(self, SO3_grid, act, mappingReduced) -> None:
        offset = 0
        for i in range(self.num_resolutions):
            num_coefficients = mappingReduced.res_size[i]

            x_res = self.embedding[
                :, offset : offset + num_coefficients
            ].contiguous()
            to_grid_mat = SO3_grid[self.lmax_list[i]][
                self.mmax_list[i]
            ].get_to_grid_mat(self.device)
            from_grid_mat = SO3_grid[self.lmax_list[i]][
                self.mmax_list[i]
            ].get_from_grid_mat(self.device)

            x_grid = torch.einsum("bai,zic->zbac", to_grid_mat, x_res)
            x_grid = act(x_grid)
            x_res = torch.einsum("bai,zbac->zic", from_grid_mat, x_grid)

            self.embedding[:, offset : offset + num_coefficients] = x_res
            offset = offset + num_coefficients

    # Compute a sample of the grid
    def to_grid(self, SO3_grid, lmax: int = -1) -> torch.Tensor:
        if lmax == -1:
            lmax = max(self.lmax_list)

        to_grid_mat_lmax = SO3_grid[lmax][lmax].get_to_grid_mat(self.device)
        grid_mapping = SO3_grid[lmax][lmax].mapping

        offset = 0
        x_grid = torch.tensor([], device=self.device)

        for i in range(self.num_resolutions):
            num_coefficients = int((self.lmax_list[i] + 1) ** 2)
            x_res = self.embedding[
                :, offset : offset + num_coefficients
            ].contiguous()
            to_grid_mat = to_grid_mat_lmax[
                :,
                :,
                grid_mapping.coefficient_idx(
                    self.lmax_list[i], self.lmax_list[i]
                ),
            ]
            x_grid = torch.cat(
                [x_grid, torch.einsum("bai,zic->zbac", to_grid_mat, x_res)],
                dim=3,
            )
            offset = offset + num_coefficients

        return x_grid

    # Compute irreps from grid representation
    def _from_grid(self, x_grid, SO3_grid, lmax: int = -1) -> None:
        if lmax == -1:
            lmax = max(self.lmax_list)

        from_grid_mat_lmax = SO3_grid[lmax][lmax].get_from_grid_mat(
            self.device
        )
        grid_mapping = SO3_grid[lmax][lmax].mapping

        offset = 0
        offset_channel = 0
        for i in range(self.num_resolutions):
            from_grid_mat = from_grid_mat_lmax[
                :,
                :,
                grid_mapping.coefficient_idx(
                    self.lmax_list[i], self.lmax_list[i]
                ),
            ]
            x_res = torch.einsum(
                "bai,zbac->zic",
                from_grid_mat,
                x_grid[
                    :,
                    :,
                    :,
                    offset_channel : offset_channel + self.num_channels,
                ],
            )
            num_coefficients = int((self.lmax_list[i] + 1) ** 2)
            self.embedding[:, offset : offset + num_coefficients] = x_res
            offset = offset + num_coefficients
            offset_channel = offset_channel + self.num_channels


class SO3_Rotation(torch.nn.Module):
    """
    Helper functions for Wigner-D rotations

    Args:
        rot_mat3x3 (tensor):    Rotation matrix
        lmax_list (list:int):   List of maximum degree of the spherical harmonics
    """

    def __init__(
        self,
        rot_mat3x3: torch.Tensor,
        lmax: List[int],
    ) -> None:
        super().__init__()
        self.device = rot_mat3x3.device
        self.dtype = rot_mat3x3.dtype

        self.wigner = self.RotationToWignerDMatrix(rot_mat3x3, 0, lmax)
        self.wigner_inv = torch.transpose(self.wigner, 1, 2).contiguous()

        self.wigner = self.wigner.detach()
        self.wigner_inv = self.wigner_inv.detach()

        self.set_lmax(lmax)

    # Initialize coefficients for reshape l<-->m
    def set_lmax(self, lmax) -> None:
        self.lmax = lmax
        self.mapping = CoefficientMapping(
            [self.lmax], [self.lmax], self.device
        )

    # Rotate the embedding
    def rotate(self, embedding, out_lmax, out_mmax) -> torch.Tensor:
        out_mask = self.mapping.coefficient_idx(out_lmax, out_mmax)
        wigner = self.wigner[:, out_mask, :]
        return torch.bmm(wigner, embedding)

    # Rotate the embedding by the inverse of the rotation matrix
    def rotate_inv(self, embedding, in_lmax, in_mmax) -> torch.Tensor:
        in_mask = self.mapping.coefficient_idx(in_lmax, in_mmax)
        wigner_inv = self.wigner_inv[:, :, in_mask]

        return torch.bmm(wigner_inv, embedding)

    # Compute Wigner matrices from rotation matrix
    def RotationToWignerDMatrix(
        self, edge_rot_mat: torch.Tensor, start_lmax: int, end_lmax: int
    ) -> torch.Tensor:
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
            block = self.wigner_D(lmax, alpha, beta, gamma)
            end = start + block.size()[1]
            wigner[:, start:end, start:end] = block
            start = end

        return wigner.detach()

    # Borrowed from e3nn @ 0.4.0:
    # https://github.com/e3nn/e3nn/blob/0.4.0/e3nn/o3/_wigner.py#L37
    #
    # In 0.5.0, e3nn shifted to torch.matrix_exp which is significantly slower:
    # https://github.com/e3nn/e3nn/blob/0.5.0/e3nn/o3/_wigner.py#L92
    def wigner_D(self, lval, alpha, beta, gamma):
        if not lval < len(_Jd):
            raise NotImplementedError(
                f"wigner D maximum l implemented is {len(_Jd) - 1}, send us an email to ask for more"
            )

        alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
        J = _Jd[lval].to(dtype=alpha.dtype, device=alpha.device)
        Xa = self._z_rot_mat(alpha, lval)
        Xb = self._z_rot_mat(beta, lval)
        Xc = self._z_rot_mat(gamma, lval)
        return Xa @ J @ Xb @ J @ Xc

    def _z_rot_mat(self, angle: torch.Tensor, lv: int) -> torch.Tensor:
        shape, device, dtype = angle.shape, angle.device, angle.dtype
        M = angle.new_zeros((*shape, 2 * lv + 1, 2 * lv + 1))
        inds = torch.arange(0, 2 * lv + 1, 1, device=device)
        reversed_inds = torch.arange(2 * lv, -1, -1, device=device)
        frequencies = torch.arange(lv, -lv - 1, -1, dtype=dtype, device=device)
        M[..., inds, reversed_inds] = torch.sin(frequencies * angle[..., None])
        M[..., inds, inds] = torch.cos(frequencies * angle[..., None])
        return M


class SO3_Grid(torch.nn.Module):
    """
    Helper functions for grid representation of the irreps

    Args:
        lmax (int):   Maximum degree of the spherical harmonics
        mmax (int):   Maximum order of the spherical harmonics
    """

    def __init__(
        self,
        lmax: int,
        mmax: int,
    ) -> None:
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.lat_resolution = 2 * (self.lmax + 1)
        if lmax == mmax:
            self.long_resolution = 2 * (self.mmax + 1) + 1
        else:
            self.long_resolution = 2 * (self.mmax) + 1

        self.initialized = False

    def _initialize(self, device: torch.device) -> None:
        if self.initialized is True:
            return
        self.mapping = CoefficientMapping([self.lmax], [self.lmax], device)

        to_grid = ToS2Grid(
            self.lmax,
            (self.lat_resolution, self.long_resolution),
            normalization="integral",
            device=device,
        )

        self.to_grid_mat = torch.einsum(
            "mbi,am->bai", to_grid.shb, to_grid.sha
        ).detach()
        self.to_grid_mat = self.to_grid_mat[
            :, :, self.mapping.coefficient_idx(self.lmax, self.mmax)
        ]

        from_grid = FromS2Grid(
            (self.lat_resolution, self.long_resolution),
            self.lmax,
            normalization="integral",
            device=device,
        )

        self.from_grid_mat = torch.einsum(
            "am,mbi->bai", from_grid.sha, from_grid.shb
        ).detach()
        self.from_grid_mat = self.from_grid_mat[
            :, :, self.mapping.coefficient_idx(self.lmax, self.mmax)
        ]

        self.initialized = True

    # Compute matrices to transform irreps to grid
    def get_to_grid_mat(self, device: torch.device):
        self._initialize(device)
        return self.to_grid_mat

    # Compute matrices to transform grid to irreps
    def get_from_grid_mat(self, device: torch.device):
        self._initialize(device)
        return self.from_grid_mat

    # Compute grid from irreps representation
    def to_grid(
        self, embedding: torch.Tensor, lmax: int, mmax: int
    ) -> torch.Tensor:
        self._initialize(embedding.device)
        to_grid_mat = self.to_grid_mat[
            :, :, self.mapping.coefficient_idx(lmax, mmax)
        ]
        grid = torch.einsum("bai,zic->zbac", to_grid_mat, embedding)
        return grid

    # Compute irreps from grid representation
    def from_grid(
        self, grid: torch.Tensor, lmax: int, mmax: int
    ) -> torch.Tensor:
        self._initialize(grid.device)
        from_grid_mat = self.from_grid_mat[
            :, :, self.mapping.coefficient_idx(lmax, mmax)
        ]
        embedding = torch.einsum("bai,zbac->zic", from_grid_mat, grid)
        return embedding
