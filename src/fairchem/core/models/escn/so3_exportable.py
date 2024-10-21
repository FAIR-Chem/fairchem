from __future__ import annotations

import math

import torch

try:
    from e3nn import o3
    from e3nn.o3 import FromS2Grid, ToS2Grid
except ImportError:
    pass


# Borrowed from e3nn @ 0.4.0:
# https://github.com/e3nn/e3nn/blob/0.4.0/e3nn/o3/_wigner.py#L37
#
# In 0.5.0, e3nn shifted to torch.matrix_exp which is significantly slower:
# https://github.com/e3nn/e3nn/blob/0.5.0/e3nn/o3/_wigner.py#L92
def wigner_D(
    lv: int,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    _Jd: list[torch.Tensor],
) -> torch.Tensor:
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    J = _Jd[lv]
    Xa = _z_rot_mat(alpha, lv)
    Xb = _z_rot_mat(beta, lv)
    Xc = _z_rot_mat(gamma, lv)
    return Xa @ J @ Xb @ J @ Xc


def _z_rot_mat(angle: torch.Tensor, lv: int) -> torch.Tensor:
    M = angle.new_zeros((*angle.shape, 2 * lv + 1, 2 * lv + 1))

    # The following code needs to replaced for a for loop because
    # torch.export barfs on outer product like operations
    # ie: torch.outer(frequences, angle) (same as frequencies * angle[..., None])
    # will place a non-sense Guard on the dimensions of angle when attempting to export setting
    # angle (edge dimensions) as dynamic. This may be fixed in torch2.4.

    # inds = torch.arange(0, 2 * lv + 1, 1, device=device)
    # reversed_inds = torch.arange(2 * lv, -1, -1, device=device)
    # frequencies = torch.arange(lv, -lv - 1, -1, dtype=dtype, device=device)
    # M[..., inds, reversed_inds] = torch.sin(frequencies * angle[..., None])
    # M[..., inds, inds] = torch.cos(frequencies * angle[..., None])

    inds = list(range(0, 2 * lv + 1, 1))
    reversed_inds = list(range(2 * lv, -1, -1))
    frequencies = list(range(lv, -lv - 1, -1))
    for i in range(len(frequencies)):
        M[..., inds[i], reversed_inds[i]] = torch.sin(frequencies[i] * angle)
        M[..., inds[i], inds[i]] = torch.cos(frequencies[i] * angle)
    return M


def rotation_to_wigner(
    edge_rot_mat: torch.Tensor, start_lmax: int, end_lmax: int, Jd: list[torch.Tensor]
) -> torch.Tensor:
    x = edge_rot_mat @ edge_rot_mat.new_tensor([0.0, 1.0, 0.0])
    alpha, beta = o3.xyz_to_angles(x)
    R = (
        o3.angles_to_matrix(alpha, beta, torch.zeros_like(alpha)).transpose(-1, -2)
        @ edge_rot_mat
    )
    gamma = torch.atan2(R[..., 0, 2], R[..., 0, 0])

    size = int((end_lmax + 1) ** 2) - int((start_lmax) ** 2)
    wigner = torch.zeros(len(alpha), size, size, device=edge_rot_mat.device)
    start = 0
    for lmax in range(start_lmax, end_lmax + 1):
        block = wigner_D(lmax, alpha, beta, gamma, Jd)
        end = start + block.size()[1]
        wigner[:, start:end, start:end] = block
        start = end

    return wigner.detach()


class CoefficientMapping(torch.nn.Module):
    """
    Helper module for coefficients used to reshape l <--> m and to get coefficients of specific degree or order

    Args:
        lmax_list (list:int):   List of maximum degree of the spherical harmonics
        mmax_list (list:int):   List of maximum order of the spherical harmonics
        use_rotate_inv_rescale (bool):  Whether to pre-compute inverse rotation rescale matrices
    """

    def __init__(
        self,
        lmax_list,
        mmax_list,
    ):
        super().__init__()

        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(lmax_list)

        # TODO: remove this for loops here associated with lmax and mmax lists
        assert len(self.lmax_list) == 1
        assert len(self.mmax_list) == 1

        # Compute the degree (l) and order (m) for each entry of the embedding
        l_harmonic = torch.tensor([]).long()
        m_harmonic = torch.tensor([]).long()
        m_complex = torch.tensor([]).long()

        self.res_size = torch.zeros([self.num_resolutions]).long().tolist()

        offset = 0
        for i in range(self.num_resolutions):
            for l in range(self.lmax_list[i] + 1):
                mmax = min(self.mmax_list[i], l)
                m = torch.arange(-mmax, mmax + 1).long()
                m_complex = torch.cat([m_complex, m], dim=0)
                m_harmonic = torch.cat([m_harmonic, torch.abs(m).long()], dim=0)
                l_harmonic = torch.cat([l_harmonic, m.fill_(l).long()], dim=0)
            self.res_size[i] = len(l_harmonic) - offset
            offset = len(l_harmonic)

        num_coefficients = len(l_harmonic)
        # `self.to_m` moves m components from different L to contiguous index
        to_m = torch.zeros([num_coefficients, num_coefficients])
        self.m_size = torch.zeros([max(self.mmax_list) + 1]).long().tolist()

        offset = 0
        for m in range(max(self.mmax_list) + 1):
            idx_r, idx_i = self.complex_idx(m, -1, m_complex, l_harmonic)

            for idx_out, idx_in in enumerate(idx_r):
                to_m[idx_out + offset, idx_in] = 1.0
            offset = offset + len(idx_r)

            self.m_size[m] = int(len(idx_r))

            for idx_out, idx_in in enumerate(idx_i):
                to_m[idx_out + offset, idx_in] = 1.0
            offset = offset + len(idx_i)

        to_m = to_m.detach()

        # save tensors and they will be moved to GPU
        self.register_buffer("l_harmonic", l_harmonic)
        self.register_buffer("m_harmonic", m_harmonic)
        self.register_buffer("m_complex", m_complex)
        self.register_buffer("to_m", to_m)

        self.pre_compute_coefficient_idx()

    # Return mask containing coefficients of order m (real and imaginary parts)
    def complex_idx(self, m, lmax, m_complex, l_harmonic):
        """
        Add `m_complex` and `l_harmonic` to the input arguments
        since we cannot use `self.m_complex`.
        """
        if lmax == -1:
            lmax = max(self.lmax_list)

        indices = torch.arange(len(l_harmonic))
        # Real part
        mask_r = torch.bitwise_and(l_harmonic.le(lmax), m_complex.eq(m))
        mask_idx_r = torch.masked_select(indices, mask_r)

        mask_idx_i = torch.tensor([]).long()
        # Imaginary part
        if m != 0:
            mask_i = torch.bitwise_and(l_harmonic.le(lmax), m_complex.eq(-m))
            mask_idx_i = torch.masked_select(indices, mask_i)

        return mask_idx_r, mask_idx_i

    def pre_compute_coefficient_idx(self):
        """
        Pre-compute the results of `coefficient_idx()` and access them with `prepare_coefficient_idx()`
        """
        lmax = max(self.lmax_list)
        for l in range(lmax + 1):
            for m in range(lmax + 1):
                mask = torch.bitwise_and(self.l_harmonic.le(l), self.m_harmonic.le(m))
                indices = torch.arange(len(mask))
                mask_indices = torch.masked_select(indices, mask)
                self.register_buffer(f"coefficient_idx_l{l}_m{m}", mask_indices)

    def prepare_coefficient_idx(self):
        """
        Construct a list of buffers
        """
        lmax = max(self.lmax_list)
        coefficient_idx_list = []
        for l in range(lmax + 1):
            l_list = []
            for m in range(lmax + 1):
                l_list.append(getattr(self, f"coefficient_idx_l{l}_m{m}", None))
            coefficient_idx_list.append(l_list)
        return coefficient_idx_list

    # Return mask containing coefficients less than or equal to degree (l) and order (m)
    def coefficient_idx(self, lmax: int, mmax: int):
        if lmax > max(self.lmax_list) or mmax > max(self.lmax_list):
            mask = torch.bitwise_and(self.l_harmonic.le(lmax), self.m_harmonic.le(mmax))
            indices = torch.arange(len(mask), device=mask.device)
            return torch.masked_select(indices, mask)
        else:
            temp = self.prepare_coefficient_idx()
            return temp[lmax][mmax]

    def pre_compute_rotate_inv_rescale(self):
        lmax = max(self.lmax_list)
        for l in range(lmax + 1):
            for m in range(lmax + 1):
                mask_indices = self.coefficient_idx(l, m)
                rotate_inv_rescale = torch.ones(
                    (1, int((l + 1) ** 2), int((l + 1) ** 2))
                )
                for l_sub in range(l + 1):
                    if l_sub <= m:
                        continue
                    start_idx = l_sub**2
                    length = 2 * l_sub + 1
                    rescale_factor = math.sqrt(length / (2 * m + 1))
                    rotate_inv_rescale[
                        :,
                        start_idx : (start_idx + length),
                        start_idx : (start_idx + length),
                    ] = rescale_factor
                rotate_inv_rescale = rotate_inv_rescale[:, :, mask_indices]
                self.register_buffer(
                    f"rotate_inv_rescale_l{l}_m{m}", rotate_inv_rescale
                )

    def __repr__(self):
        return f"{self.__class__.__name__}(lmax_list={self.lmax_list}, mmax_list={self.mmax_list})"


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
        normalization: str = "integral",
        resolution: int | None = None,
        rescale: bool = False,
    ):
        super().__init__()

        self.lmax = lmax
        self.mmax = mmax
        self.lat_resolution = 2 * (self.lmax + 1)
        if lmax == mmax:
            self.long_resolution = 2 * (self.mmax + 1) + 1
        else:
            self.long_resolution = 2 * (self.mmax) + 1
        if resolution is not None:
            self.lat_resolution = resolution
            self.long_resolution = resolution

        self.mapping = CoefficientMapping([self.lmax], [self.lmax])

        device = "cpu"

        to_grid = ToS2Grid(
            self.lmax,
            (self.lat_resolution, self.long_resolution),
            normalization=normalization,  # normalization="integral",
            device=device,
        )
        to_grid_mat = torch.einsum("mbi, am -> bai", to_grid.shb, to_grid.sha).detach()
        # rescale based on mmax
        if rescale and lmax != mmax:
            for lval in range(lmax + 1):
                if lval <= mmax:
                    continue
                start_idx = lval**2
                length = 2 * lval + 1
                rescale_factor = math.sqrt(length / (2 * mmax + 1))
                to_grid_mat[:, :, start_idx : (start_idx + length)] = (
                    to_grid_mat[:, :, start_idx : (start_idx + length)] * rescale_factor
                )
        to_grid_mat = to_grid_mat[
            :, :, self.mapping.coefficient_idx(self.lmax, self.mmax)
        ]

        from_grid = FromS2Grid(
            (self.lat_resolution, self.long_resolution),
            self.lmax,
            normalization=normalization,  # normalization="integral",
            device=device,
        )
        from_grid_mat = torch.einsum(
            "am, mbi -> bai", from_grid.sha, from_grid.shb
        ).detach()
        # rescale based on mmax
        if rescale and lmax != mmax:
            for lval in range(lmax + 1):
                if lval <= mmax:
                    continue
                start_idx = lval**2
                length = 2 * lval + 1
                rescale_factor = math.sqrt(length / (2 * mmax + 1))
                from_grid_mat[:, :, start_idx : (start_idx + length)] = (
                    from_grid_mat[:, :, start_idx : (start_idx + length)]
                    * rescale_factor
                )
        from_grid_mat = from_grid_mat[
            :, :, self.mapping.coefficient_idx(self.lmax, self.mmax)
        ]

        # save tensors and they will be moved to GPU
        self.register_buffer("to_grid_mat", to_grid_mat)
        self.register_buffer("from_grid_mat", from_grid_mat)

    # Compute matrices to transform irreps to grid
    def get_to_grid_mat(self, device=None):
        return self.to_grid_mat

    # Compute matrices to transform grid to irreps
    def get_from_grid_mat(self, device=None):
        return self.from_grid_mat

    # Compute grid from irreps representation
    def to_grid(self, embedding, lmax: int, mmax: int):
        to_grid_mat = self.to_grid_mat[:, :, self.mapping.coefficient_idx(lmax, mmax)]
        return torch.einsum("bai, zic -> zbac", to_grid_mat, embedding)

    # Compute irreps from grid representation
    def from_grid(self, grid, lmax: int, mmax: int):
        from_grid_mat = self.from_grid_mat[
            :, :, self.mapping.coefficient_idx(lmax, mmax)
        ]
        return torch.einsum("bai, zbac -> zic", from_grid_mat, grid)
