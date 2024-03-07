import os

import torch

from torch.cuda import nvtx


# Borrowed from e3nn @ 0.4.0:
# https://github.com/e3nn/e3nn/blob/0.4.0/e3nn/o3/_wigner.py#L10
# _Jd is a list of tensors of shape (2l+1, 2l+1)
_Jd = torch.load(os.path.join(os.path.dirname(__file__), "Jd.pt"))

_Jd_ondevice_dict = {}


def _Jd_ondevice_and_dtype(device, dtype):
    device_and_dtype = (device, dtype)
    if device_and_dtype not in _Jd_ondevice_dict:
        _Jd_ondevice_dict[device_and_dtype] = {
            lv: _Jd[lv].to(dtype=dtype, device=device)
            for lv in range(len(_Jd))
        }
    return _Jd_ondevice_dict[device_and_dtype]


# Borrowed from e3nn @ 0.4.0:
# https://github.com/e3nn/e3nn/blob/0.4.0/e3nn/o3/_wigner.py#L37
#
# In 0.5.0, e3nn shifted to torch.matrix_exp which is significantly slower:
# https://github.com/e3nn/e3nn/blob/0.5.0/e3nn/o3/_wigner.py#L92


def wigner_D(
    lv: int, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor
) -> torch.Tensor:
    if not lv < len(_Jd):
        raise NotImplementedError(
            f"wigner D maximum l implemented is {len(_Jd) - 1}, send us an email to ask for more"
        )

    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    # J = _Jd[lv].to(dtype=alpha.dtype, device=alpha.device)
    J = _Jd_ondevice_and_dtype(device=alpha.device, dtype=alpha.dtype)[lv]
    Xa = _z_rot_mat(alpha, lv)
    Xb = _z_rot_mat(beta, lv)
    Xc = _z_rot_mat(gamma, lv)
    r = Xa @ J @ Xb @ J @ Xc
    return r


cached_aranges = {}


def get_cached_arange(start, end, step, device, dtype=None):
    key = (start, end, step, device, dtype)
    if key not in cached_aranges:
        cached_aranges[key] = torch.arange(
            start, end, step, dtype=dtype, device=device
        )
    return cached_aranges[key]


def _z_rot_mat(angle: torch.Tensor, lv: int) -> torch.Tensor:
    shape, device, dtype = angle.shape, angle.device, angle.dtype
    M = angle.new_zeros((*shape, 2 * lv + 1, 2 * lv + 1))
    # inds = torch.arange(0, 2 * lv + 1, 1, device=device)
    inds = get_cached_arange(0, 2 * lv + 1, 1, device=device)
    # reversed_inds = torch.arange(2 * lv, -1, -1, device=device)
    reversed_inds = get_cached_arange(2 * lv, -1, -1, device=device)
    # frequencies = torch.arange(lv, -lv - 1, -1, dtype=dtype, device=device)
    frequencies = get_cached_arange(
        lv, -lv - 1, -1, dtype=dtype, device=device
    )
    tmp = frequencies * angle[..., None]
    M[..., inds, reversed_inds] = torch.sin(tmp)
    M[..., inds, inds] = torch.cos(tmp)
    return M
