import torch
import torch.nn.functional as F

from typing import Tuple

def diagonal(input, dim1=0, dim2=1):
    return torch.diagonal(input, dim1=dim1, dim2=dim2).permute(2, 0, 1)

def diag_embed(input, dim1=0, dim2=1):
    return torch.diag_embed(input.permute(1, 2, 0), dim1=dim1, dim2=dim2)

def masked_sum(
        T: torch.Tensor, 
        mask: torch.Tensor,
        dim: Tuple[int] = (0, 1),
        keepdim: bool = False,
        scale: float = 1,
    ) -> torch.Tensor:
    mask = mask[..., None].float()
    return  (T * mask).sum(dim, keepdim=keepdim) / scale

def masked_mean(
        T: torch.Tensor, 
        mask: torch.Tensor,
        dim: Tuple[int] = (0, 1),
        keepdim: bool = False,
        scale: float = 1,
    ) -> torch.Tensor:
    mask = mask[..., None].float()
    return (T * mask).sum(dim, keepdim=keepdim) / mask.sum(dim, keepdim=keepdim) 

def masked_var(
        T: torch.Tensor, 
        mask: torch.Tensor,
        dim: Tuple[int] = (0, 1),
        keepdim: bool = False,
        scale: float = 1,
    ) -> torch.Tensor:
    mean = masked_mean(T, mask, dim=dim, keepdim=True)
    return masked_mean((T - mean).square(), mask, dim=dim, keepdim=keepdim)

def masked_min(
        T: torch.Tensor, 
        mask: torch.Tensor,
        dim: Tuple[int] = (0, 1),
        keepdim: bool = False,
        scale: float = 1,
    ) -> torch.Tensor:
    T = T.masked_fill(mask[..., None], torch.inf)
    return T.amin(dim=dim, keepdim=keepdim)

def masked_max(
        T: torch.Tensor, 
        mask: torch.Tensor,
        dim: Tuple[int] = (0, 1),
        keepdim: bool = False,
        scale: float = 1,
    ) -> torch.Tensor:
    T = T.masked_fill(mask[..., None], - torch.inf)
    return T.amax(dim=dim, keepdim=keepdim)

name_mapping = {
    "sum": masked_sum,
    "mean": masked_mean,
    "min": masked_min,
    "max": masked_max,
    "var": masked_var
}