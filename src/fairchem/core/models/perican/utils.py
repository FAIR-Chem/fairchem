import torch
import torch.nn.functional as F

def masked_sum(T: torch.Tensor, mask: torch.Tensor, keepdim: bool =False):
    mask = mask[..., None].float()
    return (T * mask).sum((0, 1), keepdim=keepdim)

def masked_mean(T: torch.Tensor, mask: torch.Tensor, keepdim: bool =False):
    mask = mask[..., None].float()
    return (T * mask).sum((0, 1), keepdim=keepdim) / mask.sum((0, 1), keepdim=keepdim) 

def masked_var(T: torch.Tensor, mask: torch.Tensor, keepdim: bool =False):
    mean = masked_mean(T, mask, keepdim=True)
    return masked_mean((T - mean).square(), mask, keepdim=keepdim)

def masked_min(T: torch.Tensor, mask: torch.Tensor, keepdim: bool =False):
    T = T.masked_fill(mask[..., None], torch.inf)
    return T.min(dim=(0, 1), keepdim=keepdim)

def masked_max(T: torch.Tensor, mask: torch.Tensor, keepdim: bool =False):
    T = T.masked_fill(mask[..., None], - torch.inf)
    return T.max(dim=(0, 1), keepdim=keepdim)