import torch
import torch.nn as nn
import torch.nn.functional as F

class T2toT2(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()

        self.linear1 = nn.Linear(15 * in_channels, out_channels)
        self.linear2 = nn.Linear(out_channels, out_channels)

    def forward(
        self,
        T: torch.Tensor, # (L, L, N, C)
        padded_mask: torch.Tensor # (L, L, N)
    ):
        # apply masking
        T.masked_fill_(~padded_mask, 0)

        # prepare inputs
        L = T.size(0)
        diag = torch.diagonal(T, dim1=0, dim2=1)
        diag_sum = torch.diagonal(T, dim1=0, dim2=1).sum(dim=0)
        row_sum = torch.sum(T, dim=0)
        col_sum = torch.sum(T, dim=1)
        transpose = T.transpose(0, 1)
        all_sum = torch.sum(T, dim=(0, 1))

        # compute outputs
        outputs = []
        outputs.append(torch.diag_embed(diag, dim1=0, dim2=1))
        outputs.append(diag[None, :])
        outputs.append(diag[:, None])
        outputs.append(torch.diag_embed(row_sum, dim1=0, dim2=1))
        outputs.append(torch.diag_embed(col_sum, dim1=0, dim2=1))
        outputs.append(torch.diag_embed(diag_sum[None].expand(L, -1, -1), dim1=0, dim2=1))
        outputs.append(transpose)
        outputs.append(T)
        outputs.append(diag_sum[None, :])
        outputs.append(row_sum[None, :])
        outputs.append(col_sum[None, :])
        outputs.append(torch.diag_embed(all_sum[None], dim1=0, dim2=1))
        outputs.append(col_sum[:, None])
        outputs.append(row_sum[:, None])
        outputs.append(all_sum[None, None])
        outputs = torch.cat(outputs, dim=3) # (L, L, N, 15 * C)

class MaskedBatchNorm2d(nn.Module):
    def __init__(
        self,
        num_channels: int = 128,
        eps: float = 1e-5,
        momentum: float = 0.1
    ):
        super().__init__()

        self.register_buffer("running_mean", torch.zeros((num_channels,)))
        self.register_buffer("running_var", torch.ones((num_channels,)))

        self.weight = nn.Parameter(torch.empty((num_channels,)))
        self.bias = nn.Parameter(torch.empty((num_channels,)))
                                               
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.fill_(0)
        self.running_var.fill_(1)
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        T: torch.Tensor,
        padded_mask: torch.Tensor
    ):
        if self.training:
            mean
            
