import logging
from typing import Optional

import torch
from torch import nn

from ocpmodels.common import distutils


class L2MAELoss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction
        assert reduction in ["mean", "sum"]

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        dists = torch.norm(input - target, p=2, dim=-1)
        if self.reduction == "mean":
            return torch.mean(dists)
        elif self.reduction == "sum":
            return torch.sum(dists)


class AtomwiseL2Loss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction
        assert reduction in ["mean", "sum"]

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        natoms: torch.Tensor,
    ):
        assert natoms.shape[0] == input.shape[0] == target.shape[0]
        assert len(natoms.shape) == 1  # (nAtoms, )

        dists = torch.norm(input - target, p=2, dim=-1)
        loss = natoms * dists

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)


class DDPLoss(nn.Module):
    def __init__(
        self, loss_fn, loss_name: str = "mae", reduction: str = "mean"
    ) -> None:
        super().__init__()
        self.loss_fn = loss_fn
        self.loss_name = loss_name
        self.reduction = reduction
        assert reduction in ["mean", "mean_all", "sum"]

        # for forces, we want to sum over xyz errors and average over batches/atoms (mean)
        # for other metrics, we want to average over all axes (mean_all) or leave as a sum (sum)
        if reduction == "mean_all":
            self.loss_fn.reduction = "mean"
        else:
            self.loss_fn.reduction = "sum"

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        natoms: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
    ):
        # ensure torch doesn't do any unwanted broadcasting
        assert (
            input.shape == target.shape
        ), f"Mismatched shapes: {input.shape} and {target.shape}"

        # zero out nans, if any
        found_nans_or_infs = not torch.all(input.isfinite())
        if found_nans_or_infs is True:
            logging.warning("Found nans while computing loss")
            input = torch.nan_to_num(input, nan=0.0)

        if self.loss_name.startswith("atomwise"):
            loss = self.loss_fn(input, target, natoms)
        else:
            loss = self.loss_fn(input, target)

        if self.reduction == "mean":
            num_samples = (
                batch_size
                if self.loss_name.startswith("atomwise")
                else input.shape[0]
            )
            num_samples = distutils.all_reduce(
                num_samples, device=input.device
            )
            # Multiply by world size since gradients are averaged
            # across DDP replicas
            return loss * distutils.get_world_size() / num_samples
        else:
            # if reduction is sum or mean over all axes, no other operations are needed
            return loss
