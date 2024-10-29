from __future__ import annotations

import logging
from typing import Literal

import torch
from torch import nn

from fairchem.core.common import distutils
from fairchem.core.common.registry import registry


@registry.register_loss("mae")
class MAELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.L1Loss()
        # reduction should be none as it is handled in DDPLoss
        self.loss.reduction = "none"

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, natoms: torch.Tensor
    ) -> torch.Tensor:
        return self.loss(pred, target)


@registry.register_loss("mse")
class MSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.MSELoss()
        # reduction should be none as it is handled in DDPLoss
        self.loss.reduction = "none"

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, natoms: torch.Tensor
    ) -> torch.Tensor:
        return self.loss(pred, target)


@registry.register_loss("per_atom_mae")
class PerAtomMAELoss(nn.Module):
    """
    Simply divide a loss by the number of atoms/nodes in the graph.
    Current this loss is intened to used with scalar values, not vectors or higher tensors.
    """

    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.L1Loss()
        # reduction should be none as it is handled in DDPLoss
        self.loss.reduction = "none"

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, natoms: torch.Tensor
    ) -> torch.Tensor:
        _natoms = torch.reshape(natoms, target.shape)
        # check if target is a scalar
        assert target.dim() == 1 or (target.dim() == 2 and target.shape[1] == 1)
        # check per_atom shape
        assert (target / _natoms).shape == target.shape
        return self.loss(pred / _natoms, target / _natoms)


@registry.register_loss("l2norm")
@registry.register_loss("l2mae")
class L2NormLoss(nn.Module):
    """
    Currently this loss is intened to used with vectors.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, natoms: torch.Tensor
    ) -> torch.Tensor:
        assert target.dim() == 2
        assert target.shape[1] != 1
        return torch.linalg.vector_norm(pred - target, ord=2, dim=-1)


class DDPLoss(nn.Module):
    """
    This class is a wrapper around a loss function that does a few things
    like handle nans and importantly ensures the reduction is done
    correctly for DDP. The main issue is that DDP averages gradients
    over replicas — this only works out of the box if the dimension
    you are averaging over is completely consistent across all replicas.
    In our case, that is not true for the number of atoms per batch and
    there are edge cases when the batch size differs between replicas
    e.g. if the dataset size is not divisible by the batch_size.

    Scalars are relatively straightforward to handle, but vectors and higher tensors
    are a bit trickier. Below are two examples of forces.

    Forces input: [Nx3] target: [Nx3]
    Forces are a vector of length 3 (x,y,z) for each atom.
    Number of atoms per batch (N) is different for each DDP replica.

    MSE example:
    #### Local loss computation ####
    local_loss = MSELoss(input, target) -> [Nx3]
    num_samples = local_loss.numel() -> [Nx3]
    local_loss = sum(local_loss [Nx3]) -> [1] sum reduces the loss to a scalar
    global_samples = all_reduce(num_samples) -> [N0x3 + N1x3 + N2x3 + ...] = [1] where N0 is the number of atoms on replica 0
    local_loss = local_loss * world_size / global_samples -> [1]
    #### Global loss computation ####
    global_loss = sum(local_loss / world_size) -> [1]
    == sum(local_loss / global_samples) # this is the desired corrected mean

    Norm example:
    #### Local loss computation ####
    local_loss = L2MAELoss(input, target) -> [N]
    num_samples = local_loss.numel() -> [N]
    local_loss = sum(local_loss [N]) -> [1] sum reduces the loss to a scalar
    global_samples = all_reduce(num_samples) -> [N0 + N1 + N2 + ...] = [1] where N0 is the number of atoms on replica 0
    local_loss = local_loss * world_size / global_samples -> [1]
    #### Global loss computation ####
    global_loss = sum(local_loss / world_size) -> [1]
    == sum(local_loss / global_samples) # this is the desired corrected mean
    """

    def __init__(
        self,
        loss_name,
        reduction: Literal["mean", "sum"],
    ) -> None:
        super().__init__()
        self.loss_fn = registry.get_loss_class(loss_name)()
        # default reduction is mean
        self.reduction = reduction if reduction is not None else "mean"
        self.reduction_map = {
            "mean": self.mean,
            "sum": self.sum,
        }
        assert self.reduction in list(
            self.reduction_map.keys()
        ), "Reduction must be one of: 'mean', 'sum'"

    def sum(self, input, loss, natoms):
        # this sum will reduce the loss down to a single scalar
        return torch.sum(loss)

    def _ddp_mean(self, num_samples, loss):
        global_samples = distutils.all_reduce(num_samples, device=loss.device)
        # Multiply by world size since gradients are averaged across DDP replicas
        # warning this is probably incorrect for any model parallel approach
        return loss * distutils.get_world_size() / global_samples

    def mean(self, input, loss, natoms):
        # total elements to take the mean over
        # could be batch_size, num_atoms, num_atomsx3, etc
        num_samples = loss.numel()
        # this sum will reduce the loss down from num_sample -> 1
        loss = self.sum(input, loss, natoms)
        return self._ddp_mean(num_samples, loss)

    def _reduction(self, input, loss, natoms):
        if self.reduction in self.reduction_map:
            return self.reduction_map[self.reduction](input, loss, natoms)
        else:
            raise ValueError("Reduction must be one of: 'mean', 'sum'")

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        natoms: torch.Tensor,
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

        loss = self.loss_fn(input, target, natoms)
        return self._reduction(input, loss, natoms)
