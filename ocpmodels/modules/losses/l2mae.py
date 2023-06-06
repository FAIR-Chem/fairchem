from typing import Literal
import torch
import torch.nn.functional as F


def l2mae(
    input: torch.Tensor,  # (N, 3)
    target: torch.Tensor,  # (N, 3)
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    # Take the L2 norm of the difference between the input and target tensors.
    # If input and target are positions, this is the distance between them.
    dists = F.pairwise_distance(input, target, p=2)  # (N,)

    if reduction == "mean":
        return torch.mean(dists)
    elif reduction == "sum":
        return torch.sum(dists)
    elif reduction == "none":
        return dists
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
