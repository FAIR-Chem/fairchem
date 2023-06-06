from typing import Literal
import torch
import torch.nn.functional as F


def atomwisel2(
    input: torch.Tensor,  # (N, 3)
    target: torch.Tensor,  # (N, 3)
    natoms: torch.Tensor,  # (N,)
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    # Take the L2 norm of the difference between the input and target tensors.
    # If input and target are positions, this is the distance between them.
    dists = F.pairwise_distance(input, target, p=2)  # (N,)

    # Scale up the loss by the number of atoms in each molecule.
    loss = dists * natoms  # (N,)

    # Reduce the loss.
    match reduction:
        case "mean":
            loss = torch.mean(loss)
        case "sum":
            loss = torch.sum(loss)
        case "none":
            pass
        case _:
            raise ValueError(f"Invalid reduction: {reduction}")
    return loss
