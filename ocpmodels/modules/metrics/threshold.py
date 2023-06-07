from typing import TypedDict
from typing_extensions import override

import torch
import torchmetrics
from einops import reduce
from torch_scatter import scatter


class WithinThreshold(torchmetrics.Metric):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    le_threshold: torch.Tensor
    total: torch.Tensor

    def __init__(self, threshold: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.threshold = threshold
        self.add_state(
            "le_threshold", default=torch.tensor(0), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    @override
    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        error = torch.abs(target - prediction)
        success = (error < self.threshold).sum()
        self.le_threshold += success

        total = target.size(0)
        self.total += total

    @override
    def compute(self):
        return self.le_threshold / self.total


class EnergyForces(TypedDict):
    energy: torch.Tensor
    forces: torch.Tensor


class EnergyForcesWithinThreshold(torchmetrics.Metric):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    le_threshold: torch.Tensor
    total: torch.Tensor

    def __init__(
        self,
        energy_threshold: float,
        forces_threshold: float,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.energy_threshold = energy_threshold
        self.forces_threshold = forces_threshold
        self.add_state(
            "le_threshold", default=torch.tensor(0), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    @override
    def update(
        self,
        prediction: EnergyForces,
        target: EnergyForces,
        batch_idx: torch.Tensor,
    ):
        B = target["energy"].shape[0]

        # Compute the fmax for each graph
        error_forces = torch.abs(
            target["forces"] - prediction["forces"]
        )  # n 3
        max_error_forces = scatter(
            error_forces,
            batch_idx,
            dim=0,
            dim_size=B,
            reduce="max",
        )  # b 3

        # In some circumstances (specifically when free_atoms_only is True for metric computation),
        # there may not be any free atoms. In these cases, the max_error_forces will be 0.
        # To deal w/ this, we compute the natoms_per_graph  and set the max_error_forces to inf for graphs w/ no free atoms.
        natoms_per_graph = scatter(
            torch.ones_like(batch_idx),
            batch_idx,
            dim=0,
            dim_size=B,
            reduce="sum",
        )  # b
        max_error_forces[natoms_per_graph == 0] = float("inf")

        # Compute fmax per graph
        max_error_forces = reduce(
            max_error_forces,
            "b p -> b",
            "max",
        )  # b

        # Compute the energy MAEs
        error_energy = torch.abs(target["energy"] - prediction["energy"])  # b

        success = (error_energy < self.energy_threshold) & (
            max_error_forces < self.forces_threshold
        )
        self.le_threshold += success.sum()
        self.total += success.size(0)

    @override
    def compute(self):
        return self.le_threshold / self.total
