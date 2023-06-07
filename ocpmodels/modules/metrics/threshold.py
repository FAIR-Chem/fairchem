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
        batch: torch.Tensor,
        graph_mask: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
    ):
        B = target["energy"].shape[0]

        # Compute the fmax for each graph
        error_forces = torch.abs(
            target["forces"] - prediction["forces"]
        )  # n 3
        batch_forces = batch
        if node_mask is not None:
            error_forces = error_forces[node_mask]
            batch_forces = batch_forces[node_mask]
        max_error_forces = scatter(
            error_forces,
            batch_forces,
            dim=0,
            dim_size=B,
            reduce="max",
        )  # b 3
        max_error_forces = reduce(max_error_forces, "b p -> b", "max")  # b
        if graph_mask is not None:
            max_error_forces = max_error_forces[graph_mask]  # b

        # compute the energy MAEs
        error_energy = torch.abs(target["energy"] - prediction["energy"])  # b
        if graph_mask is not None:
            error_energy = error_energy[graph_mask]  # b

        success = (error_energy < self.energy_threshold) & (
            max_error_forces < self.forces_threshold
        )
        self.le_threshold += success.sum()
        self.total += success.size(0)

    @override
    def compute(self):
        return self.le_threshold / self.total
