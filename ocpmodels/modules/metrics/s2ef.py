import torch
import torch.nn as nn
import torchmetrics
from ll import TypedConfig
from torch_geometric.data import Batch
from typing_extensions import override

from .threshold import EnergyForcesWithinThreshold


class S2EFMetricsConfig(TypedConfig):
    free_atoms_only: bool = False


class S2EFMetrics(nn.Module):
    @override
    def __init__(self, config: S2EFMetricsConfig):
        super().__init__()

        self.config = config

        self.energy_mae = torchmetrics.MeanAbsoluteError()
        self.energy_force_within_threshold = EnergyForcesWithinThreshold(
            energy_threshold=0.02, forces_threshold=0.03
        )

        self.forces_mae = torchmetrics.MeanAbsoluteError()
        self.forces_x_mae = torchmetrics.MeanAbsoluteError()
        self.forces_y_mae = torchmetrics.MeanAbsoluteError()
        self.forces_z_mae = torchmetrics.MeanAbsoluteError()
        self.forces_cos = torchmetrics.CosineSimilarity(reduction="mean")
        self.forces_magnitude = torchmetrics.MeanAbsoluteError()

    @override
    def forward(
        self,
        batch: Batch,
        energy_prediction: torch.Tensor,
        force_prediction: torch.Tensor,
    ) -> dict[str, torchmetrics.Metric]:
        # Compute all metrics and return a dictionary
        # which maps metric name to metric instance
        energy_target = batch.y  # (B,)
        forces_target = batch.force  # (N, 3)
        batch_idx = batch.batch  # (N,)

        if self.config.free_atoms_only:
            # mask out fixed atoms
            mask = ~batch.fixed  # (B,)
            mask = mask[batch_idx]  # (N,)
            force_prediction = force_prediction[mask]
            forces_target = forces_target[mask]
            batch_idx = batch_idx[mask]

        # energy
        self.energy_mae(energy_prediction, energy_target)

        # force
        self.forces_mae(force_prediction, forces_target)
        self.forces_x_mae(force_prediction[:, 0], forces_target[:, 0])
        self.forces_y_mae(force_prediction[:, 1], forces_target[:, 1])
        self.forces_z_mae(force_prediction[:, 2], forces_target[:, 2])
        self.forces_cos(force_prediction, forces_target)
        self.forces_magnitude(
            torch.linalg.vector_norm(force_prediction, ord=2, dim=-1),
            torch.linalg.vector_norm(forces_target, ord=2, dim=-1),
        )

        # others
        self.energy_force_within_threshold(
            {
                "energy": energy_prediction,
                "forces": force_prediction,
            },
            {
                "energy": energy_target,
                "forces": forces_target,
            },
            batch_idx,
        )

        # now return all metrics
        return {
            "energy_mae": self.energy_mae,
            "forces_mae": self.forces_mae,
            "forces_x_mae": self.forces_x_mae,
            "forces_y_mae": self.forces_y_mae,
            "forces_z_mae": self.forces_z_mae,
            "forces_cos": self.forces_cos,
            "forces_magnitude": self.forces_magnitude,
            "energy_force_within_threshold": self.energy_force_within_threshold,
        }
