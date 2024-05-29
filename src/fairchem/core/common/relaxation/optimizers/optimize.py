"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Code based on ase.optimize
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from ase.calculators.calculator import PropertyNotImplementedError
from ase.optimize.optimize import Optimizable
from fairchem.core.common.relaxation.ase_utils import batch_to_atoms
from fairchem.core.common.utils import radius_graph_pbc

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from torch_geometric.data import Batch
    from fairchem.core.trainers import BaseTrainer


ALL_CHANGES = [
    "pos",
    "atomic_numbers",
    "cell",
    "pbc",
]


def compare_batches(
    batch1: Batch | None,
    batch2: Batch,
    tol: float = 1e-6,
    excluded_properties: list[str] | None = None,
) -> list[str]:
    """Compare properties between two batches

    Args:
        batch1: atoms batch
        batch2: atoms batch
        tol: tolerance used to compare equility of floating point properties
        excluded_properties: list of properties to exclude from comparison

    Returns:
        list of system changes, property names that are differente between batch1 and batch2
    """
    system_changes = []

    if batch1 is None:
        system_changes = ALL_CHANGES
    else:
        properties_to_check = set(ALL_CHANGES)
        if excluded_properties:
            properties_to_check -= set(excluded_properties)

        # Check properties that aren't
        for prop in ALL_CHANGES:
            if prop in properties_to_check:
                properties_to_check.remove(prop)
                if not torch.allclose(
                    getattr(batch1, prop), getattr(batch2, prop), atol=tol
                ):
                    system_changes.append(prop)

    return system_changes


class OptimizableBatch(Optimizable):
    """A Batch version of ase Optimizable Atoms

    This class can be used with ML relaxations in fairchem.core.relaxations.ml_relaxation
    or in ase relaxations classes, i.e. ase.optimize.lbfgs
    """

    ignored_changes: set[str] = {}

    def __init__(
        self, batch: Batch, trainer: BaseTrainer, transform=None, numpy: bool = False
    ):
        """Initialize Optimizable Batch

        Args:
            batch: A batch of atoms graph data
            model: An instance of a BaseTrainer derived class
            transform: graph transform
            numpy: wether to cast results to numpy arrays
        """
        self.batch = batch
        self.cached_batch = None
        self.trainer = trainer
        self.transform = transform
        self.numpy = numpy
        self.results = {}

    def check_state(self, batch: Batch, tol: float = 1e-12):
        """Check for any system changes since last calculation."""
        return compare_batches(
            self.cached_batch,
            batch,
            tol=tol,
            excluded_properties=set(self.ignored_changes),
        )

    def get_property(self, name):
        """Get a predicted property by name."""
        system_changes = self.check_state(self.batch)

        if len(system_changes) > 0:
            self.results = self.trainer.predict(
                self.batch, per_image=False, disable_tqdm=True
            )
            if self.numpy:
                self.results = {
                    key: pred.item() if pred.numel() == 1 else pred.cpu().numpy()
                    for key, pred in self.results.items()
                }
            self.cached_batch = self.batch.clone()

        if name not in self.results:
            raise PropertyNotImplementedError(
                f"{name} not present in this " "calculation"
            )

        return self.results[name]

    def get_positions(self):
        if self.numpy:
            return self.batch.pos.cpu().numpy()

        return self.batch.pos

    def set_positions(self, positions: torch.Tensor | NDArray):
        if isinstance(positions, np.ndarray):
            positions = torch.tensor(
                positions, dtype=torch.float32, device=self.batch.pos.device
            )

        self.batch.pos = positions.to(dtype=torch.float32)

    def get_forces(self, apply_constraint: bool = False):
        forces = self.get_property("forces")
        if apply_constraint:
            fixed_idx = torch.where(self.batch.fixed == 1)[0]
            forces[fixed_idx] = 0
        return forces

    def get_potential_energy(self):
        return self.get_property("energy").sum()

    def get_potential_energies(self):
        return self.get_property("energy")

    def iterimages(self):
        # XXX document purpose of iterimages
        yield self.batch

    def converged(self, forces, fmax):
        if self.numpy:
            return np.linalg.norm(forces, axis=1).max() < fmax

        return torch.linalg.norm(forces, axis=1).max() < fmax

    def get_atoms(self):
        """Get ase Atoms objects corresponding to the batch"""
        return batch_to_atoms(self.batch)

    def update_graph(self):
        """Update the graph if model does not use otf_graph"""
        edge_index, cell_offsets, num_neighbors = radius_graph_pbc(self.batch, 6, 50)
        self.batch.edge_index = edge_index
        self.batch.cell_offsets = cell_offsets
        self.batch.neighbors = num_neighbors
        if self.transform is not None:
            self.batch = self.transform(self.batch)

    def __len__(self):
        # TODO: return 3 * len(self.atoms), because we want the length
        # of this to be the number of DOFs
        return len(self.batch.pos)
