"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Code based on ase.optimize
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch
from ase.calculators.calculator import PropertyNotImplementedError
from ase.stress import voigt_6_to_full_3x3_stress

from fairchem.core.common.relaxation.ase_utils import batch_to_atoms
from fairchem.core.common.utils import radius_graph_pbc

# unreleased ASE has Optimizable, last released version 3.22.1 does not
# thankfully we can get away with backwards compatibility by creating a dummy
# submit an issue to the ASE repo to make new releases if you find this code ugly :(
try:
    from ase.optimize.optimize import Optimizable
except ImportError:

    class Optimizable:
        pass


if TYPE_CHECKING:
    from collections.abc import Sequence

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
    excluded_properties: set[str] | None = None,
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

    ignored_changes: ClassVar[dict[str]] = {}

    def __init__(
        self,
        batch: Batch,
        trainer: BaseTrainer,
        transform: torch.nn.Module | None = None,
        numpy: bool = False,
    ):
        """Initialize Optimizable Batch

        Args:
            batch: A batch of atoms graph data
            model: An instance of a BaseTrainer derived class
            transform: graph transform
            numpy: whether to cast results to numpy arrays
        """
        self.batch = batch.to(trainer.device)
        self.trainer = trainer
        self.transform = transform
        self.numpy = numpy
        self.cached_batch = None
        self._torch_results = {}
        self.results = {}

    @property
    def device(self):
        return self.trainer.device

    @property
    def batch_indices(self):
        """Get the batch indices specifying which position/force corresponds to which batch."""
        return self.batch.batch

    def check_state(self, batch: Batch, tol: float = 1e-12):
        """Check for any system changes since last calculation."""
        return compare_batches(
            self.cached_batch,
            batch,
            tol=tol,
            excluded_properties=set(self.ignored_changes),
        )

    def get_property(self, name, no_numpy: bool = False):
        """Get a predicted property by name."""
        system_changes = self.check_state(self.batch)

        if len(system_changes) > 0:
            self._torch_results = self.trainer.predict(
                self.batch, per_image=False, disable_tqdm=True
            )
            if self.numpy:
                self.results = {
                    key: pred.item() if pred.numel() == 1 else pred.cpu().numpy()
                    for key, pred in self._torch_results.items()
                }
            else:
                self.results = self._torch_results

            self.cached_batch = self.batch.clone()

        if name not in self.results:
            raise PropertyNotImplementedError(
                f"{name} not present in this " "calculation"
            )

        return self.results[name] if no_numpy is False else self._torch_results[name]

    def get_positions(self):
        """Get the batch positions"""
        return self.batch.pos.cpu().numpy() if self.numpy else self.batch.pos.clone()

    def set_positions(self, positions: torch.Tensor | NDArray):
        "Set the batch positions"
        if isinstance(positions, np.ndarray):
            positions = torch.tensor(positions, dtype=torch.float32, device=self.device)

        self.batch.pos = positions.to(dtype=torch.float32, device=self.device)

    def get_forces(self, apply_constraint: bool = False):
        """Get predicted batch forces."""
        forces = self.get_property("forces")
        if apply_constraint:
            fixed_idx = torch.where(self.batch.fixed == 1)[0]
            forces[fixed_idx] = 0
        return forces

    def get_potential_energy(self, **kwargs):
        """Get predicted energy as the sum of all batch energies."""
        # ASE 3.22.1 expects a check for force_consistent calculations
        if kwargs.get("force_consistent", False) is True:
            raise PropertyNotImplementedError(
                "force_consistent calculations are not implemented"
            )
        return self.get_property("energy").sum()

    def get_potential_energies(self):
        """Get the predicted energy for each system in batch."""
        return self.get_property("energy")

    def get_cells(self):
        """Get batch crystallographic cells."""
        return self.batch.cell

    def set_cells(self, cells: torch.Tensor | NDArray):
        """Set batch cells."""
        assert self.batch.cell.shape == cells.shape, "Cell shape mismatch"
        if isinstance(cells, np.ndarray):
            cells = torch.tensor(cells, dtype=torch.float32, device=self.device)
        self.batch.cell = cells.to(dtype=torch.float32, device=self.device)

    def get_volumes(self):
        """Get a tensor of volumes for each cell in batch"""
        cells = self.get_cells()
        return torch.linalg.det(cells)

    def iterimages(self):
        # XXX document purpose of iterimages - this is just needed to work with ASE optimizers
        yield self.batch

    def converged(self, forces, fmax):
        """Check if norm of all predicted forces are below fmax"""
        if self.numpy:
            return np.linalg.norm(forces, axis=1).max() < fmax

        return torch.linalg.norm(forces, axis=1).max() < fmax

    def get_atoms_list(self):
        """Get ase Atoms objects corresponding to the batch"""
        return batch_to_atoms(self.batch)

    def update_graph(self):
        """Update the graph if model does not use otf_graph."""
        edge_index, cell_offsets, num_neighbors = radius_graph_pbc(self.batch, 6, 50)
        self.batch.edge_index = edge_index
        self.batch.cell_offsets = cell_offsets
        self.batch.neighbors = num_neighbors
        if self.transform is not None:
            self.batch = self.transform(self.batch)

    def __len__(self):
        # TODO: this might be changed in ASE to be 3 * len(self.atoms)
        return len(self.batch.pos)


class OptimizableUnitCellBatch(OptimizableBatch):
    """Modify the supercell and the atom positions in relaxations.

    Based on ase UnitCellFilter to work on data batches
    """

    def __init__(
        self,
        batch: Batch,
        trainer: BaseTrainer,
        transform: torch.nn.Module | None = None,
        numpy: bool = False,
        mask: Sequence[bool] | None = None,
        cell_factor: float | None = None,
        hydrostatic_strain: bool = False,
        constant_volume: bool = False,
        scalar_pressure: float = 0.0,
    ):
        """Create a filter that returns the forces and unit cell stresses together, for simultaneous optimization.

        For full details see:
            E. B. Tadmor, G. S. Smith, N. Bernstein, and E. Kaxiras,
            Phys. Rev. B 59, 235 (1999)

        Args:
            batch: A batch of atoms graph data
            model: An instance of a BaseTrainer derived class
            transform: graph transform
            numpy: whether to cast results to numpy arrays
            mask: a boolean mask specifying which strain components are allowed to relax
            cell_factor:
                Factor by which deformation gradient is multiplied to put
                it on the same scale as the positions when assembling
                the combined position/cell vector. The stress contribution to
                the forces is scaled down by the same factor. This can be thought
                of as a very simple preconditioners. Default is number of atoms
                which gives approximately the correct scaling.
            hydrostatic_strain:
                Constrain the cell by only allowing hydrostatic deformation.
                The virial tensor is replaced by np.diag([np.trace(virial)]*3).
            constant_volume:
                Project out the diagonal elements of the virial tensor to allow
                relaxations at constant volume, e.g. for mapping out an
                energy-volume curve. Note: this only approximately conserves
                the volume and breaks energy/force consistency so can only be
                used with optimizers that do require a line minimisation
                (e.g. FIRE).
            scalar_pressure:
                Applied pressure to use for enthalpy pV term. As above, this
                breaks energy/force consistency.
        """
        super().__init__(batch=batch, trainer=trainer, transform=transform, numpy=numpy)

        self.orig_cells = self.get_cells().clone()
        self.stress = None

        if mask is None:
            mask = torch.eye(3, device=self.device)

        # TODO make sure mask is on GPU
        if mask.shape == (6,):
            self.mask = torch.tensor(
                voigt_6_to_full_3x3_stress(mask.detach().cpu()),
                device=self.device,
            )
        elif mask.shape == (3, 3):
            self.mask = mask
        else:
            raise ValueError("shape of mask should be (3,3) or (6,)")

        if cell_factor is None:
            cell_factor = self.batch.natoms.to(torch.float32).mean().item()

        self.hydrostatic_strain = hydrostatic_strain
        self.constant_volume = constant_volume
        self.pressure = scalar_pressure * torch.eye(3, device=self.device)
        self.cell_factor = cell_factor
        self.stress = None
        self._batch_trace = torch.vmap(torch.trace)
        self._batch_diag = torch.vmap(lambda x: x * torch.eye(3, device=x.device))

    @cached_property
    def batch_indices(self):
        """Get the batch indices specifying which position/force corresponds to which batch.

        We augment this to specify the batch indices for augmented positions and forces.
        """
        augmented_batch = torch.repeat_interleave(
            torch.arange(
                len(self.batch), dtype=self.batch.batch.dtype, device=self.device
            ),
            3,
        )
        return torch.cat([self.batch.batch, augmented_batch])

    def deform_grad(self):
        """Get the cell deformation matrix"""
        return torch.transpose(
            torch.linalg.solve(self.orig_cells, self.get_cells()), 1, 2
        )

    def get_positions(self):
        """Get positions and cell deformation gradient."""
        cur_deform_grad = self.deform_grad()
        natoms = self.batch.num_nodes
        pos = torch.zeros(
            (natoms + 3 * len(self.get_cells()), 3),
            dtype=self.batch.pos.dtype,
            device=self.device,
        )

        # Augmented positions are the self.atoms.positions but without the applied deformation gradient
        pos[:natoms] = torch.linalg.solve(
            cur_deform_grad[self.batch.batch, :, :],
            self.batch.pos.view(-1, 3, 1),
        ).view(-1, 3)

        # cell DOFs are the deformation gradient times a scaling factor
        pos[natoms:] = self.cell_factor * cur_deform_grad.view(-1, 3)
        return pos.cpu().numpy() if self.numpy else pos

    def set_positions(self, positions: torch.Tensor | NDArray):
        """Set positions and cell.

        positions has shape (natoms + ncells * 3, 3).
        the first natoms rows are the positions of the atoms, the last nsystems * three rows are the deformation tensor
        for each cell.
        """
        if isinstance(positions, np.ndarray):
            positions = torch.tensor(positions, dtype=torch.float32, device=self.device)
        else:
            positions = positions.to(dtype=torch.float32, device=self.device)

        natoms = self.batch.num_nodes
        new_atom_positions = positions[:natoms]
        new_deform_grad = positions[natoms:].view(-1, 3, 3) / self.cell_factor

        # TODO check that in fact symmetry is preserved setting cells and positions
        # Set the new cell from the original cell and the new deformation gradient.  Both current and final structures
        # should preserve symmetry.
        new_cells = torch.bmm(self.orig_cells, torch.transpose(new_deform_grad, 1, 2))
        self.set_cells(new_cells)

        # Set the positions from the ones passed in (which are without the deformation gradient applied) and the new
        # deformation gradient. This should also preserve symmetry
        new_atom_positions = torch.bmm(
            new_atom_positions.view(-1, 1, 3),
            torch.transpose(
                new_deform_grad[self.batch.batch, :, :].view(-1, 3, 3), 1, 2
            ),
        )
        super().set_positions(new_atom_positions.view(-1, 3))

    def get_potential_energy(self, **kwargs):
        """
        returns potential energy including enthalpy PV term.
        """
        atoms_energy = super().get_potential_energy(**kwargs)
        return atoms_energy + self.pressure[0, 0] * self.get_volumes().sum()

    def get_forces(self, **kwargs):
        """Get forces and unit cell stress."""
        stress = self.get_property("stress", no_numpy=True).view(-1, 3, 3)
        atom_forces = self.get_property("forces", no_numpy=True)
        volumes = self.get_volumes().view(-1, 1, 1)

        virial = -volumes * stress + self.pressure.view(-1, 3, 3)
        cur_deform_grad = self.deform_grad()
        atom_forces = torch.bmm(
            atom_forces.view(-1, 1, 3),
            cur_deform_grad[self.batch.batch, :, :].view(-1, 3, 3),
        )
        virial = torch.linalg.solve(
            cur_deform_grad, torch.transpose(virial, dim0=1, dim1=2)
        )
        virial = torch.transpose(virial, dim0=1, dim1=2)

        # TODO this does not work yet! maybe _batch_trace gives an issue
        if self.hydrostatic_strain:
            virial = self._batch_diag(self._batch_trace(virial) / 3.0)

        # Zero out components corresponding to fixed lattice elements
        if (self.mask != 1.0).any():
            virial *= self.mask.view(-1, 3, 3)

        if self.constant_volume:
            virial[:, range(3), range(3)] -= self._batch_trace(virial).view(3, -1) / 3.0

        natoms = self.batch.num_nodes
        augmented_forces = torch.zeros(
            (natoms + 3 * len(self.get_cells()), 3),
            device=self.device,
            dtype=atom_forces.dtype,
        )
        augmented_forces[:natoms] = atom_forces.view(-1, 3)
        augmented_forces[natoms:] = virial.view(-1, 3) / self.cell_factor

        self.stress = -virial.view(-1, 9) / volumes.view(-1, 1)
        return augmented_forces.cpu().numpy() if self.numpy else augmented_forces

    def __len__(self):
        return len(self.batch.pos) + 3 * len(self.batch)
