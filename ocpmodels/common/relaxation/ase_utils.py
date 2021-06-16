"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

"""
Utilities to interface OCP models/trainers with the Atomic Simulation
Environment (ASE)
"""
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator as sp
from ase.constraints import FixAtoms

from ocpmodels.common.utils import radius_graph_pbc
from ocpmodels.datasets.trajectory_lmdb import data_list_collater
from ocpmodels.preprocessing import AtomsToGraphs


def batch_to_atoms(batch):
    n_systems = batch.neighbors.shape[0]
    natoms = batch.natoms.tolist()
    numbers = torch.split(batch.atomic_numbers, natoms)
    fixed = torch.split(batch.fixed, natoms)
    forces = torch.split(batch.force, natoms)
    positions = torch.split(batch.pos, natoms)
    tags = torch.split(batch.tags, natoms)
    cells = batch.cell
    energies = batch.y.tolist()

    atoms_objects = []
    for idx in range(n_systems):
        atoms = Atoms(
            numbers=numbers[idx].tolist(),
            positions=positions[idx].cpu().detach().numpy(),
            tags=tags[idx].tolist(),
            cell=cells[idx].cpu().detach().numpy(),
            constraint=FixAtoms(mask=fixed[idx].tolist()),
            pbc=[True, True, True],
        )
        calc = sp(
            atoms=atoms,
            energy=energies[idx],
            forces=forces[idx].cpu().detach().numpy(),
        )
        atoms.set_calculator(calc)
        atoms_objects.append(atoms)

    return atoms_objects


class OCPCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, trainer, pbc_graph=False):
        """
        OCP-ASE Calculator

        Args:
            trainer: Object
                ML trainer for energy and force predictions.
        """
        Calculator.__init__(self)
        self.trainer = trainer
        self.pbc_graph = pbc_graph
        self.a2g = AtomsToGraphs(
            max_neigh=50,
            radius=6,
            r_energy=False,
            r_forces=False,
            r_distances=False,
        )

    def train(self):
        self.trainer.train()

    def load_pretrained(self, checkpoint_path):
        """
        Load existing trained model

        Args:
            checkpoint_path: string
                Path to trained model
        """
        try:
            self.trainer.load_pretrained(checkpoint_path)
        except NotImplementedError:
            print("Unable to load checkpoint!")

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        data_object = self.a2g.convert(atoms)
        batch = data_list_collater([data_object])
        if self.pbc_graph:
            edge_index, cell_offsets, neighbors = radius_graph_pbc(
                batch, 6, 50, batch.pos.device
            )
            batch.edge_index = edge_index
            batch.cell_offsets = cell_offsets
            batch.neighbors = neighbors

        predictions = self.trainer.predict(batch, per_image=False)
        if self.trainer.name == "s2ef":
            self.results["energy"] = predictions["energy"].item()
            self.results["forces"] = predictions["forces"].cpu().numpy()

        elif self.trainer.name == "is2re":
            self.results["energy"] = predictions["energy"].item()
