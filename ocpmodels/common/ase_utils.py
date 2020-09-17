import os
import sys
from os import path

import ase.io
import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator as sp
from ase.calculators.vasp import Vasp2
from ase.constraints import FixAtoms
from torch_geometric.data import Batch

from ocdata.vasp import write_vasp_input_files
from ocpmodels.common.efficient_validation.bfgs_torch import BFGS
from ocpmodels.common.efficient_validation.lbfgs_torch import LBFGS, TorchCalc
from ocpmodels.common.meter import mae, mae_ratio, mean_l2_distance
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import radius_graph_pbc
from ocpmodels.datasets.trajectory_lmdb import data_list_collater
from ocpmodels.preprocessing import AtomsToGraphs


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
        predictions = self.trainer.predict(batch, per_image=True)

        self.results["energy"] = predictions["energy"][0]
        self.results["forces"] = predictions["forces"]


def relax_eval(
    batch, model, metric, steps, fmax, relax_opt, return_relaxed_pos,
):
    """
    Evaluation of ML-based relaxations.
    Args:
        batch: object
        model: object
        metric: str
            Evaluation metric to be used.
        steps: int
            Max number of steps in the structure relaxation.
        fmax: float
            Structure relaxation terminates when the max force
            of the system is no bigger than fmax.
        relax_opt: str
            Optimizer and corresponding parameters to be used for structure relaxations.
        return_relaxed_pos: bool
            Whether to return relaxed positions to be written for dft
            evaluation.
    """
    # TODO: Multi-GPU implementation
    batch = batch[0]
    calc = TorchCalc(model)

    true_relaxed_pos = batch.pos_relaxed
    true_relaxed_energy = batch.y_relaxed

    # Run ML-based relaxation
    if relax_opt["name"] == "bfgs":
        dyn = BFGS(batch, calc)
    elif relax_opt["name"] == "lbfgs":
        dyn = LBFGS(batch, calc, memory=relax_opt["memory"])
    else:
        raise ValueError(f"Unknown relax optimizer: {relax_opt}")

    relaxed_batch = dyn.run(fmax=fmax, steps=steps)

    if return_relaxed_pos:
        natoms = batch.natoms.tolist()
        positions = torch.split(batch.pos, natoms)
        relaxed_positions = [pos.tolist() for pos in positions]
    else:
        relaxed_positions = None

    ml_relaxed_energy = relaxed_batch.y.cpu()
    ml_relaxed_pos = relaxed_batch.pos.cpu()

    energy_error = eval(metric)(true_relaxed_energy, ml_relaxed_energy)
    structure_error = torch.mean(
        eval(metric)(ml_relaxed_pos, true_relaxed_pos,)
    )
    return energy_error, structure_error, relaxed_positions


# TODO write intermediate steps into traj file similar to ase optimizers
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
            pbc=True,
        )
        calc = sp(
            atoms=atoms,
            energy=energies[idx],
            forces=forces[idx].cpu().detach().numpy(),
        )
        atoms.set_calculator(calc)
        atoms_objects.append(atoms)

    return atoms_objects
