import os.path
import sys

import ase.io
import numpy as np
import torch.nn as nn
from ase import Atoms, units
from ase.build import add_adsorbate, fcc100, molecule
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.md import nvtberendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS
from matplotlib import pyplot as plt

from ocpmodels.datasets import TrajectoryDataset
from ocpmodels.trainers import ForcesTrainer


def run_relaxation(calculator, filename, steps=500):
    slab = fcc100("Cu", size=(3, 3, 3))
    ads = molecule("CO")
    add_adsorbate(slab, ads, 4, offset=(1, 1))
    cons = FixAtoms(
        indices=[
            atom.index for atom in slab if (atom.tag == 2 or atom.tag == 3)
        ]
    )
    slab.set_constraint([cons])
    slab.center(vacuum=13.0, axis=2)
    slab.set_pbc(True)
    slab.set_calculator(calculator)
    dyn = BFGS(slab, trajectory=filename)
    dyn.run(fmax=0.01, steps=steps)


if __name__ == "__main__":

    # Generate sample training data
    run_relaxation(
        calculator=EMT(), filename="./COCu_emt_relax.traj", steps=200
    )

    task = {
        "dataset": "trajectory",
        "description": "Regressing to energies and trajectory for a trajectory dataset",
        "labels": ["potential energy"],
        "metric": "mae",
        "type": "regression",
        "grad_input": "atomic trajectory",
        "ml_relax": False,
        "relaxation_steps": 100,
        "relaxation_fmax": 0.01,
    }

    model = {
        "name": "schnet",
        "hidden_channels": 128,
        "num_filters": 128,
        "num_interactions": 3,
        "num_gaussians": 200,
        "cutoff": 6.0,
    }

    src = "./"
    traj = "COCu_emt_relax.traj"
    full_traj = ase.io.read(src + traj, ":")

    dataset = {
        "src": src,
        "traj": traj,
        "train_size": len(full_traj),
        "val_size": 0,
        "test_size": 0,
    }

    optimizer = {
        "batch_size": 32,
        "lr_gamma": 0.1,
        "lr_initial": 0.0003,
        "lr_milestones": [20, 30],
        "max_epochs": 100,
        "warmup_epochs": 10,
        "warmup_factor": 0.2,
        "force_coefficient": 30,
        "criterion": nn.L1Loss(),
    }

    identifier = "CGCNN_COCu_emt_relax"
    trainer = ForcesTrainer(
        task=task,
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        identifier=identifier,
        print_every=5,
        is_debug=True,
        seed=1,
    )

    trainer.train()
    mae_energy, mae_structure_ratio = trainer.validate_relaxation(test_dir="./relax_eval/")
