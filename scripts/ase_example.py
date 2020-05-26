import os.path
import sys

import ase.io
import numpy as np
from ase import Atoms, units
from ase.build import add_adsorbate, fcc100, molecule
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.md import nvtberendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from ocpmodels.common.ase_calc import OCP
from ocpmodels.datasets import *
from ocpmodels.trainers import *

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def run_md(calculator, steps, filename):
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
    slab.wrap(pbc=[True] * 3)
    slab.set_calculator(calculator)
    np.random.seed(1)
    MaxwellBoltzmannDistribution(slab, 300 * units.kB)
    dyn = nvtberendsen.NVTBerendsen(
        slab, 1 * units.fs, 300, taut=300 * units.fs
    )
    traj = ase.io.Trajectory(filename, "w", slab)
    dyn.attach(traj.write, interval=1)
    dyn.run(steps)


if __name__ == "__main__":
    # Generate training data
    run_md(
        calculator=EMT(), steps=5000, filename="../data/data/COCu_emt_5k.traj"
    )

    task = {
        "dataset": "co_cu_md",
        "description": "Regressing to binding energies for an MD trajectory of CO on Cu",
        "labels": ["potential energy"],
        "metric": "mae",
        "type": "regression",
        "grad_input": "atomic forces",
        # whether to multiply / scale gradient wrt input
        "grad_input_mult": -1,
        # indexing which attributes in the input vector to compute gradients for.
        # data.x[:, grad_input_start_idx:grad_input_end_idx]
        "grad_input_start_idx": 92,
        "grad_input_end_idx": 95,
    }

    model = {
        "name": "cgcnn",
        "atom_embedding_size": 32,
        "fc_feat_size": 64,
        "num_fc_layers": 2,
        "num_graph_conv_layers": 3,
    }

    dataset = {
        "src": "../data/data/",
        "traj": "COCu_emt_5k.traj",
        "train_size": 3000,
        "val_size": 1000,
        "test_size": 1000,
    }

    optimizer = {
        "batch_size": 32,
        "lr_gamma": 0.1,
        "lr_initial": 0.0003,
        "lr_milestones": [20, 30],
        "max_epochs": 100,
        "warmup_epochs": 10,
        "warmup_factor": 0.2,
    }

    trainer = MDTrainer(
        task=task,
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        identifier="p_energy_forces_emt_traj_1ktrain_seed1",
        print_every=5,
        is_debug=True,
        seed=1,
    )

    gnn_calc = OCP(trainer)
    gnn_calc.train()

    # Replicate MD with trained model
    run_md(calculator=gnn_calc, steps=5000, filename="./gnn_md.traj")
