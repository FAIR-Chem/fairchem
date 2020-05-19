import os.path
import sys

import ase.io
import numpy as np

from ocpmodels.common.ase_calc import OCP
from ocpmodels.datasets import *
from ocpmodels.trainers import *

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))



if __name__ == "__main__":
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
        "traj": "COCu_ber_50ps_300K.traj",
        "train_size": 10,
        "val_size": 0,
        "test_size": 0,
    }

    optimizer = {
        "batch_size": 32,
        "lr_gamma": 0.1,
        "lr_initial": 0.0003,
        "lr_milestones": [20, 30],
        "max_epochs": 10,
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

    test_traj = ase.io.read("../data/data/COCu_ber_50ps_300K.traj", ":10")
    calc = OCP(trainer)
    calc.train()

    actual_energies = np.array(
        [
            image.get_potential_energy(apply_constraint=False)
            for image in test_traj
        ]
    )
    actual_forces = np.concatenate(
        np.array(
            [image.get_forces(apply_constraint=False) for image in test_traj]
        )
    )

    pred_energies = np.array(
        [calc.get_potential_energy(image) for image in test_traj]
    )
    pred_forces = np.concatenate(
        np.array([calc.get_forces(image) for image in test_traj])
    )

    # Consistency check with reported losses
    energy_mae = np.mean(np.abs(actual_energies - pred_energies))
    force_x_mae = np.mean(np.abs(actual_forces[:, 0] - pred_forces[:, 0]))
    force_y_mae = np.mean(np.abs(actual_forces[:, 1] - pred_forces[:, 1]))
    force_z_mae = np.mean(np.abs(actual_forces[:, 2] - pred_forces[:, 2]))

    print(energy_mae)
    print(force_x_mae)
    print(force_y_mae)
    print(force_z_mae)
