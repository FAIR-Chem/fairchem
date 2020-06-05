import os
import sys

import numpy as np

from ocpmodels.trainers import MDTrainer

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
        "grad_input_start_idx": 0,
        "grad_input_end_idx": 3,
    }

    model = {
        "name": "schnet",
        "hidden_channels": 128,
        "num_filters": 128,
        "num_interactions": 3,
        "num_gaussians": 200,
        "cutoff": 6.0,
    }

    dataset = {
        "src": "data/data/2020_06_03_rattle_emt",
        "traj": "COCu_emt_5images.traj",
        "train_size": 5,
        "val_size": 0,
        "test_size": 0,
        "normalize_labels": True,
    }

    optimizer = {
        "batch_size": 5,
        "lr_gamma": 0.1,
        "lr_initial": 0.001,
        "lr_milestones": [100, 125],
        "max_epochs": 200,
        "warmup_epochs": 50,
        "warmup_factor": 0.2,
        "force_coefficient": 10,
    }

    trainer = MDTrainer(
        task=task,
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        identifier="schnet-debug",
        print_every=1,
        is_debug=False,
        seed=1,
    )

    trainer.train()
    predictions = trainer.predict(dataset, verbose=False, batch_size=5)
    print(predictions["energy"])
