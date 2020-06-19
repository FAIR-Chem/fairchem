import os
import sys

import numpy as np

from ocpmodels.trainers import ForcesTrainer

if __name__ == "__main__":
    task = {
        "dataset": "trajectory",
        "description": "Regressing to binding energies for an MD trajectory of CO on Cu",
        "labels": ["potential energy"],
        "metric": "mae",
        "type": "regression",
        "grad_input": "atomic forces",
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

    trainer = ForcesTrainer(
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
