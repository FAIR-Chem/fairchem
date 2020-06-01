import os
import sys

import numpy as np

from ocpmodels.datasets import *
from ocpmodels.trainers import MDTrainer

# import submitit

sys.path.insert(0, os.getcwd())


def main_helper():
    task = {
        "dataset": "ulissigroup_co",
        "description": "Regressing to binding energies for an MD trajectory of CO on Cu",
        "labels": ["binding energy"],
        "metric": "mae",
        "type": "regression",
        # "grad_input": "atomic forces",
        # whether to multiply / scale gradient wrt input
        "grad_input_mult": -1,
        # indexing which attributes in the input vector to compute gradients for.
        # data.x[:, grad_input_start_idx:grad_input_end_idx]
        # "grad_input_start_idx": 92,
        # "grad_input_end_idx": 95,
    }

    model = {
        "name": "cnn3d_local",
        "regress_forces": False,
        "max_atomic_number": 90,
        "display_weights": False,
        "display_base_name": "/",  # change to directory of your choice
    }

    dataset = {
        "src": "data/data/2020_05_21_ulissigroup_co_with_positions/",
        "train_size": 15000,
        "val_size": 2000,
        "test_size": 2000,
        "normalize_labels": True,
    }

    optimizer = {
        "batch_size": 32,
        "lr_gamma": 0.1,
        "lr_initial": 0.001,
        "lr_milestones": [50, 120],
        "max_epochs": 200,
        "warmup_epochs": 1,
        "warmup_factor": 0.2,
    }
    trainer = MDTrainer(
        task=task,
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        identifier="binding_energy_forces_15ktrain_seed2",
        print_every=5,
        is_debug=False,
        seed=2,
    )

    trainer.train()

    dataset_config = {
        "src": "data/data/2020_05_21_ulissigroup_co_with_positions/",
    }
    predictions = trainer.predict(dataset_config)
    np.save(
        os.path.join(trainer.config["cmd"]["results_dir"], "preds.npy"),
        predictions,
    )


if __name__ == "__main__":
    main_helper()

    #  executor = submitit.AutoExecutor(folder="logs")
    #  executor.update_parameters(timeout_min=1, slurm_partition="learnfair")
    #  job = executor.submit(main_helper)
    #  print(job.job_id)
