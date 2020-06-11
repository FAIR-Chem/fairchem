import os
import sys

import numpy as np

from ocpmodels.datasets import COCuMD
from ocpmodels.trainers import MDTrainer


def main_helper():
    task = {
        "dataset": "co_cu_md",
        "description": "Regressing to binding energies for an MD trajectory of CO on Cu",
        "labels": ["potential energy"],
        "metric": "mae",
        "type": "regression",
        "grad_input": "atomic forces",
    }

    model = {
        "name": "cnn3d_local",
        "regress_forces": True,
        "max_atomic_number": 90,
        "num_conv1_filters": 16,
        "num_conv2_filters": 32,
        "num_conv3_filters": 32,
        "num_conv4_filters": 32,
        "display_weights": False,
    }

    dataset = {
        "src": "data/data/2020_04_14_muhammed_md",
        "traj": "COCu_DFT_10ps.traj",
        "train_size": 1000,
        "val_size": 1000,
        "test_size": 2000,
        "normalize_labels": True,
    }

    optimizer = {
        "batch_size": 32,
        "lr_gamma": 0.1,
        "lr_initial": 0.001,
        "lr_milestones": [30, 60],
        "max_epochs": 100,
        "warmup_epochs": 1,
        "warmup_factor": 0.2,
    }
    trainer = MDTrainer(
        task=task,
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        identifier="p_energy_with_positions_forces_1ktrain_seed2",
        print_every=5,
        is_debug=False,
        seed=2,
    )

    trainer.train()

    dataset_config = {
        "src": "data/data/2020_04_14_muhammed_md",
        "traj": "COCu_DFT_10ps.traj",
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
