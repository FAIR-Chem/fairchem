import copy
import os
from pathlib import Path
from typing import List

import numpy as np
import pytest

from ocpmodels.common.utils import setup_logging
from ocpmodels.datasets.lmdb_dataset import LmdbDataset
from ocpmodels.trainers import OCPTrainer

from ocpmodels import models  # isort: skip
from ocpmodels.common import logger  # isort: skip
from ocpmodels.models.gemnet.gemnet import GemNetT  # isort: skip

setup_logging()


def test_e2e_s2ef(
    tutorial_dataset_path: Path, snapshot, torch_deterministic: None
) -> None:
    train_src = tutorial_dataset_path / "s2ef/val_20"
    val_src = tutorial_dataset_path / "s2ef/train_100"

    train_dataset = LmdbDataset({"src": train_src})

    target_energies: List[float] = []
    for data in train_dataset:
        target_energies.append(data.y)

    target_mean = np.mean(target_energies)
    target_stdev = np.std(target_energies)

    # Task
    task = {
        "dataset": "trajectory_lmdb",  # dataset used for the S2EF task
        "description": "Regressing to energies and forces for DFT trajectories from OCP",
        "type": "regression",
        "metric": "mae",
        "labels": ["potential energy"],
        "grad_input": "atomic forces",
        "train_on_free_atoms": True,
        "eval_on_free_atoms": True,
    }

    # Model
    model = {
        "name": "gemnet_t",
        "num_spherical": 7,
        "num_radial": 8,
        "num_blocks": 3,
        "emb_size_atom": 8,
        "emb_size_edge": 8,
        "emb_size_trip": 4,
        "emb_size_rbf": 4,
        "emb_size_cbf": 4,
        "emb_size_bil_trip": 8,
        "num_before_skip": 1,
        "num_after_skip": 2,
        "num_concat": 1,
        "num_atom": 3,
        "cutoff": 6.0,
        "max_neighbors": 50,
        "rbf": {"name": "gaussian"},
        "envelope": {
            "name": "polynomial",
            "exponent": 5,
        },
        "cbf": {"name": "spherical_harmonics"},
        "extensive": True,
        "otf_graph": False,
        "output_init": "HeOrthogonal",
        "activation": "silu",
        "scale_file": "configs/s2ef/all/gemnet/scaling_factors/gemnet-dT.json",
        "regress_forces": True,
        "direct_forces": True,
    }

    # Optimizer
    optimizer = {
        "batch_size": 1,  # originally 32
        "eval_batch_size": 1,  # originally 32
        "num_workers": 0,
        "lr_initial": 0.001,
        "optimizer": "AdamW",
        "optimizer_params": {"amsgrad": True},
        "scheduler": "Null",
        "mode": "min",
        "factor": 0.8,
        "patience": 3,
        "max_epochs": 10,  # used for demonstration purposes
        "force_coefficient": 100,
        # "ema_decay": 0.999,
        "clip_grad_norm": 10,
        "loss_energy": "mae",
        "loss_force": "l2mae",
        "eval_every": 200,
    }

    # Dataset
    dataset = [
        {
            "src": train_src,
            "normalize_labels": True,
            "target_mean": target_mean,
            "target_std": target_stdev,
            "grad_target_mean": 0.0,
            "grad_target_std": target_stdev,
        },  # train set
        {"src": val_src},  # val set (optional)
    ]

    trainer = OCPTrainer(
        task=task,
        model=copy.deepcopy(
            model
        ),  # copied for later use, not necessary in practice.
        dataset=dataset,
        optimizer=optimizer,
        identifier="e2e-s2ef-test",
        run_dir=tutorial_dataset_path,  # directory to save results if is_debug=False. Prediction files are saved here so be careful not to override!
        is_debug=False,  # if True, do not save checkpoint, logs, or results
        outputs={},
        loss_fns={},
        eval_metrics={},
        name="s2ef",
        print_every=5,
        seed=1000,  # random seed to use
        logger="tensorboard",  # logger of choice (tensorboard and wandb supported)
        local_rank=0,
        cpu=True,
        # amp=True, # use PyTorch Automatic Mixed Precision (faster training and less memory usage),
    )

    trainer.train()

    checkpoint_path = os.path.join(
        trainer.config["cmd"]["checkpoint_dir"], "best_checkpoint.pt"
    )

    # # Dataset
    dataset.append(
        {"src": train_src},  # test set (optional)
    )

    pretrained_trainer = OCPTrainer(
        task=task,
        model=copy.deepcopy(model),
        dataset=dataset,
        optimizer=optimizer,
        identifier="e2e-s2ef-test-val",
        run_dir=tutorial_dataset_path,  # directory to save results if is_debug=False. Prediction files are saved here so be careful not to override!
        is_debug=False,  # if True, do not save checkpoint, logs, or results
        print_every=10,
        outputs={},
        loss_fns={},
        eval_metrics={},
        name="s2ef",
        seed=990,  # random seed to use
        logger="tensorboard",  # logger of choice (tensorboard and wandb supported)
        local_rank=0,
        cpu=True,
        # amp=True, # use PyTorch Automatic Mixed Precision (faster training and less memory usage),
    )

    pretrained_trainer.load_checkpoint(checkpoint_path=checkpoint_path)

    predictions = pretrained_trainer.predict(
        pretrained_trainer.test_loader,
        results_file="s2ef_results",
        disable_tqdm=False,
    )

    energies = predictions["energy"]
    forces = predictions["forces"]

    # breakpoint()
    assert snapshot == energies.shape
    assert snapshot == forces.shape

    assert snapshot == pytest.approx(energies, abs=2.0)
    assert snapshot == pytest.approx(forces, abs=1.0)
