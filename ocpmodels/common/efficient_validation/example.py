import argparse
import copy
import glob
import os
import pickle
import random
import sys
import time

import ase.io
import numpy as np
import torch
from ase.optimize import BFGS, LBFGS
from torch import nn

from bfgs_torch import BFGS as BFGS_torch
from lbfgs_torch import LBFGS as LBFGS_torch
from lbfgs_torch import TorchCalc
from ocpmodels.common.ase_utils import OCPCalculator as OCP
from ocpmodels.common.utils import radius_graph_pbc
from ocpmodels.datasets.trajectory_lmdb import data_list_collater
from ocpmodels.preprocessing import AtomsToGraphs
from ocpmodels.trainers import ForcesTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str)
parser.add_argument("--relaxopt", choices=["bfgs", "lbfgs"], default="lbfgs")
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--lbfgs-mem", type=int, default=50)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument(
    "--pbc-graph", action="store_true", help="Flag to use pbc graph"
)
args = parser.parse_args()

task = {
    "dataset": "trajectory_lmdb",
    "description": "Regressing to energies and forces for a trajectory dataset",
    "labels": ["potential energy"],
    "metric": "mae",
    "type": "regression",
    "grad_input": "atomic forces",
    "relax_dataset": {
        "src": args.src,
    },
    "write_pos": True,
    "relaxation_steps": args.steps,
    "relax_opt": {
        "name": args.relaxopt,
        "memory": args.lbfgs_mem,
    },
}

model = {
    "name": "schnet",
    "hidden_channels": 1024,
    "num_filters": 256,
    "num_interactions": 5,
    "num_gaussians": 200,
    "cutoff": 6.0,
    "use_pbc": True,
}

train_dataset = {
    "src": "/home/mshuaibi/baselines-backup/data_backup/1k_train",
    "normalize_labels": False,
}

optimizer = {
    "batch_size": 32,
    "eval_batch_size": args.batch_size,
    "lr_gamma": 0.1,
    "lr_initial": 0.0003,
    "lr_milestones": [20, 30],
    "num_workers": 10,
    "max_epochs": 50,
    "warmup_epochs": 10,
    "warmup_factor": 0.2,
    "force_coefficient": 30,
    "criterion": nn.L1Loss(),
    "num_gpus": 1,
}

identifier = "example"
trainer = ForcesTrainer(
    task=task,
    model=model,
    dataset=train_dataset,
    optimizer=optimizer,
    identifier=identifier,
    print_every=5,
    is_debug=False,
    logger="wandb",
    seed=1,
)

trainer.load_pretrained(
    "/home/mshuaibi/baselines-backup/checkpoints/2020-09-15-13-50-39-schnet_20M_restart_09_15_run8/checkpoint.pt",
    ddp_to_dp=True,
)
trainer.validate_relaxation()
