import argparse
import os
from pathlib import Path

import torch
from ase.optimize import BFGS, LBFGS
from torch import nn

from ocpmodels.common import distutils
from ocpmodels.common.relaxation.ase_utils import OCPCalculator, batch_to_atoms
from ocpmodels.datasets.single_point_lmdb import SinglePointLmdbDataset
from ocpmodels.datasets.trajectory_lmdb import data_list_collater
from ocpmodels.trainers.forces_trainer import ForcesTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=Path, default=None)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--local_rank", type=int, default=0)
args = parser.parse_args()

distutils.setup(
    {
        "submit": False,
        "distributed_backend": "nccl",
    }
)

task = {
    "dataset": "trajectory_lmdb",
    "description": "Regressing to energies and forces for a trajectory dataset",
    "labels": ["potential energy"],
    "metric": "mae",
    "type": "regression",
    "grad_input": "atomic forces",
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
    "src": "/home/mshuaibi/projects/baselines-backup/data_backup/1k_train/",
    "normalize_labels": False,
}

optimizer = {
    "batch_size": 32,
    "eval_batch_size": 32,
    "lr_gamma": 0.1,
    "lr_initial": 0.0003,
    "lr_milestones": [20, 30],
    "num_workers": 10,
    "max_epochs": 50,
    "warmup_epochs": 10,
    "warmup_factor": 0.2,
    "force_coefficient": 30,
    "criterion": nn.L1Loss(),
}

identifier = "debug"
trainer = ForcesTrainer(
    task=task,
    model=model,
    dataset=train_dataset,
    optimizer=optimizer,
    identifier=identifier,
    print_every=5,
    is_debug=False,
    seed=1,
    local_rank=args.local_rank,
)

trainer.load_pretrained(
    "/home/mshuaibi/projects/baselines-backup/checkpoints/2020-09-15-13-50-39-schnet_20M_restart_09_15_run8/checkpoint.pt",
)

relax_dataset = SinglePointLmdbDataset(
    {"src": "data/09_29_val_is2rs_lmdb/data.lmdb"}
)
os.makedirs("debug", exist_ok=True)

for data in relax_dataset:
    calc = OCPCalculator(trainer)
    data.y = data.y_init
    id = data.id.item()
    collated_data = data_list_collater([data])
    atoms_object = batch_to_atoms(collated_data)[0]
    atoms_object.set_calculator(calc)
    dyn = LBFGS(atoms_object, trajectory="debug/ase/{}.traj".format(id))
    dyn.run(steps=args.steps, fmax=0)

distutils.cleanup()
