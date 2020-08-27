import ase.io
import numpy as np
import torch
from ase.optimize import BFGS
from torch import nn

from bfgs_torch import BFGS as BFGS_torch
from bfgs_torch import TorchCalc
from ocpmodels.trainers import ForcesTrainer

task = {
    "dataset": "trajectory_lmdb",
    "description": "Regressing to energies and forces for a trajectory dataset",
    "labels": ["potential energy"],
    "metric": "mae",
    "type": "regression",
    "grad_input": "atomic forces",
    "relax_dataset": {
        "src": "/private/home/mshuaibi/baselines/ocpmodels/common/efficient_validation/relax"
    },
}

model = {
    "name": "schnet",
    "hidden_channels": 128,
    "num_filters": 128,
    "num_interactions": 3,
    "num_gaussians": 200,
    "cutoff": 6.0,
    "use_pbc": False,
}

train_dataset = {
    "src": "/private/home/mshuaibi/baselines/ocpmodels/common/efficient_validation/train",
    "normalize_labels": False,
}

optimizer = {
    "batch_size": 32,
    "lr_gamma": 0.1,
    "lr_initial": 0.0003,
    "lr_milestones": [20, 30],
    "num_workers": 32,
    "max_epochs": 50,
    "warmup_epochs": 10,
    "warmup_factor": 0.2,
    "force_coefficient": 30,
    "criterion": nn.L1Loss(),
}

identifier = "water_example"
trainer = ForcesTrainer(
    task=task,
    model=model,
    dataset=train_dataset,
    optimizer=optimizer,
    identifier=identifier,
    print_every=5,
    is_debug=True,
    seed=1,
)

trainer.load_pretrained(
    "/private/home/mshuaibi/baselines/ocpmodels/common/efficient_validation/checkpoint.pt"
)
trainer.validate_relaxation()
