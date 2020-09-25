import argparse

import torch
from torch import nn

from ocpmodels.common import distutils
from ocpmodels.trainers.dist_forces_trainer import DistributedForcesTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--relaxopt", choices=["bfgs", "lbfgs"], default="lbfgs")
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--lbfgs-mem", type=int, default=50)
parser.add_argument("--steps", type=int, default=300)
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
    "relax_dataset": {"src": "data/init_to_relaxed/1k/train"},
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
    "num_interactions": 3,
    "num_gaussians": 200,
    "cutoff": 6.0,
    "use_pbc": True,
}

train_dataset = {
    "src": "data/init_to_relaxed/1k/train",
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
}

identifier = "debug"
trainer = DistributedForcesTrainer(
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

checkpoint = torch.load(
    "/private/home/mshuaibi/baselines/expts/ocp_expts/pre_final/ocp20M_08_16/checkpoints/2020-08-16-21-53-06-ocp20Mv6_schnet_lr0.0001_ch1024_fltr256_gauss200_layrs3_pbc/checkpoint.pt",
    map_location=f"cuda:{args.local_rank}",
)
trainer.model.module.load_state_dict(checkpoint["state_dict"])

trainer.validate_relaxation()
