import argparse
from pathlib import Path

import torch
from torch import nn

from ocpmodels.common import distutils
from ocpmodels.trainers.forces_trainer import ForcesTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--relaxopt", choices=["bfgs", "lbfgs"], default="lbfgs")
parser.add_argument("--traj-dir", type=Path, default=None)
parser.add_argument("--src", type=Path, default=None)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--lbfgs-maxstep", type=float, default=0.01)
parser.add_argument("--lbfgs-mem", type=int, default=100)
parser.add_argument("--lbfgs-damping", type=float, default=0.25)
parser.add_argument("--lbfgs-alpha", type=float, default=100.0)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--write-pos", action="store_true")
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
    "relax_dataset": {"src": "data/09_29_val_is2rs_lmdb"},
    "relaxation_steps": args.steps,
    "write_pos": args.write_pos,
    "relax_opt": {
        "name": args.relaxopt,
        "maxstep": args.lbfgs_maxstep,
        "memory": args.lbfgs_mem,
        "damping": args.lbfgs_damping,
        "alpha": args.lbfgs_alpha,
        "traj_dir": args.traj_dir,
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
    "src": "/home/mshuaibi/projects/baselines-backup/data_backup/1k_train/",
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

trainer.validate_relaxation()
distutils.cleanup()
