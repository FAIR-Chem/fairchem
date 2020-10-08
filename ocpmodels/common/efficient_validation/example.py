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
parser.add_argument("--lbfgs-maxstep", type=float, default=0.04)
parser.add_argument("--lbfgs-mem", type=int, default=50)
parser.add_argument("--lbfgs-damping", type=float, default=1.)
parser.add_argument("--lbfgs-alpha", type=float, default=70.)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--write_pos", action="store_true")
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
    "name": "dimenet",
    "cutoff": 6.0,
    "hidden_channels": 128,
    "max_angles_per_image": 50000,
    "num_after_skip": 2,
    "num_before_skip": 1,
    "num_blocks": 2,
    "num_output_layers": 3,
    "num_radial": 6,
    "num_spherical": 7,
    "use_pbc": True,
}

train_dataset = {
    # "src": "/home/mshuaibi/baselines-backup/data_backup/1k_train/",
    "src": "/private/home/mshuaibi/baselines/data/data/ocp_s2ef/train/200k/",
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
    # "/home/mshuaibi/baselines-backup/checkpoints/2020-09-15-13-50-39-schnet_20M_restart_09_15_run8/checkpoint.pt",
    "/checkpoint/abhshkdz/ocp_baselines_run/checkpoints/2020-10-04-23-07-09-dimenet_2M_run1/checkpoint.pt",
)

trainer.validate_relaxation()
distutils.cleanup()
