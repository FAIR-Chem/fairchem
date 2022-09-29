"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import os
from pathlib import Path


class Flags:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Graph Networks for Electrocatalyst Design"
        )
        self.add_core_args()

    def get_parser(self):
        return self.parser

    def add_core_args(self):
        self.parser.add_argument_group("Core Arguments")
        self.parser.add_argument(
            "--mode",
            choices=["train", "predict", "run-relaxations", "validate"],
            help="Whether to train the model, make predictions, or to run relaxations",
        )
        self.parser.add_argument(
            "--config-yml",
            type=Path,
            help="Path to a config file listing data, model, optim parameters.",
        )
        self.parser.add_argument(
            "--wandb_name",
            default="",
            type=str,
            help="Experiment identifier to use as wandb name",
        )
        self.parser.add_argument(
            "--is_debug",
            action="store_true",
            help="Whether this is a debugging run or not",
            default=False,
        )
        self.parser.add_argument(
            "--is_hpo",
            action="store_true",
            help="Whether this is a HPO run or not",
            default=False,
        )
        self.parser.add_argument(
            "--run-dir",
            default="$SCRATCH/ocp/runs/$SLURM_JOB_ID",
            type=str,
            help="Directory to store checkpoint/log/result directory",
        )
        self.parser.add_argument(
            "--print-every",
            default=1000,
            type=int,
            help="Log every N iterations (default: 10)",
        )
        self.parser.add_argument(
            "--seed", default=0, type=int, help="Seed for torch, cuda, numpy"
        )
        self.parser.add_argument(
            "--amp",
            action="store_true",
            help="Use mixed-precision training",
            default=False,
        )
        self.parser.add_argument(
            "--checkpoint", type=str, help="Model checkpoint to load"
        )
        self.parser.add_argument(
            "--timestamp-id",
            default=None,
            type=str,
            help="Override time stamp ID. "
            "Useful for seamlessly continuing model training in logger.",
        )
        # Cluster args
        self.parser.add_argument(
            "--sweep-yml",
            default=None,
            type=Path,
            help="Path to a config file with parameter sweeps",
        )
        self.parser.add_argument(
            "--submit", action="store_true", help="Submit job to cluster"
        )
        self.parser.add_argument(
            "--summit", action="store_true", help="Running on Summit cluster"
        )
        self.parser.add_argument(
            "--logdir",
            default="$SCRATCH/ocp/runs/$SLURM_JOB_ID",
            type=Path,
            help="Where to store logs",
        )
        self.parser.add_argument(
            "--slurm-partition",
            default="ocp",
            type=str,
            help="Name of partition",
        )
        self.parser.add_argument(
            "--slurm-mem", default=80, type=int, help="Memory (in gigabytes)"
        )
        self.parser.add_argument(
            "--slurm-timeout", default=72, type=int, help="Time (in hours)"
        )
        self.parser.add_argument(
            "--num-gpus", default=1, type=int, help="Number of GPUs to request"
        )
        self.parser.add_argument(
            "--distributed", action="store_true", help="Run with DDP"
        )
        self.parser.add_argument(
            "--cpu", action="store_true", help="Run CPU only training"
        )
        self.parser.add_argument(
            "--num-nodes",
            default=1,
            type=int,
            help="Number of Nodes to request",
        )
        self.parser.add_argument(
            "--distributed-port",
            type=int,
            default=os.environ.get("MASTER_PORT", 13356),
            help="Port on master for DDP",
        )
        self.parser.add_argument(
            "--distributed-backend",
            type=str,
            default="nccl",
            help="Backend for DDP",
        )
        self.parser.add_argument("--local_rank", default=0, type=int, help="Local rank")
        # Additional arguments
        self.parser.add_argument(
            "--new_gnn",
            action="store_false",
            help="Whether to use original GNN models or modified ones",
        )
        self.parser.add_argument(
            "--fa",
            default=False,
            choices=[False, "2D", "3D"],
            help="Specify which frame averaging method to use",
        )
        self.parser.add_argument(
            "--note",
            type=str,
            default="",
            help="Note describing this run to be added to the logger",
        )
        self.parser.add_argument(
            "--logger",
            type=str,
            default="wandb",
            help="Logger to use. Options: [wandb, tensorboard, dummy]",
            choices=["wandb", "tensorboard", "dummy"],
        )
        self.parser.add_argument(
            "--wandb_tags",
            type=str,
            default="",
            help="Comma-separated tags for wandb",
        )
        self.parser.add_argument(
            "--test_ri",
            type=bool,
            default=False,
            help="Test rotation invariance of model",
        )
        self.parser.add_argument(
            "--print_every",
            type=int,
            default=100,
            help="Printing frequency (in steps)",
        )
        self.parser.add_argument(
            "--frame_averaging",
            type=str,
            default="",
            help="Frame averaging method to use",
            choices=["", "2D", "3D"],  # @AlDu -> update
        )
        self.parser.add_argument(
            "--choice_fa",
            type=str,
            default="",
            help="Frame averaging method to use",
            choices=["random", "e3", "det", "all"],  # @AlDu -> check
        )


flags = Flags()
