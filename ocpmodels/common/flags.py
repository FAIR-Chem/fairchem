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
            default="train",
            help="Whether to train the model, make predictions, or to run relaxations",
        )
        self.parser.add_argument(
            "--config-yml",
            type=Path,
            help="LEGACY Path to a config file listing data, model, optim parameters.",
        )
        self.parser.add_argument(
            "--config",
            type=str,
            help="Descriptor for the run configuration as '{model}-{task}-{split}'.",
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
            default=-1,
            type=int,
            help="Log every N iterations (default: -1 = end of epoch)",
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
            "--silent",
            action="store_true",
            help="Prevent the trainer from printing some stuff",
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
            "--no_cpus_to_workers",
            action="store_true",
            default=False,
            help="Match dataloader workers to available cpus "
            + "(may be divided by number of GPUs)",
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
            "--note",
            type=str,
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
            "--wandb_project",
            type=str,
            default="ocp-3",
            help="WandB project name to use",
        )
        self.parser.add_argument(
            "--use_pbc",
            type=bool,
            default=True,
            help="Whether to use periodic boundary conditions",
        )
        self.parser.add_argument(
            "--test_ri",
            type=bool,
            default=False,
            help="Test rotation invariance of model",
        )
        self.parser.add_argument(
            "--frame_averaging",
            type=str,
            default="",
            help="Frame averaging method to use",
            choices=["", "2D", "3D", "DA"],
        )
        self.parser.add_argument(
            "--fa_frames",
            type=str,
            default="",
            help="Frame averaging method to use",
            choices=["", "random", "det", "all", "se3-all", "se3-random", "se3-det"],
        )
        self.parser.add_argument(
            "--graph_rewiring",
            type=str,
            help="How to rewire the graph",
            choices=[
                "",
                "remove-tag-0",
                "one-supernode-per-graph",
                "one-supernode-per-atom-type",
                "one-supernode-per-atom-type-dist",
            ],
        )
        self.parser.add_argument(
            "--eval_on_test",
            action="store_true",
            default=False,
            help="Evaluate on test set",
        )
        self.parser.add_argument(
            "--narval",
            action="store_true",
            default=False,
            help="is on Narval DRAC cluster",
        )
        self.parser.add_argument(
            "--cp_data_to_tmpdir",
            type=bool,
            default=False,
            help="Don't copy LMDB data to $SLURM_TMPDIR and work from there",
        )
        self.parser.add_argument(
            "--log_train_every",
            type=int,
            default=100,
            help="Log training loss every n steps",
        )
        self.parser.add_argument(
            "--orion_search_path",
            "-o",
            type=str,
            help="Path to an orion search space yaml file",
        )
        self.parser.add_argument(
            "--unique_exp_name",
            "-u",
            type=str,
            help="Name for this experiment. If the experiment name already exists,"
            + " the search space MUST be the same. If it is not, the job will crash."
            + " If you change the search space, you must change the experiment name.",
        )


flags = Flags()
