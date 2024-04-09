"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
from pathlib import Path


class Flags:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(
            description="Graph Networks for Electrocatalyst Design"
        )
        self.add_core_args()

    def get_parser(self) -> argparse.ArgumentParser:
        return self.parser

    def add_core_args(self) -> None:
        self.parser.add_argument_group("Core Arguments")
        self.parser.add_argument(
            "--mode",
            choices=["train", "predict", "run-relaxations", "validate"],
            required=True,
            help="Whether to train the model, make predictions, or to run relaxations",
        )
        self.parser.add_argument(
            "--config-yml",
            required=True,
            type=Path,
            help="Path to a config file listing data, model, optim parameters.",
        )
        self.parser.add_argument(
            "--identifier",
            default="",
            type=str,
            help="Experiment identifier to append to checkpoint/log/result directory",
        )
        self.parser.add_argument(
            "--debug",
            action="store_true",
            help="Whether this is a debugging run or not",
        )
        self.parser.add_argument(
            "--run-dir",
            default="./",
            type=str,
            help="Directory to store checkpoint/log/result directory",
        )
        self.parser.add_argument(
            "--print-every",
            default=10,
            type=int,
            help="Log every N iterations (default: 10)",
        )
        self.parser.add_argument(
            "--seed", default=0, type=int, help="Seed for torch, cuda, numpy"
        )
        self.parser.add_argument(
            "--amp", action="store_true", help="Use mixed-precision training"
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
            "--logdir", default="logs", type=Path, help="Where to store logs"
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
            default=13356,
            help="Port on master for DDP",
        )
        self.parser.add_argument(
            "--distributed-backend",
            type=str,
            default="nccl",
            help="Backend for DDP",
        )
        self.parser.add_argument(
            "--local_rank", default=0, type=int, help="Local rank"
        )
        self.parser.add_argument(
            "--no-ddp", action="store_true", help="Do not use DDP"
        )
        self.parser.add_argument(
            "--gp-gpus",
            type=int,
            default=None,
            help="Number of GPUs to split the graph over (only for Graph Parallel training)",
        )


flags = Flags()
