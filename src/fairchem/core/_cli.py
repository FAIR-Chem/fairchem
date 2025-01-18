"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

from submitit import AutoExecutor
from submitit.helpers import Checkpointable, DelayedSubmission
from torch.distributed.launcher.api import LaunchConfig, elastic_launch

from fairchem.core.common import distutils
from fairchem.core.common.flags import flags
from fairchem.core.common.utils import (
    build_config,
    create_grid,
    new_trainer_context,
    save_experiment_log,
    setup_logging,
)

if TYPE_CHECKING:
    import argparse


class Runner(Checkpointable):
    def __init__(self) -> None:
        self.config = None

    def __call__(self, config: dict) -> None:
        with new_trainer_context(config=config) as ctx:
            self.config = ctx.config
            self.task = ctx.task
            self.trainer = ctx.trainer
            self.task.setup(self.trainer)
            self.task.run()

    def checkpoint(self, *args, **kwargs):
        new_runner = Runner()
        self.trainer.save(checkpoint_file="checkpoint.pt", training_state=True)
        self.config["checkpoint"] = self.task.chkpt_path
        self.config["timestamp_id"] = self.trainer.timestamp_id
        if self.trainer.logger is not None:
            self.trainer.logger.mark_preempting()
        logging.info(
            f'Checkpointing callback is triggered, checkpoint saved to: {self.config["checkpoint"]}, timestamp_id: {self.config["timestamp_id"]}'
        )
        return DelayedSubmission(new_runner, self.config)


def runner_wrapper(config: dict):
    Runner()(config)


def main(
    args: argparse.Namespace | None = None, override_args: list[str] | None = None
):
    """Run the main fairchem program."""
    setup_logging()

    if args is None:
        parser: argparse.ArgumentParser = flags.get_parser()
        args, override_args = parser.parse_known_args()

    # TODO: rename num_gpus -> num_ranks everywhere
    assert (
        args.num_gpus > 0
    ), "num_gpus is used to determine number ranks, so it must be at least 1"
    config = build_config(args, override_args)

    if args.submit:  # Run on cluster
        slurm_add_params = config.get("slurm", None)  # additional slurm arguments
        configs = create_grid(config, args.sweep_yml) if args.sweep_yml else [config]

        logging.info(f"Submitting {len(configs)} jobs")
        executor = AutoExecutor(folder=args.logdir / "%j", slurm_max_num_timeout=3)
        executor.update_parameters(
            name=args.identifier,
            mem_gb=args.slurm_mem,
            timeout_min=args.slurm_timeout * 60,
            slurm_partition=args.slurm_partition,
            gpus_per_node=args.num_gpus,
            cpus_per_task=(config["optim"]["num_workers"] + 1),
            tasks_per_node=args.num_gpus,
            nodes=args.num_nodes,
            slurm_additional_parameters=slurm_add_params,
            slurm_qos=args.slurm_qos,
            slurm_account=args.slurm_account,
        )
        for config in configs:
            config["slurm"] = copy.deepcopy(executor.parameters)
            config["slurm"]["folder"] = str(executor.folder)
        jobs = executor.map_array(Runner(), configs)
        logging.info(f"Submitted jobs: {', '.join([job.job_id for job in jobs])}")
        log_file = save_experiment_log(args, jobs, configs)
        logging.info(f"Experiment log saved to: {log_file}")

    else:  # Run locally on a single node, n-processes
        if args.num_gpus > 1:
            logging.info(f"Running in local mode with {args.num_gpus} ranks")
            # HACK to disable multiprocess dataloading in local mode
            # there is an open issue where LMDB's environment cannot be pickled and used
            # during torch multiprocessing https://github.com/pytorch/examples/issues/526
            if "optim" in config and "num_workers" in config["optim"]:
                config["optim"]["num_workers"] = 0
                logging.info(
                    "WARNING: running in local mode, setting dataloading num_workers to 0, see https://github.com/pytorch/examples/issues/526"
                )

            launch_config = LaunchConfig(
                min_nodes=1,
                max_nodes=1,
                nproc_per_node=args.num_gpus,
                rdzv_backend="c10d",
                max_restarts=0,
            )
            elastic_launch(launch_config, runner_wrapper)(config)
        else:
            logging.info(
                "Running in local mode without elastic launch (single gpu only)"
            )
            distutils.setup_env_local()
            runner_wrapper(config)


if __name__ == "__main__":
    main()
