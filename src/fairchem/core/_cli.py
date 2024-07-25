"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import logging
import os
from typing import TYPE_CHECKING

from submitit import AutoExecutor
from submitit.helpers import Checkpointable, DelayedSubmission
from torch.distributed.launcher.api import LaunchConfig, elastic_launch

from fairchem.core.common.flags import flags
from fairchem.core.common.paths import LOG_DIR_NAME, get_log_dir
from fairchem.core.common.utils import (
    build_config,
    create_grid,
    new_trainer_context,
    save_experiment_job_spec,
    setup_logging,
)

if TYPE_CHECKING:
    import argparse


class Runner(Checkpointable):
    def __init__(self, distributed: bool = False) -> None:
        self.config = None
        self.distributed = distributed

    def __call__(self, config: dict) -> None:
        with new_trainer_context(config=config, distributed=self.distributed) as ctx:
            self.config = ctx.config
            self.task = ctx.task
            self.trainer = ctx.trainer
            self.task.setup(self.trainer)
            self.task.run()

    def checkpoint(self, *args, **kwargs):
        new_runner = Runner(self.distributed)
        self.trainer.save(checkpoint_file="checkpoint.pt", training_state=True)
        self.config["checkpoint"] = self.task.chkpt_path
        self.config["timestamp_id"] = self.trainer.timestamp_id
        if self.trainer.logger is not None:
            self.trainer.logger.mark_preempting()
        return DelayedSubmission(new_runner, self.config)


def runner_wrapper(distributed: bool, config: dict):
    Runner(distributed=distributed)(config)


def main():
    """Run the main fairchem program."""
    setup_logging()

    parser: argparse.ArgumentParser = flags.get_parser()
    args: argparse.Namespace
    override_args: list[str]
    args, override_args = parser.parse_known_args()
    config = build_config(args, override_args)
    os.makedirs(args.run_dir, exist_ok=True)

    if args.submit:  # Run on cluster
        slurm_add_params = config.get("slurm", None)  # additional slurm arguments
        configs = create_grid(config, args.sweep_yml) if args.sweep_yml else [config]

        logging.info(f"Submitting {len(configs)} jobs")
        executor = AutoExecutor(folder=os.path.join(args.run_dir, "%j", LOG_DIR_NAME), slurm_max_num_timeout=3)
        executor.update_parameters(
            name=args.identifier,
            mem_gb=args.slurm_mem,
            timeout_min=args.slurm_timeout * 60,
            slurm_partition=args.slurm_partition,
            gpus_per_node=args.num_gpus,
            cpus_per_task=(config["optim"]["num_workers"] + 1),
            tasks_per_node=(args.num_gpus if args.distributed else 1),
            nodes=args.num_nodes,
            slurm_additional_parameters=slurm_add_params,
            slurm_qos=args.slurm_qos,
            slurm_account=args.slurm_account,
        )
        for config in configs:
            config["slurm"] = copy.deepcopy(executor.parameters)
            config["slurm"]["folder"] = str(executor.folder)
        jobs = executor.map_array(Runner(distributed=args.distributed), configs)
        logging.info(f"Submitted jobs: {', '.join([job.job_id for job in jobs])}")
        exp_spec = save_experiment_job_spec(os.path.join(args.run_dir, "job_specs"), jobs, configs)
        logging.info(f"Experiment spec: {exp_spec}")
        logging.info(f"Experiment log directories: {', '.join([get_log_dir(job.job_id, args.run_dir) for job in jobs])}")

    else:  # Run locally on a single node, n-processes
        if args.distributed:
            logging.info(
                f"Running in distributed local mode with {args.num_gpus} ranks"
            )
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
            elastic_launch(launch_config, runner_wrapper)(args.distributed, config)
        else:
            logging.info("Running in non-distributed local mode")
            assert (
                args.num_gpus == 1
            ), "Can only run with a single gpu in non distributed local mode, use --distributed flag instead if using >1 gpu"
            runner_wrapper(args.distributed, config)


if __name__ == "__main__":
    main()
