"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import copy
import logging
from typing import List

import submitit

from ocpmodels.common.flags import flags
from ocpmodels.common.utils import (
    build_config,
    create_grid,
    new_trainer_context,
    save_experiment_log,
    setup_logging,
)


class Runner(submitit.helpers.Checkpointable):
    def __init__(self) -> None:
        self.config = None

    def __call__(self, config) -> None:
        with new_trainer_context(args=args, config=config) as ctx:
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
        return submitit.helpers.DelayedSubmission(new_runner, self.config)


if __name__ == "__main__":
    setup_logging()

    parser: argparse.ArgumentParser = flags.get_parser()
    args: argparse.Namespace
    override_args: List[str]
    args, override_args = parser.parse_known_args()
    config = build_config(args, override_args)

    if args.submit:  # Run on cluster
        slurm_add_params = config.get(
            "slurm", None
        )  # additional slurm arguments
        if args.sweep_yml:  # Run grid search
            configs = create_grid(config, args.sweep_yml)
        else:
            configs = [config]

        logging.info(f"Submitting {len(configs)} jobs")
        executor = submitit.AutoExecutor(
            folder=args.logdir / "%j", slurm_max_num_timeout=3
        )
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
        )
        for config in configs:
            config["slurm"] = copy.deepcopy(executor.parameters)
            config["slurm"]["folder"] = str(executor.folder)
        jobs = executor.map_array(Runner(), configs)
        logging.info(
            f"Submitted jobs: {', '.join([job.job_id for job in jobs])}"
        )
        log_file = save_experiment_log(args, jobs, configs)
        logging.info(f"Experiment log saved to: {log_file}")

    else:  # Run locally
        Runner()(config)
