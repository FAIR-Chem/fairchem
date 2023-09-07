"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import copy
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import submitit

from ocpmodels.common.utils import (
    create_grid,
    new_trainer_context,
    save_experiment_log,
)


class Runner(submitit.helpers.Checkpointable):
    def __init__(self) -> None:
        self.config = None

    def __call__(self, config, distributed) -> None:
        with new_trainer_context(
            config=config, distributed=distributed
        ) as ctx:
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


def run_with_config(
    config: Dict[str, Any],
    sweep_yml: Optional[Path],
    submit: bool,
    logdir: str,
    identifier: str,
    slurm_mem: int,
    slurm_timeout: int,
    slurm_partition: str,
    num_gpus: int,
    num_nodes: int,
    distributed: bool,
) -> None:
    if submit:  # Run on cluster
        slurm_add_params = config.get(
            "slurm", None
        )  # additional slurm arguments
        if sweep_yml:  # Run grid search
            configs = create_grid(config, str(sweep_yml))
        else:
            configs = [config]

        logging.info(f"Submitting {len(configs)} jobs")
        executor = submitit.AutoExecutor(
            folder=logdir / "%j",
            slurm_max_num_timeout=3,
        )
        executor.update_parameters(
            name=identifier,
            mem_gb=slurm_mem,
            timeout_min=slurm_timeout * 60,
            slurm_partition=slurm_partition,
            gpus_per_node=num_gpus,
            cpus_per_task=(config["optim"]["num_workers"] + 1),
            tasks_per_node=(num_gpus if distributed else 1),
            nodes=num_nodes,
            slurm_additional_parameters=slurm_add_params,
        )
        for config in configs:
            config["slurm"] = copy.deepcopy(executor.parameters)
            config["slurm"]["folder"] = str(executor.folder)
        jobs = executor.map_array(
            Runner(), configs, [distributed] * len(configs)
        )
        logging.info(
            f"Submitted jobs: {', '.join([job.job_id for job in jobs])}"
        )
        log_file = save_experiment_log(logdir, jobs, configs)
        logging.info(f"Experiment log saved to: {log_file}")

    else:  # Run locally
        Runner()(config, distributed)
