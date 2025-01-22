"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import hydra
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from fairchem.core.components.runner import Runner


from submitit import AutoExecutor
from submitit.helpers import Checkpointable, DelayedSubmission

from fairchem.core.common import distutils
from fairchem.core.common.utils import get_timestamp_uid, setup_env_vars, setup_imports

# this effects the cli only since the actual job will be run in subprocesses or remoe
logging.basicConfig(level=logging.INFO)


class SchedulerType(str, Enum):
    LOCAL = "local"
    SLURM = "slurm"


class DeviceType(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"


@dataclass
class SchedulerConfig:
    mode: SchedulerType = SchedulerType.LOCAL
    ranks_per_node: int = 1
    num_nodes: int = 1
    slurm: dict = field(
        default_factory=lambda: {
            "mem_gb": 80,  # slurm mem in GB
            "timeout_hr": 72,  # slurm timeout in hours, default to 7 days
            "partition": None,
            "cpus_per_task": 8,
            "qos": None,
            "account": None,
        }
    )


@dataclass
class FairchemJobConfig:
    run_name: str = field(default_factory=lambda: uuid.uuid4().hex.upper()[0:8])
    timestamp_id: str = field(default_factory=lambda: get_timestamp_uid())
    run_dir: str = field(default_factory=lambda: tempfile.TemporaryDirectory().name)
    log_dir: str = "logs"
    device_type: DeviceType = DeviceType.CUDA
    debug: bool = False
    scheduler: SchedulerConfig = field(default_factory=lambda: SchedulerConfig)


class Submitit(Checkpointable):
    def __call__(self, dict_config: DictConfig) -> None:
        self.config = dict_config
        job_config: FairchemJobConfig = OmegaConf.to_object(dict_config.job)
        # TODO: setup_imports is not needed if we stop instantiating models with Registry.
        setup_imports()
        setup_env_vars()
        distutils.setup(map_job_config_to_dist_config(job_config))
        self._init_logger()
        runner: Runner = hydra.utils.instantiate(dict_config.runner)
        runner.load_state()
        runner.fairchem_config = job_config
        runner.run()
        distutils.cleanup()

    def _init_logger(self) -> None:
        # optionally instantiate a singleton wandb logger, intentionally only supporting the new wandb logger
        # don't start logger if in debug mode
        if (
            "logger" in self.config
            and distutils.is_master()
            and not self.config.job.debug
        ):
            # get a partial function from the config and instantiate wandb with it
            logger_initializer = hydra.utils.instantiate(self.config.logger)
            simple_config = OmegaConf.to_container(
                self.config, resolve=True, throw_on_missing=True
            )
            logger_initializer(
                config=simple_config,
                run_id=self.config.cli_args.timestamp_id,
                run_name=self.config.cli_args.identifier,
                log_dir=self.config.cli_args.logdir,
            )

    def checkpoint(self, *args, **kwargs) -> DelayedSubmission:
        # TODO: this is yet to be tested properly
        logging.info("Submitit checkpointing callback is triggered")
        new_runner = Submitit()
        self.runner.save_state()
        logging.info("Submitit checkpointing callback is completed")
        return DelayedSubmission(new_runner, self.config, self.cli_args)


def map_job_config_to_dist_config(job_cfg: FairchemJobConfig) -> dict:
    scheduler_config = job_cfg.scheduler
    return {
        "world_size": scheduler_config.num_nodes * scheduler_config.ranks_per_node,
        "distributed_backend": "gloo"
        if job_cfg.device_type == DeviceType.CPU
        else "nccl",
        "submit": scheduler_config.mode == SchedulerType.SLURM,
        "summit": None,
        "cpu": job_cfg.device_type == DeviceType.CPU,
        "use_cuda_visibile_devices": True,
    }


def get_hydra_config_from_yaml(
    config_yml: str, overrides_args: list[str]
) -> DictConfig:
    # Load the configuration from the file
    os.environ["HYDRA_FULL_ERROR"] = "1"
    config_directory = os.path.dirname(os.path.abspath(config_yml))
    config_name = os.path.basename(config_yml)
    hydra.initialize_config_dir(config_directory, version_base="1.1")
    return hydra.compose(config_name=config_name, overrides=overrides_args)


def runner_wrapper(config: DictConfig):
    Submitit()(config)


def main(
    args: argparse.Namespace | None = None, override_args: list[str] | None = None
):
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config-yml", type=str, required=True)
        args, override_args = parser.parse_known_args()

    cfg = get_hydra_config_from_yaml(args.config_yml, override_args)
    # merge default structured config with job
    cfg = OmegaConf.merge({"job": OmegaConf.structured(FairchemJobConfig)}, cfg)

    log_dir = os.path.join(cfg.job.run_dir, cfg.job.timestamp_id, cfg.job.log_dir)
    os.makedirs(cfg.job.run_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    job_cfg = cfg.job
    scheduler_cfg = cfg.job.scheduler

    logging.info(f"Running fairchemv2 cli with {cfg}")
    if scheduler_cfg.mode == SchedulerType.SLURM:  # Run on cluster
        executor = AutoExecutor(folder=log_dir, slurm_max_num_timeout=3)
        executor.update_parameters(
            name=job_cfg.run_name,
            mem_gb=scheduler_cfg.slurm.mem_gb,
            timeout_min=scheduler_cfg.slurm.timeout_hr * 60,
            slurm_partition=scheduler_cfg.slurm.partition,
            gpus_per_node=scheduler_cfg.ranks_per_node,
            cpus_per_task=scheduler_cfg.slurm.cpus_per_task,
            tasks_per_node=scheduler_cfg.ranks_per_node,
            nodes=scheduler_cfg.num_nodes,
            slurm_qos=scheduler_cfg.slurm.qos,
            slurm_account=scheduler_cfg.slurm.account,
        )
        job = executor.submit(runner_wrapper, cfg)
        logging.info(
            f"Submitted job id: {job_cfg.timestamp_id}, slurm id: {job.job_id}, logs: {job_cfg.log_dir}"
        )
    else:
        from torch.distributed.launcher.api import LaunchConfig, elastic_launch

        if scheduler_cfg.ranks_per_node > 1:
            logging.info(f"Running in local mode with {job_cfg.ranks_per_node} ranks")
            launch_config = LaunchConfig(
                min_nodes=1,
                max_nodes=1,
                nproc_per_node=scheduler_cfg.ranks_per_node,
                rdzv_backend="c10d",
                max_restarts=0,
            )
            elastic_launch(launch_config, runner_wrapper)(cfg)
        else:
            logging.info("Running in local mode without elastic launch")
            distutils.setup_env_local()
            runner_wrapper(cfg)
