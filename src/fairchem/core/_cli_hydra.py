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
from enum import Enum
from typing import TYPE_CHECKING

import hydra
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from fairchem.core.components.runner import Runner


from submitit import AutoExecutor
from submitit.helpers import Checkpointable, DelayedSubmission
from torch.distributed.launcher.api import LaunchConfig, elastic_launch

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


class SchedulerConfig:
    scheduler_type: SchedulerType = SchedulerType.LOCAL
    ranks_per_node: int = 1
    num_nodes: int = 1
    slurm: dict = None

    def __post_init__(self):
        defaults = {
            "slurm_mem": 80,  # slurm mem in GB
            "timeout_min": 4320,  # slurm timeout in mins, default to 7 days
            "slurm_partition": None,
            "cpus_per_task": 8,
            "slurm_qos": None,
            "slurm_account": None,
        }
        # fuse defaults with user inputs
        self.slurm = defaults.update(self.slurm)


class FairchemJobConfig:
    def __init__(
        self,
        run_name: str | None = None,
        timestamp_id: str | None = None,
        run_dir: str | None = None,
        log_dir: str | None = None,
        device_type: DeviceType = DeviceType.CUDA,
    ):
        self.run_name = run_name or uuid.uuid4().hex.upper()[0:8]
        self.timestamp_id = timestamp_id or get_timestamp_uid()
        self.device_type = device_type
        if not run_dir:
            self.run_dir = tempfile.TemporaryDirectory().name
        if not log_dir:
            self.log_dir = os.path.join(self.run_dir, self.timestamp_id, "logs")
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)


class Submitit(Checkpointable):
    def __call__(self, dict_config: DictConfig) -> None:
        self.config = dict_config
        # TODO: setup_imports is not needed if we stop instantiating models with Registry.
        setup_imports()
        setup_env_vars()
        distutils.setup(map_cli_args_to_dist_config(dict_config.cli_args))
        self._init_logger()
        runner: Runner = hydra.utils.instantiate(dict_config.runner)
        runner.load_state()
        runner.run()
        distutils.cleanup()

    def _init_logger(self) -> None:
        # optionally instantiate a singleton wandb logger, intentionally only supporting the new wandb logger
        # don't start logger if in debug mode
        if (
            "logger" in self.config
            and distutils.is_master()
            and not self.config.cli_args.debug
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


def map_cli_args_to_dist_config(cli_args: DictConfig) -> dict:
    return {
        "world_size": cli_args.num_nodes * cli_args.num_gpus,
        "distributed_backend": "gloo" if cli_args.cpu else "nccl",
        "submit": cli_args.submit,
        "summit": None,
        "cpu": cli_args.cpu,
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
    job_cfg: FairchemJobConfig = hydra.utils.instantiate(cfg.job)
    scheduler: SchedulerConfig = hydra.utils.instantiate(cfg.scheduler)
    logging.info(f"Running fairchemv2 cli with {job_cfg}, {scheduler}")

    if job_cfg.scheduler_type == SchedulerType.SLURM:  # Run on cluster
        executor = AutoExecutor(folder=job_cfg.log_dir, slurm_max_num_timeout=3)
        executor.update_parameters(
            name=job_cfg.run_name,
            mem_gb=scheduler.slurm.slurm_mem,
            timeout_min=scheduler.slurm.slurm_timeout * 60,
            slurm_partition=scheduler.slurm.slurm_partition,
            gpus_per_node=scheduler.num_gpus,
            cpus_per_task=scheduler.slurm.cpus_per_task,
            tasks_per_node=scheduler.num_gpus,
            nodes=scheduler.num_nodes,
            slurm_qos=scheduler.slurm.slurm_qos,
            slurm_account=scheduler.slurm.slurm_account,
        )
        job = executor.submit(runner_wrapper, cfg)
        logging.info(
            f"Submitted job id: {job_cfg.timestamp_id}, slurm id: {job.job_id}, logs: {job_cfg.log_dir}"
        )
    else:
        if scheduler.num_gpus > 1:
            logging.info(f"Running in local mode with {job_cfg.num_gpus} ranks")
            launch_config = LaunchConfig(
                min_nodes=1,
                max_nodes=1,
                nproc_per_node=scheduler.num_gpus,
                rdzv_backend="c10d",
                max_restarts=0,
            )
            elastic_launch(launch_config, runner_wrapper)(cfg)
        else:
            logging.info("Running in local mode without elastic launch")
            distutils.setup_env_local()
            runner_wrapper(cfg)
