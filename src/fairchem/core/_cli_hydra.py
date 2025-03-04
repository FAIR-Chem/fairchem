"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import tempfile
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationKeyError

from fairchem.core.common import gp_utils

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from fairchem.core.components.runner import Runner

from submitit import AutoExecutor
from submitit.helpers import Checkpointable, DelayedSubmission

from fairchem.core.common import distutils
from fairchem.core.common.logger import WandBSingletonLogger
from fairchem.core.common.utils import (
    get_cluster_name,
    get_commit_hash,
    get_subdirectories_sorted_by_time,
    get_timestamp_uid,
    setup_env_vars,
    setup_logging,
)

# this effects the cli only since the actual job will be run in subprocesses or remoe
logging.basicConfig(level=logging.INFO)


ALLOWED_TOP_LEVEL_KEYS = {"job", "runner"}

LOG_DIR_NAME = "logs"
CHECKPOINT_DIR_NAME = "checkpoints"
RESULTS_DIR = "results"
CONFIG_FILE_NAME = "canonical_config.yaml"
PREEMPTION_STATE_DIR_NAME = "preemption_state"


class SchedulerType(str, Enum):
    LOCAL = "local"
    SLURM = "slurm"


class DeviceType(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"


@dataclass
class SlurmConfig:
    mem_gb: int = 80
    timeout_hr: int = 168
    cpus_per_task: int = 8
    partition: Optional[str] = None  # noqa: UP007 python 3.9 requires Optional still
    qos: Optional[str] = None  # noqa: UP007 python 3.9 requires Optional still
    account: Optional[str] = None  # noqa: UP007 python 3.9 requires Optional still


@dataclass
class SchedulerConfig:
    mode: SchedulerType = SchedulerType.LOCAL
    ranks_per_node: int = 1
    num_nodes: int = 1
    slurm: SlurmConfig = field(default_factory=lambda: SlurmConfig)


@dataclass
class Metadata:
    # read-only metadata about the job, not user inputs
    commit: str
    log_dir: str
    checkpoint_dir: str
    results_dir: str
    config_path: str
    preemption_checkpoint_dir: str
    cluster_name: str


@dataclass
class JobConfig:
    run_name: str = field(
        default_factory=lambda: get_timestamp_uid() + uuid.uuid4().hex.upper()[0:4]
    )
    timestamp_id: str = field(default_factory=lambda: get_timestamp_uid())
    run_dir: str = field(default_factory=lambda: tempfile.TemporaryDirectory().name)
    device_type: DeviceType = DeviceType.CUDA
    debug: bool = False
    scheduler: SchedulerConfig = field(default_factory=lambda: SchedulerConfig)
    logger: Optional[dict] = None  # noqa: UP007 python 3.9 requires Optional still
    seed: int = 0
    deterministic: bool = False
    runner_state_path: Optional[str] = None  # noqa: UP007
    # read-only metadata about the job, not user inputs
    metadata: Optional[Metadata] = None  # noqa: UP007
    graph_parallel_group_size: Optional[int] = None  # noqa: UP007

    def __post_init__(self) -> None:
        self.metadata = Metadata(
            commit=get_commit_hash(),
            log_dir=os.path.join(self.run_dir, self.timestamp_id, LOG_DIR_NAME),
            checkpoint_dir=os.path.join(
                self.run_dir, self.timestamp_id, CHECKPOINT_DIR_NAME
            ),
            results_dir=os.path.join(self.run_dir, self.timestamp_id, RESULTS_DIR),
            config_path=os.path.join(self.run_dir, self.timestamp_id, CONFIG_FILE_NAME),
            preemption_checkpoint_dir=os.path.join(
                self.run_dir,
                self.timestamp_id,
                CHECKPOINT_DIR_NAME,
                PREEMPTION_STATE_DIR_NAME,
            ),
            cluster_name=get_cluster_name(),
        )


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _set_deterministic_mode() -> None:
    # this is required for full cuda deterministic mode
    logging.info("Setting deterministic mode!")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


class Submitit(Checkpointable):
    def __init__(self) -> None:
        self.config = None
        self.runner = None

    def __call__(self, dict_config: DictConfig) -> None:
        self.config = dict_config
        # TODO also load job config here
        setup_env_vars()
        setup_logging()

        dist_config = map_job_config_to_dist_config(self.config.job)
        distutils.setup(dist_config)
        if self.config.job.graph_parallel_group_size is not None:
            gp_utils.setup_graph_parallel_groups(
                self.config.job.graph_parallel_group_size,
                dist_config["distributed_backend"],
            )

        self._init_logger()
        _set_seeds(self.config.job.seed)
        if self.config.job.deterministic:
            _set_deterministic_mode()

        self.runner: Runner = hydra.utils.instantiate(dict_config.runner)
        self.runner.config = self.config
        # must call resume state AFTER the runner has been initialized
        if self.config.job.runner_state_path:
            self.runner.load_state(self.config.job.runner_state_path)
        self.runner.run()
        distutils.cleanup()

    def _init_logger(self) -> None:
        if (
            self.config.job.logger
            and distutils.is_master()
            and not self.config.job.debug
        ):
            # get a partial function from the config and instantiate wandb with it
            # currently code assumes that we only use the WandBSingletonLogger
            logger_initializer = hydra.utils.instantiate(self.config.job.logger)
            simple_config = OmegaConf.to_container(
                self.config, resolve=True, throw_on_missing=True
            )
            logger_initializer(
                config=simple_config,
                run_id=self.config.job.timestamp_id,
                run_name=self.config.job.run_name,
                log_dir=self.config.job.metadata.log_dir,
            )

    def checkpoint(self, *args, **kwargs) -> DelayedSubmission:
        logging.error("Submitit checkpointing callback is triggered")
        # TODO: preemption state saving doesn't work with DCP because submitit only calls checkpoint
        # on rank0, which will cause the system to deadlock.
        # save_path = self.config.job.metadata.preemption_checkpoint_dir
        # self.runner.save_state(save_path)
        # For now, use the last found checkpoint
        cfg_copy = self.config.copy()
        ckpt_dirs_time = get_subdirectories_sorted_by_time(
            self.config.job.metadata.checkpoint_dir
        )
        if len(ckpt_dirs_time) > 0:
            # pick the lastest one
            cfg_copy.job.runner_state_path = ckpt_dirs_time[-1][0]
            logging.info(
                f"Job will resume using the state found at: {cfg_copy.job.runner_state_path}"
            )
        if WandBSingletonLogger.initialized():
            WandBSingletonLogger.get_instance().mark_preempting()
        logging.info("Submitit checkpointing callback is completed")
        return DelayedSubmission(Submitit(), cfg_copy)


def map_job_config_to_dist_config(job_cfg: JobConfig) -> dict:
    scheduler_config = job_cfg.scheduler
    return {
        "world_size": scheduler_config.num_nodes * scheduler_config.ranks_per_node,
        "distributed_backend": (
            "gloo" if job_cfg.device_type == DeviceType.CPU else "nccl"
        ),
        "submit": scheduler_config.mode == SchedulerType.SLURM,
        "summit": None,
        "cpu": job_cfg.device_type == DeviceType.CPU,
        "use_cuda_visibile_devices": True,
    }


def get_canonical_config(config: DictConfig) -> DictConfig:
    # manually initialize metadata, because OmegaConf currently doesn't call __post_init__ on dataclasses
    job = OmegaConf.to_object(config.job)
    job.__post_init__()
    config.job = job
    # check that each key other than the allowed top level keys are used in config
    # find all top level keys are not in the allowed set
    all_keys = set(config.keys()).difference(ALLOWED_TOP_LEVEL_KEYS)
    used_keys = set()
    for key in all_keys:
        # make a copy of all keys except the key in question
        copy_cfg = OmegaConf.create({k: v for k, v in config.items() if k != key})
        try:
            OmegaConf.resolve(copy_cfg)
        except InterpolationKeyError:
            # if this error is thrown, this means the key was actually required
            used_keys.add(key)

    unused_keys = all_keys.difference(used_keys)
    if unused_keys != set():
        raise ValueError(
            f"Found unused keys in the config: {unused_keys}, please remove them!, only keys other than {ALLOWED_TOP_LEVEL_KEYS} or ones that are used as variables are allowed."
        )

    # resolve the config to fully replace the variables and delete all top level keys except for the ALLOWED_TOP_LEVEL_KEYS
    OmegaConf.resolve(config)
    return OmegaConf.create(
        {k: v for k, v in config.items() if k in ALLOWED_TOP_LEVEL_KEYS}
    )


def get_hydra_config_from_yaml(
    config_yml: str, overrides_args: list[str]
) -> DictConfig:
    # Load the configuration from the file
    os.environ["HYDRA_FULL_ERROR"] = "1"
    config_directory = os.path.dirname(os.path.abspath(config_yml))
    config_name = os.path.basename(config_yml)
    hydra.initialize_config_dir(config_directory, version_base="1.1")
    cfg = hydra.compose(config_name=config_name, overrides=overrides_args)
    # merge default structured config with initialized job object
    cfg = OmegaConf.merge({"job": OmegaConf.structured(JobConfig)}, cfg)
    # canonicalize config (remove top level keys that just used replacing variables)
    return get_canonical_config(cfg)


def runner_wrapper(config: DictConfig):
    Submitit()(config)


def main(
    args: argparse.Namespace | None = None, override_args: list[str] | None = None
):
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("-c", "--config", type=str, required=True)
        args, override_args = parser.parse_known_args()

    cfg = get_hydra_config_from_yaml(args.config, override_args)
    log_dir = cfg.job.metadata.log_dir
    os.makedirs(cfg.job.run_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    OmegaConf.save(cfg, cfg.job.metadata.config_path)
    logging.info(f"saved canonical config to {cfg.job.metadata.config_path}")

    scheduler_cfg = cfg.job.scheduler
    logging.info(f"Running fairchemv2 cli with {cfg}")
    if scheduler_cfg.mode == SchedulerType.SLURM:  # Run on cluster
        assert (
            os.getenv("SLURM_SUBMIT_HOST") is None
        ), "SLURM DID NOT SUBMIT JOB!! Please do not submit jobs from an active slurm job (srun or otherwise)"
        executor = AutoExecutor(folder=log_dir, slurm_max_num_timeout=3)
        executor.update_parameters(
            name=cfg.job.run_name,
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
        job = executor.submit(Submitit(), cfg)
        logging.info(
            f"Submitted job id: {cfg.job.timestamp_id}, slurm id: {job.job_id}, logs: {cfg.job.metadata.log_dir}"
        )
    else:
        from torch.distributed.launcher.api import LaunchConfig, elastic_launch

        if scheduler_cfg.ranks_per_node > 1:
            logging.info(
                f"Running in local mode with {scheduler_cfg.ranks_per_node} ranks"
            )
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
