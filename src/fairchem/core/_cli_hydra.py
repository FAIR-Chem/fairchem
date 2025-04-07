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

    from fairchem.core.components.reducer import Reducer
    from fairchem.core.components.runner import Runner

from submitit import AutoExecutor
from submitit.core.utils import JobPaths, cloudpickle_dump
from submitit.helpers import Checkpointable, DelayedSubmission
from submitit.slurm.slurm import SlurmJobEnvironment

from fairchem.core.common import distutils
from fairchem.core.common.logger import WandBSingletonLogger
from fairchem.core.common.utils import (
    get_cluster_name,
    get_commit_hash,
    get_timestamp_uid,
    setup_env_vars,
    setup_logging,
)

# this effects the cli only since the actual job will be run in subprocesses or remoe
logging.basicConfig(level=logging.INFO)


ALLOWED_TOP_LEVEL_KEYS = {"job", "runner", "reducer"}

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


class RunType(str, Enum):
    RUN = "run"
    REDUCE = "reduce"


@dataclass
class SlurmConfig:
    mem_gb: int = 80
    timeout_hr: int = 168
    cpus_per_task: int = 8
    partition: Optional[str] = None  # noqa: UP007 omegaconf in python 3.9 does not backport annotations
    qos: Optional[str] = None  # noqa: UP007 omegaconf in python 3.9 does not backport annotations
    account: Optional[str] = None  # noqa: UP007 omegaconf in python 3.9 does not backport annotations


@dataclass
class SchedulerConfig:
    mode: SchedulerType = SchedulerType.LOCAL
    ranks_per_node: int = 1
    num_nodes: int = 1
    num_array_jobs: int = 1
    slurm: SlurmConfig = field(default_factory=lambda: SlurmConfig)


@dataclass
class SlurmEnv:
    # reflects the job_id given by submitit (slurm id with array job id and array task id if they exist)
    job_id: Optional[str] = None  # noqa: UP007 omegaconf in python 3.9 does not backport annotations
    # reflects SLURM_JOB_ID only
    raw_job_id: Optional[str] = None  # noqa: UP007 omegaconf in python 3.9 does not backport annotations
    # SLURM_ARRAY_JOB_ID
    array_job_id: Optional[str] = None  # noqa: UP007 omegaconf in python 3.9 does not backport annotations
    # SLURM_ARRAY_TASK_ID
    array_task_id: Optional[str] = None  # noqa: UP007 omegaconf in python 3.9 does not backport annotations
    # reflects SLURM_RESTART_COUNT env variable
    restart_count: Optional[str] = None  # noqa: UP007 omegaconf in python 3.9 does not backport annotations


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
    array_job_num: int = 0
    slurm_env: SlurmEnv = field(default_factory=lambda: SlurmEnv())


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
    logger: Optional[dict] = None  # noqa: UP007 omegaconf in python 3.9 does not backport annotations
    seed: int = 0
    deterministic: bool = False
    runner_state_path: Optional[str] = None  # noqa: UP007 omegaconf in python 3.9 does not backport annotations
    # read-only metadata about the job, not user inputs
    metadata: Optional[Metadata] = None  # noqa: UP007 omegaconf in python 3.9 does not backport annotations
    graph_parallel_group_size: Optional[int] = None  # noqa: UP007

    def __post_init__(self) -> None:
        self.run_dir = os.path.abspath(self.run_dir)
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


def _get_slurm_env() -> SlurmEnv:
    slurm_job_env = SlurmJobEnvironment()
    try:
        slurm_env = SlurmEnv(
            job_id=slurm_job_env.job_id,
            raw_job_id=slurm_job_env.raw_job_id,
            array_job_id=slurm_job_env.array_job_id,
            array_task_id=slurm_job_env.array_task_id,
            restart_count=os.environ.get("SLURM_RESTART_COUNT"),
        )
    except KeyError:
        # slurm environment variables are undefined, running locally
        slurm_env = SlurmEnv()

    return slurm_env


def remove_runner_state_from_submission(log_folder: str, job_id: str) -> None:
    # (HACK) Decouple the job from the runner state by manually modifying it
    # this ensures the saved runner state is not re-submitted in the event of a node failure
    # ie: if the job was started at state t=T, a requeue during node failure would resubmit the job
    # starting at state t=T again without calling the checkpoint callback, losing all progress in between.
    job_path = JobPaths(folder=log_folder, job_id=job_id)
    if os.path.isfile(job_path.submitted_pickle):
        submission_obj = DelayedSubmission.load(job_path.submitted_pickle)
        submission_obj.args[0].job.runner_state_path = None
        cloudpickle_dump(submission_obj, job_path.submitted_pickle)


class Submitit(Checkpointable):
    def __init__(self) -> None:
        self.config = None
        self.runner = None

    def __call__(
        self, dict_config: DictConfig, run_type: RunType = RunType.RUN
    ) -> None:
        self.config = dict_config
        self.run_type = run_type
        # modify the config metadata to add slurm info if they exist
        self.config.job.metadata.slurm_env = _get_slurm_env()

        setup_env_vars()
        setup_logging()

        dist_config = map_job_config_to_dist_config(self.config.job)
        logging.info("Setting up distributed backend...")
        distutils.setup(dist_config)
        distutils.synchronize()
        if (
            distutils.is_master()
            and self.config.job.scheduler.mode == SchedulerType.SLURM
        ):
            # this pickle file is shared across all processes so can only modify this on the main rank
            remove_runner_state_from_submission(
                dict_config.job.metadata.log_dir,
                self.config.job.metadata.slurm_env.job_id,
            )

        if self.config.job.graph_parallel_group_size is not None:
            logging.info("Setting up graph parallel...")
            gp_utils.setup_graph_parallel_groups(
                self.config.job.graph_parallel_group_size,
                dist_config["distributed_backend"],
            )

        self._init_logger()

        _set_seeds(self.config.job.seed)
        if self.config.job.deterministic:
            _set_deterministic_mode()

        logging.info("Calling runner.run() ...")
        if run_type == RunType.RUN:
            self.runner: Runner = hydra.utils.instantiate(self.config.runner)
            self.runner.job_config = self.config.job
            # must call resume state AFTER the runner has been initialized
            self.runner.load_state(self.config.job.runner_state_path)
            self.runner.run()
        elif run_type == RunType.REDUCE:
            self.reducer: Reducer = hydra.utils.instantiate(self.config.reducer)
            self.reducer.job_config = self.config.job
            self.reducer.runner_config = self.config.runner
            # must call resume state AFTER the runner has been initialized
            self.reducer.load_state(self.config.job.runner_state_path)
            self.reducer.reduce()
        else:
            raise ValueError(f"run type {run_type} is not recognized!")

        distutils.cleanup()

    def _init_logger(self) -> None:
        if (
            self.config.job.logger
            and distutils.is_master()
            and not self.config.job.debug
            and self.config.job.metadata.array_job_num == 0
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
        save_path = self.config.job.metadata.preemption_checkpoint_dir
        cfg_copy = self.config.copy()
        # only assign if the save was successful
        cfg_copy.job.runner_state_path = None

        if (
            self.run_type == RunType.RUN
            and self.runner.save_state(save_path, is_preemption=True)
        ) or (
            self.run_type == RunType.REDUCE
            and self.reducer.save_state(save_path, is_preemption=True)
        ):
            cfg_copy.job.runner_state_path = save_path

        if WandBSingletonLogger.initialized():
            WandBSingletonLogger.get_instance().mark_preempting()
        logging.info(
            f"Submitit checkpointing callback is completed, resuming with use the following state: {save_path}"
        )
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


def _runner_wrapper(config: DictConfig, run_type: RunType = RunType.RUN):
    # This is needed when using elastic_launch for local runs since it looks for
    # the __name__ attribute of the function, Submitit.__call__ does not have one
    Submitit()(config, run_type)


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
        if scheduler_cfg.num_array_jobs == 1:
            job = executor.submit(Submitit(), cfg)
            logging.info(
                f"Submitted job id: {cfg.job.timestamp_id}, slurm id: {job.job_id}, logs: {cfg.job.metadata.log_dir}"
            )
            jobs = [job]
        elif scheduler_cfg.num_array_jobs > 1:
            executor.update_parameters(
                slurm_array_parallelism=scheduler_cfg.num_array_jobs,
            )

            jobs = []
            with executor.batch():
                for job_number in range(scheduler_cfg.num_array_jobs):
                    _cfg = cfg.copy()
                    _cfg.job.metadata.array_job_num = job_number
                    job = executor.submit(Submitit(), _cfg)
                    jobs.append(job)
            logging.info(f"Submitted {len(jobs)} jobs: {jobs[0].job_id.split('_')[0]}")

        if "reducer" in cfg:
            job_id = jobs[0].job_id.split("_")[0]
            executor.update_parameters(
                name=f"{cfg.job.run_name}_reduce",
                # set a single node, or do we want the same config as the Runner or a separate JobConfig
                nodes=1,
                slurm_dependency=f"afterok:{job_id}",
                slurm_additional_parameters={
                    "kill-on-invalid-dep": "yes"
                },  # kill the reducer if run fails
            )
            executor.submit(Submitit(), cfg, RunType.REDUCE)
    else:
        from torch.distributed.launcher.api import LaunchConfig, elastic_launch

        assert (
            (scheduler_cfg.num_nodes) <= 1
        ), f"You cannot use more than one node (scheduler_cfg.num_nodes={scheduler_cfg.num_nodes}) in LOCAL mode"
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
            elastic_launch(launch_config, _runner_wrapper)(cfg)
            if "reducer" in cfg:
                elastic_launch(launch_config, _runner_wrapper)(cfg, RunType.REDUCE)
        else:
            logging.info("Running in local mode without elastic launch")
            distutils.setup_env_local()
            Submitit()(cfg)
            if "reducer" in cfg:
                Submitit()(cfg, RunType.REDUCE)
