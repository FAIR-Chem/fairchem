"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import hydra

if TYPE_CHECKING:
    import argparse

    from omegaconf import DictConfig

    from fairchem.core.components.runner import Runner


from submitit import AutoExecutor
from submitit.helpers import Checkpointable, DelayedSubmission
from torch.distributed.launcher.api import LaunchConfig, elastic_launch

from fairchem.core.common import distutils
from fairchem.core.common.flags import flags
from fairchem.core.common.utils import get_timestamp_uid, setup_env_vars, setup_imports

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Submitit(Checkpointable):
    def __call__(self, dict_config: DictConfig, cli_args: argparse.Namespace) -> None:
        self.config = dict_config
        self.cli_args = cli_args
        # TODO: setup_imports is not needed if we stop instantiating models with Registry.
        setup_imports()
        setup_env_vars()
        try:
            distutils.setup(map_cli_args_to_dist_config(cli_args))
            self.runner: Runner = hydra.utils.instantiate(dict_config.runner)
            self.runner.load_state()
            self.runner.run()
        finally:
            distutils.cleanup()

    def checkpoint(self, *args, **kwargs):
        logging.info("Submitit checkpointing callback is triggered")
        new_runner = Submitit()
        self.runner.save_state()
        logging.info("Submitit checkpointing callback is completed")
        return DelayedSubmission(new_runner, self.config, self.cli_args)


def map_cli_args_to_dist_config(cli_args: argparse.Namespace) -> dict:
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
    hydra.initialize_config_dir(config_directory)
    return hydra.compose(config_name=config_name, overrides=overrides_args)


def runner_wrapper(config: DictConfig, cli_args: argparse.Namespace):
    Submitit()(config, cli_args)


# this is meant as a future replacement for the main entrypoint
def main(
    args: argparse.Namespace | None = None, override_args: list[str] | None = None
):
    if args is None:
        parser: argparse.ArgumentParser = flags.get_parser()
        args, override_args = parser.parse_known_args()

    cfg = get_hydra_config_from_yaml(args.config_yml, override_args)
    timestamp_id = get_timestamp_uid()
    log_dir = os.path.join(args.run_dir, timestamp_id, "logs")
    if args.submit:  # Run on cluster
        executor = AutoExecutor(folder=log_dir, slurm_max_num_timeout=3)
        executor.update_parameters(
            name=args.identifier,
            mem_gb=args.slurm_mem,
            timeout_min=args.slurm_timeout * 60,
            slurm_partition=args.slurm_partition,
            gpus_per_node=args.num_gpus,
            cpus_per_task=8,
            tasks_per_node=args.num_gpus,
            nodes=args.num_nodes,
            slurm_qos=args.slurm_qos,
            slurm_account=args.slurm_account,
        )
        job = executor.submit(runner_wrapper, cfg, args)
        logger.info(
            f"Submitted job id: {timestamp_id}, slurm id: {job.job_id}, logs: {log_dir}"
        )
    else:
        if args.num_gpus > 1:
            logging.info(f"Running in local mode with {args.num_gpus} ranks")
            # HACK to disable multiprocess dataloading in local mode
            # there is an open issue where LMDB's environment cannot be pickled and used
            # during torch multiprocessing https://github.com/pytorch/examples/issues/526
            # this HACK only works for a training submission where the config is passed in here
            if "optim" in cfg and "num_workers" in cfg["optim"]:
                cfg["optim"]["num_workers"] = 0
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
            elastic_launch(launch_config, runner_wrapper)(cfg, args)
        else:
            logger.info("Running in local mode without elastic launch")
            distutils.setup_env_local()
            runner_wrapper(cfg, args)
