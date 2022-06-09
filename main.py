"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import copy
import logging
import os
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

import submitit

from ocpmodels.common import distutils
from ocpmodels.common.flags import flags
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    build_config,
    create_grid,
    resolve,
    save_experiment_log,
    setup_imports,
    setup_logging,
)


class Runner(submitit.helpers.Checkpointable):
    def __init__(self):
        self.config = None

    def __call__(self, config):
        setup_logging()
        self.config = copy.deepcopy(config)

        if args.distributed:
            distutils.setup(config)

        try:
            setup_imports()
            config = self.should_continue(config)
            config = self.read_slurm_env(config)
            self.trainer = registry.get_trainer_class(config.get("trainer", "energy"))(
                task=config["task"],
                model_attributes=config["model"],
                dataset=config["dataset"],
                optimizer=config["optim"],
                identifier=config["identifier"],
                timestamp_id=config.get("timestamp_id", None),
                run_dir=config.get("run_dir", "./"),
                is_debug=config.get("is_debug", False),
                print_every=config.get("print_every", 100),
                seed=config.get("seed", 0),
                logger=config.get("logger", "wandb"),
                local_rank=config["local_rank"],
                amp=config.get("amp", False),
                cpu=config.get("cpu", False),
                slurm=config.get("slurm", {}),
                new_gnn=config.get("new_gnn", True),
                data_split=config.get("data_split", None),
                note=config.get("note", ""),
            )
            self.task = registry.get_task_class(config["mode"])(self.config)
            self.task.setup(self.trainer)
            start_time = time.time()
            self.task.run()
            distutils.synchronize()
            logging.info(f"Total time taken: {time.time() - start_time}")
        finally:
            if args.distributed:
                distutils.cleanup()

    def checkpoint(self, *args, **kwargs):
        new_runner = Runner()
        self.trainer.save(checkpoint_file="checkpoint.pt", training_state=True)
        self.config["checkpoint"] = self.task.chkpt_path
        self.config["timestamp_id"] = self.trainer.timestamp_id
        if self.trainer.logger is not None:
            self.trainer.logger.mark_preempting()
        return submitit.helpers.DelayedSubmission(new_runner, self.config)

    def read_slurm_env(self, config):
        """
        Parses the output of `scontrol show` in order to store the slurm
        config (mem, cpu, node, gres) as a `"slurm"` key in the `config` object.

        Args:
            config (dict): Run configuration

        Returns:
            dict: Updated run config if no "slurm" key exists or it's empty
        """
        if not config.get("slurm"):
            return config

        command = f"scontrol show job {os.environ.get('SLURM_JOB_ID')}"
        scontrol = subprocess.check_output(command.split(" ")).decode("utf-8").strip()
        params = re.findall(r"TRES=(.+)\n", scontrol)
        try:
            if params:
                params = params[0]
                config["slurm"] = {}
                for kv in params.split(","):
                    k, v = kv.split("=")
                    config["slurm"][k] = v
        except Exception as e:
            print("Slurm config creation exception", e)
        finally:
            return config

    def should_continue(self, config):
        """
        Assuming runs are consistently executed in a `run_dir` with the
        `run_dir/$SLURM_JOBID` pattern, this functions looks for an existing
        directory with the same $SLURM_JOBID as the current job that contains
        a checkpoint.

        If there is one, it tries to find `best_checkpoint.ckpt`.
        If the latter does not exist, it looks for the latest checkpoint,
        assuming a naming convention like `checkpoint-{step}.pt`.

        If a checkpoint is found, its path is set in `config["checkpoint"]`.
        Otherwise, returns the original config.

        Args:
            config (dict): The original config to overwrite

        Returns:
            dict: The updated config if a checkpoint has been found
        """
        if config["checkpoint"]:
            return config

        job_id = os.environ.get("SLURM_JOBID")
        if job_id is None:
            return config

        base_dir = Path(config["run_dir"]).resolve().parent
        ckpt_dir = base_dir / job_id / "checkpoints"
        if not ckpt_dir.exists() or not ckpt_dir.is_dir():
            return config

        best_ckp = ckpt_dir / "best_checkpoint.pt"
        if best_ckp.exists():
            config["checkpoint"] = str(best_ckp)
        else:
            ckpts = list(ckpt_dir.glob("checkpoint-*.pt"))
            if not ckpts:
                return config
            latest_ckpt = sorted(ckpts, key=lambda f: f.stem)[-1]
            if latest_ckpt.exists() and latest_ckpt.is_file():
                config["checkpoint"] = str(latest_ckpt)

        return config


if __name__ == "__main__":
    setup_logging()

    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()
    if not args.mode or not args.config_yml:
        args.mode = "train"
        args.config_yml = "configs/is2re/10k/schnet/schnet.yml"
        # args.checkpoint = "checkpoints/2022-04-26-12-23-28-schnet/checkpoint.pt"
        warnings.warn("No model / mode is given; chosen as default")
    if args.logdir:
        args.logdir = resolve(args.logdir)

    config = build_config(args, override_args)

    if args.submit:  # Run on cluster
        slurm_add_params = config.get("slurm", None)  # additional slurm arguments
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
        logging.info(f"Submitted jobs: {', '.join([job.job_id for job in jobs])}")
        log_file = save_experiment_log(args, jobs, configs)
        logging.info(f"Experiment log saved to: {log_file}")

    else:  # Run locally
        Runner()(config)
