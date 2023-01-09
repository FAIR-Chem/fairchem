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
import time
import traceback
import warnings
from pathlib import Path

import torch
from orion.client import build_experiment
from yaml import safe_load

from ocpmodels.common import distutils
from ocpmodels.common.flags import flags
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    JOB_ID,
    build_config,
    continue_from_slurm_job_id,
    continue_orion_exp,
    merge_dicts,
    move_lmdb_data_to_slurm_tmpdir,
    read_slurm_env,
    resolve,
    setup_imports,
    setup_logging,
    update_from_sbatch_py_vars,
)
from ocpmodels.trainers import BaseTrainer

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.multiprocessing.set_sharing_strategy("file_system")

try:
    import ipdb  # noqa: F401

    os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
except:  # noqa: E722
    print(
        "`ipdb` is not installed. ",
        "Consider `pip install ipdb` to improve your debugging experience.",
    )


def print_warnings():
    warnings = [
        "`max_num_neighbors` is set to 40. This should be tuned per model.",
        "`tag_specific_weights` is not handled for "
        + "`regress_forces: direct_with_gradient_target` in compute_loss()",
    ]
    print("\n" + "-" * 80)
    print("🛑  OCP-DR-Lab Warnings (nota benes):")
    for warning in warnings:
        print(f"  • {warning}")
    print("Remove warnings when they are fixed in the code/configs.")
    print("-" * 80 + "\n")


class Runner:
    def __init__(self, trainer_config):
        self.trainer_config = trainer_config
        self.trainer = None

    def run(self, **hparams):
        self.original_config = copy.deepcopy(self.trainer_config)
        if distutils.is_master():
            orion_trial = hparams.pop("orion_trial", None)
            if orion_trial:
                hparams["orion_hash_params"] = orion_trial.hash_params
        self.hparams = hparams

        should_be_0 = distutils.get_rank()
        hp_list = [hparams, should_be_0]
        # print("hparams pre-broadcast: ", hparams)
        distutils.broadcast_object_list(hp_list)
        hparams, should_be_0 = hp_list
        # print("hparams post-broadcast: ", hparams)
        assert should_be_0 == 0
        if hparams:
            print("Received hyper-parameters from Orion:")
            print(hparams)

        self.trainer_config = merge_dicts(self.trainer_config, hparams)
        self.trainer_config = continue_orion_exp(self.trainer_config)
        cls = registry.get_trainer_class(self.trainer_config["trainer"])
        self.trainer: BaseTrainer = cls(**self.trainer_config)
        task = registry.get_task_class(self.trainer_config["mode"])(self.trainer_config)
        task.setup(self.trainer)
        start_time = time.time()
        print_warnings()

        signal = task.run()

        # handle job preemption / time limit
        if signal == "SIGTERM":
            print("\nJob was preempted. Wrapping up...\n")
            self.trainer.close_datasets()

        distutils.synchronize()
        logging.info(f"Total time taken: {time.time() - start_time}")
        if self.trainer.logger is not None:
            self.trainer.logger.log({"Total time": time.time() - start_time})

        objective = self.trainer.objective
        # print("objective pre-broadcast: ", objective)
        o_list = [objective]
        distutils.broadcast_object_list(o_list)
        objective = o_list[0]
        # print("objective post-broadcast: ", objective)

        return [{"name": "energy_mae", "type": "objective", "value": objective}]


if __name__ == "__main__":
    runner = error = signal = None

    setup_logging()

    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()
    args = update_from_sbatch_py_vars(args)
    if not args.config:
        args.config = "sfarinet-is2re-10k"
        # args.checkpoint = "checkpoints/2022-04-26-12-23-28-schnet/checkpoint.pt"
        warnings.warn(
            f"\n>>>> No config is provided. Defaulting to {args.config} chosen\n"
        )
    if args.logdir:
        args.logdir = resolve(args.logdir)

    trainer_config = build_config(args, override_args)
    trainer_config["optim"]["eval_batch_size"] = trainer_config["optim"]["batch_size"]

    setup_logging()
    original_trainer_config = copy.deepcopy(trainer_config)

    if args.distributed:
        distutils.setup(trainer_config)
        print("Distributed backend setup.")

    if distutils.is_master():
        trainer_config = move_lmdb_data_to_slurm_tmpdir(trainer_config)
        # distutils.synchronize()

    try:
        # -------------------
        # -----  Setup  -----
        # -------------------
        setup_imports()
        print("All things imported.")
        trainer_config = continue_from_slurm_job_id(trainer_config)
        trainer_config = read_slurm_env(trainer_config)
        runner = Runner(trainer_config)
        print("Runner ready.")
        # -------------------
        # -----  Train  -----
        # -------------------
        if args.orion_search_path and distutils.is_master():
            assert args.orion_unique_exp_name
            space = safe_load(Path(args.orion_search_path).read_text())
            print("Search Space: ", space)
            experiment = build_experiment(
                name=args.orion_unique_exp_name,
                space=space,
                algorithms={"mofa": {"seed": 123}},
            )
            experiment.workon(runner.run, max_trials_per_worker=1, n_workers=1)
        else:
            print("Starting runner.")
            runner.run()

    except Exception:
        error = True
        print(traceback.format_exc())

    finally:
        if args.distributed:
            print(
                "\nWaiting for all processes to finish with distutils.cleanup()...",
                end="",
            )
            distutils.cleanup()
            print("Done!")

        if runner and runner.trainer and runner.trainer.logger:
            runner.trainer.logger.finish(error or signal)

        if "interactive" not in os.popen(f"squeue -hj {JOB_ID}").read():
            print("\nSelf-canceling SLURM job", JOB_ID)
            os.system(f"scancel {JOB_ID}")
