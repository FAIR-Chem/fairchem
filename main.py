"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import copy
import logging
import os
import shutil
import time
import traceback
import warnings

import torch
from orion.core.utils.exceptions import ReservationRaceCondition
from yaml import dump

from ocpmodels.common import dist_utils
from ocpmodels.common.flags import flags
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    JOB_ID,
    apply_mult_factor,
    auto_note,
    build_config,
    continue_from_slurm_job_id,
    continue_orion_exp,
    load_orion_exp,
    merge_dicts,
    move_lmdb_data_to_slurm_tmpdir,
    read_slurm_env,
    resolve,
    set_max_fidelity,
    setup_imports,
    setup_logging,
    unflatten_dict,
    update_from_sbatch_py_vars,
)
from ocpmodels.trainers import BaseTrainer

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.multiprocessing.set_sharing_strategy("file_system")


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
        self.hparams = {}

    def run(self, orion_exp=None):
        orion_trial = None
        self.original_config = copy.deepcopy(self.trainer_config)
        orion_race_condition = False
        if dist_utils.is_master():
            if orion_exp:
                try:
                    orion_trial = orion_exp.suggest(1)
                    print(
                        "\n🚨  Orion reservation race condition detected. Exiting",
                        "and deleting run dir",
                    )
                    self.hparams = set_max_fidelity(
                        unflatten_dict(
                            apply_mult_factor(
                                orion_trial.params,
                                self.trainer_config.get("orion_mult_factor"),
                                sep="/",
                            ),
                            sep="/",
                        ),
                        orion_exp,
                    )
                    self.hparams["orion_hash_params"] = orion_trial.hash_params
                    self.hparams["orion_unique_exp_name"] = orion_exp.name
                except ReservationRaceCondition:
                    orion_race_condition = True
                    import wandb

                    if wandb.run is not None:
                        if wandb.run.tags:
                            wandb.run.tags = wandb.run.tags + ("RaceCondition",)
                        else:
                            wandb.run.tags = ("RaceCondition",)

        self.hparams, orion_race_condition = dist_utils.broadcast_from_master(
            self.hparams, orion_race_condition
        )
        if self.hparams:
            print("\n💎 Received hyper-parameters from Orion:")
            print(dump(self.hparams), end="\n")

        self.trainer_config = merge_dicts(self.trainer_config, self.hparams)
        self.trainer_config = continue_orion_exp(self.trainer_config)
        self.trainer_config = auto_note(self.trainer_config)
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

        dist_utils.synchronize()
        logging.info(f"Total time taken: {time.time() - start_time}")
        if self.trainer.logger is not None:
            self.trainer.logger.log({"Total time": time.time() - start_time})

        objective = dist_utils.broadcast_from_master(self.trainer.objective)

        if orion_exp is not None:
            if objective is None:
                if signal == "loss_is_nan":
                    objective = 1e12
                    print("Received NaN objective from worker. Setting to 1e12.")
                else:
                    print("Received None objective from worker. Skipping observation.")
            if objective is not None:
                orion_exp.observe(
                    orion_trial,
                    [{"type": "objective", "name": "energy_mae", "value": objective}],
                )


if __name__ == "__main__":
    runner = error = signal = None

    setup_logging()

    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()
    args = update_from_sbatch_py_vars(args)
    if args.logdir:
        args.logdir = resolve(args.logdir)

    trainer_config = build_config(args, override_args)
    trainer_config["optim"]["eval_batch_size"] = trainer_config["optim"]["batch_size"]

    original_trainer_config = copy.deepcopy(trainer_config)

    if args.distributed:
        dist_utils.setup(trainer_config)
        print("Distributed backend setup.")

    if dist_utils.is_master():
        trainer_config = move_lmdb_data_to_slurm_tmpdir(trainer_config)
        # dist_utils.synchronize()

    # -------------------
    # -----  Setup  -----
    # -------------------
    setup_imports()
    print("All things imported.")
    trainer_config = continue_from_slurm_job_id(trainer_config)
    trainer_config = read_slurm_env(trainer_config)
    runner = Runner(trainer_config)
    print("Runner ready.")

    try:
        # -------------------
        # -----  Train  -----
        # -------------------
        if args.orion_exp_config_path and dist_utils.is_master():
            experiment = load_orion_exp(args)
            print("\nStarting runner.")
            runner.run(orion_exp=experiment)
        else:
            print("Starting runner.")
            runner.run()

    except Exception:
        error = True
        print(traceback.format_exc())

    finally:
        if args.distributed:
            print(
                "\nWaiting for all processes to finish with dist_utils.cleanup()...",
                end="",
            )
            dist_utils.cleanup()
            print("Done!")

        if "interactive" not in os.popen(f"squeue -hj {JOB_ID}").read():
            print("\nSelf-canceling SLURM job in 32s", JOB_ID)
            os.popen(f"sleep 32 && scancel {JOB_ID}")

        if runner and runner.trainer and runner.trainer.logger:
            runner.trainer.logger.finish(error or signal)
