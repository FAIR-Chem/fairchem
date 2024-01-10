"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import os
import time
import traceback
import sys
import torch
from yaml import dump

from ocpmodels.common import dist_utils
from ocpmodels.common.flags import flags
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    JOB_ID,
    auto_note,
    build_config,
    merge_dicts,
    move_lmdb_data_to_slurm_tmpdir,
    resolve,
    setup_imports,
    setup_logging,
    update_from_sbatch_py_vars,
    set_min_hidden_channels,
)
from ocpmodels.common.orion_utils import (
    continue_orion_exp,
    load_orion_exp,
    sample_orion_hparams,
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
    print("\n" + "-" * 80 + "\n")
    print("🛑  OCP-DR-Lab Warnings (nota benes):")
    for warning in warnings:
        print(f"  • {warning}")
    print("Remove warnings when they are fixed in the code/configs.")
    print("\n" + "-" * 80 + "\n")


def wrap_up(args, start_time, error=None, signal=None, trainer=None):
    total_time = time.time() - start_time
    logging.info(f"Total time taken: {total_time}")
    if trainer and trainer.logger is not None:
        trainer.logger.log({"Total time": total_time})

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

    if trainer and trainer.logger:
        trainer.logger.finish(error or signal)


if __name__ == "__main__":
    error = signal = orion_exp = orion_trial = trainer = None
    orion_race_condition = False
    hparams = {}

    print()
    setup_logging()

    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()
    args = update_from_sbatch_py_vars(args)
    if args.logdir:
        args.logdir = resolve(args.logdir)

    # -- Build config

    trainer_config = build_config(args, override_args)

    if dist_utils.is_master():
        trainer_config = move_lmdb_data_to_slurm_tmpdir(trainer_config)
    dist_utils.synchronize()

    trainer_config["dataset"] = dist_utils.broadcast_from_master(
        trainer_config["dataset"]
    )

    # -- Initial setup

    setup_imports()
    print("\n🚩 All things imported.\n")
    start_time = time.time()

    try:
        # -- Orion

        if args.orion_exp_config_path and dist_utils.is_master():
            orion_exp = load_orion_exp(args)
            hparams, orion_trial = sample_orion_hparams(orion_exp, trainer_config)

            if hparams.get("orion_race_condition"):
                logging.warning("\n\n ⛔️ Orion race condition. Stopping here.\n\n")
                wrap_up(args, start_time, error, signal)
                sys.exit()

        hparams = dist_utils.broadcast_from_master(hparams)
        if hparams:
            print("\n💎 Received hyper-parameters from Orion:")
            print(dump(hparams), end="\n")
            trainer_config = merge_dicts(trainer_config, hparams)

        # -- Setup trainer

        trainer_config = continue_orion_exp(trainer_config)
        trainer_config = auto_note(trainer_config)
        trainer_config = set_min_hidden_channels(trainer_config)

        try:
            cls = registry.get_trainer_class(trainer_config["trainer"])
            trainer: BaseTrainer = cls(**trainer_config)
        except Exception as e:
            traceback.print_exc()
            logging.warning(f"\n💀 Error in trainer initialization: {e}\n")
            signal = "trainer_init_error"

        if signal is None:
            task = registry.get_task_class(trainer_config["mode"])(trainer_config)
            task.setup(trainer)
            print_warnings()

            # -- Start Training

            signal = task.run()

        # -- End of training

        # handle job preemption / time limit
        if signal == "SIGTERM":
            print("\nJob was preempted. Wrapping up...\n")
            if trainer:
                trainer.close_datasets()

        dist_utils.synchronize()

        objective = dist_utils.broadcast_from_master(
            trainer.objective if trainer else None
        )

        if orion_exp is not None:
            if objective is None:
                if signal == "loss_is_nan":
                    objective = 1e12
                    print("Received NaN objective from worker. Setting to 1e12.")
                if signal == "trainer_init_error":
                    objective = 1e12
                    print(
                        "Received trainer_init_error from worker.",
                        "Setting objective to 1e12.",
                    )
            if objective is not None:
                orion_exp.observe(
                    orion_trial,
                    [{"type": "objective", "name": "energy_mae", "value": objective}],
                )
            else:
                print("Received None objective from worker. Skipping observation.")

    except Exception:
        error = True
        print(traceback.format_exc())

    finally:
        wrap_up(args, start_time, error, signal, trainer=trainer)
