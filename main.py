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

from ocpmodels.common import distutils
from ocpmodels.common.flags import flags
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    build_config,
    resolve,
    setup_imports,
    setup_logging,
    update_from_sbatch_py_vars,
)
from ocpmodels.trainers import BaseTrainer

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

try:
    import ipdb  # noqa: F401

    os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
except:  # noqa: E722
    print(
        "`ipdb` is not installed. ",
        "Consider `pip install ipdb` to improve your debugging experience.",
    )


def read_slurm_env(config):
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
            for kv in params.split(","):
                k, v = kv.split("=")
                config["slurm"][k] = v
    except Exception as e:
        print("Slurm config creation exception", e)
    finally:
        return config


def should_continue(config):
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
    if config.get("checkpoint"):
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


if __name__ == "__main__":
    ntfy = trainer = error = None

    setup_logging()
    slurm_job_id = os.environ.get("SLURM_JOB_ID")

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

    try:
        # -------------------
        # -----  Setup  -----
        # -------------------
        setup_imports()
        trainer_config = should_continue(trainer_config)
        trainer_config = read_slurm_env(trainer_config)
        # -------------------
        # -----  Train  -----
        # -------------------
        trainer: BaseTrainer = registry.get_trainer_class(trainer_config["trainer"])(
            **trainer_config
        )
        task = registry.get_task_class(trainer_config["mode"])(trainer_config)
        task.setup(trainer)
        start_time = time.time()
        if trainer.logger is not None:
            message = f"{slurm_job_id} - Training started 🚀"
            if trainer_config.get("note"):
                message += f" - {trainer_config.get('note')}"
            if trainer_config.get("wandb_tags"):
                message += f" - {trainer_config.get('wandb_tags')}"
            trainer.logger.ntfy(message, click=trainer.logger.url)
        print_warnings()

        signal = task.run()

        # handle job preemption / time limit
        if signal == "SIGTERM":
            print("\nJob was preempted. Wrapping up...\n")
            for ds in trainer.datasets.values():
                if hasattr(ds, "close_db") and callable(ds.close_db):
                    ds.close_db()

        # -----------------
        # -----  End  -----
        # -----------------
        distutils.synchronize()
        logging.info(f"Total time taken: {time.time() - start_time}")
        if trainer.logger is not None:
            trainer.logger.log({"Total time": time.time() - start_time})

    except Exception as e:
        if trainer and trainer.logger:
            e_name = e.__class__.__name__
            trainer.logger.ntfy(
                f"{slurm_job_id} - Training failed 😭" + f"{e_name} - {str(e)}",
                click=trainer.logger.url or None,
            )
        error = True
        print(traceback.format_exc())

    finally:
        if args.distributed:
            print(
                "Waiting for all processes to finish with distutils.cleanup()...",
                end="",
            )
            distutils.cleanup()
            print("Done!")

        if trainer and trainer.logger:
            trainer.logger.finish(error or signal)

        if "interactive" not in os.popen(f"squeue -hj {slurm_job_id}").read():
            print("Self-canceling SLURM job", os.getenv("SLURM_JOB_ID"))
            os.system(f"scancel {os.getenv('SLURM_JOB_ID')}")
