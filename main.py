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
            config["slurm"] = {}
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
    args = update_from_sbatch_py_vars(args)
    if not args.mode or not args.config_yml:
        args.mode = "train"
        # args.config_yml = "configs/is2re/10k/schnet/new_schnet.yml"
        args.config_yml = "configs/is2re/10k/sfarinet/sfarinet.yml"
        # args.checkpoint = "checkpoints/2022-04-26-12-23-28-schnet/checkpoint.pt"
        warnings.warn("No model / mode is given; chosen as default")
    if args.logdir:
        args.logdir = resolve(args.logdir)

    run_config = build_config(args, override_args)
    run_config["optim"]["eval_batch_size"] = run_config["optim"]["batch_size"]

    setup_logging()
    original_run_config = copy.deepcopy(run_config)

    if args.distributed:
        distutils.setup(run_config)

    try:
        setup_imports()
        run_config = should_continue(run_config)
        run_config = read_slurm_env(run_config)
        trainer = registry.get_trainer_class(run_config.get("trainer", "energy"))(
            task=run_config["task"],
            model_attributes=run_config["model"],
            dataset=run_config["dataset"],
            optimizer=run_config["optim"],
            run_dir=run_config.get("run_dir", "./"),
            is_debug=run_config.get("is_debug", False),
            print_every=run_config.get("print_every", 100),
            seed=run_config.get("seed", 0),
            logger=run_config.get("logger", "wandb"),
            local_rank=run_config["local_rank"],
            amp=run_config.get("amp", False),
            cpu=run_config.get("cpu", False),
            slurm=run_config.get("slurm", {}),
            new_gnn=run_config.get("new_gnn", True),
            frame_averaging=run_config.get("frame_averaging", None),
            data_split=run_config.get("data_split", None),
            note=run_config.get("note", ""),
            test_invariance=run_config.get("test_ri", None),
            choice_fa=run_config.get("choice_fa", None),
            wandb_tags=run_config.get("wandb_tags", None),
        )
        task = registry.get_task_class(run_config["mode"])(run_config)
        task.setup(trainer)
        start_time = time.time()
        task.run()
        distutils.synchronize()
        logging.info(f"Total time taken: {time.time() - start_time}")
        if trainer.logger is not None:
            trainer.logger.log({"Total time": time.time() - start_time})
    finally:
        if args.distributed:
            distutils.cleanup()
