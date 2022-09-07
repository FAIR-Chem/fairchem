from minydra import resolved_args
import subprocess
import wandb
from uuid import uuid4
from datetime import datetime
from pathlib import Path
import os
import yaml


def run_command(command):
    return subprocess.check_output(command).decode("utf-8").strip()


def make_wandb_sweep_name():
    return "sweep_py_" + str(uuid4()).split("-")[-1]


def resolve(path):
    """
    Resolves a path: expand user (~) and env vars ($SCRATCH) and resovles to
    an absolute path.

    Args:
        path (Union[str, pathlib.Path]): the path to resolve

    Returns:
        pathlib.Path: the resolved Path
    """
    return Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()


def now():
    """
    Get a string describing the current time & date as:
    YYYY-MM-DD_HH-MM-SS

    Returns:
        str: now!
    """
    return str(datetime.now()).split(".")[0].replace(":", "-").replace(" ", "_")


def discover_minydra_defaults():
    """
    Returns a list containing:
    * the path to the shared configs/sweep/defaults.yaml file
    * the path to configs/sweep/$USER.yaml file if it exists


    Returns:
        list[pathlib.Path]: Path to the shared defaults and optionnally
            to a user-specific one if it exists
    """
    root = Path(__file__).resolve().parent
    defaults = [root / "configs" / "sweep" / "defaults.yaml"]
    user_config = root / "configs" / "sweep" / f"{os.environ['USER']}.yaml"
    if user_config.exists() and user_config.is_file():
        defaults.append(user_config)
    return defaults


def read_sweep_params(path):
    with resolve(path).open("r") as f:
        params = yaml.safe_load(f)
    if "parameters" in params:
        return params["parameters"]
    return params


def make_sbatch_exec(args):
    sbatch_args = " ".join([f"{k}={v}" for k, v in args.items()])

    def run_sbatch():
        wandb.init(mode="disabled")
        args = " ".join([f"--{k}={v}" for k, v in wandb.config.items()])
        py_args = f"py_args={args}"

        command = f"python sbatch.py {sbatch_args}"
        print("\n" + "=" * 30 + "\n" + command + "\n")
        print(run_command(command.split(" ") + [py_args]))
        print("\n" + "=" * 30 + "\n")
        wandb.finish(exit_code=0, quiet=True)

    return run_sbatch


if __name__ == "__main__":
    args = resolved_args(
        defaults=discover_minydra_defaults(), strict=False
    ).pretty_print()
    name = args.name or make_wandb_sweep_name()
    parameters = read_sweep_params(args.params)
    parameters["wandb_tag"] = {"values": [name]}
    sweep_configuration = {
        "name": name,
        "method": args.method,
        "parameters": parameters,
    }
    sweep_id = wandb.sweep(sweep_configuration)

    count = args.count
    del args.name
    del args.method
    del args.params
    del args.count

    wandb.agent(sweep_id, function=make_sbatch_exec(args), count=count)
