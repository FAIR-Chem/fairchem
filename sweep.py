import wandb
from minydra import resolved_args
import subprocess
from uuid import uuid4
from datetime import datetime
from pathlib import Path
import os
import sys
import yaml
import re


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


def get_sweep_out_path(name):
    n = now()
    fname = f"{n}_{name}.txt"
    sweeps_dir = resolve(__file__).parent / "data" / "sweeps"
    files = list(sweeps_dir.glob(f"*{name}.txt"))
    if len(files) > 1:
        raise ValueError("Too many matching sweep files for " + name)
    if len(files) == 1:
        f = files[0]
        f.rename(f.parent / fname)
    return sweeps_dir / fname


def make_sbatch_exec(args, log_file_path, jobs, mode):
    if mode == "print_commands":
        args.dev = True
    sbatch_args = " ".join([f"{k}={v}" for k, v in args.items()])

    def run_sbatch():
        wandb.init(mode="disabled")
        args = " ".join([f"--{k}={v}" for k, v in wandb.config.items()])
        py_args = f"py_args={args}"
        command = f"python sbatch.py {sbatch_args}"
        out = run_command(command.split(" ") + [py_args])
        job = re.findall(r"Submitted batch job (\d+)", out)
        if job:
            jobs.append(job[0])
        if mode == "print_commands":
            python_command_lines = [
                line for line in out.split("\n") if "python main.py" in line
            ]
            if python_command_lines:
                line = python_command_lines[0]
                print("\npython main.py" + line.split("python main.py")[-1] + "\n")
            pass
        else:
            print("\n" + "░" * 50 + "\n" + command + "\n")
            print(out)
            with log_file_path.open("a") as f:
                f.write("\n\n\n" + "░" * 80)
                f.write("\n" + "░" * 80)
                f.write("\n" + "░" * 80 + "\n\n\n")
                f.write(command + " " + py_args)
                f.write("\n\nResult:\n")
                f.write(out)
                f.write("\n\n")
            print("\n" + "░" * 50 + "\n")
        wandb.finish(exit_code=0, quiet=True)

    return run_sbatch


if __name__ == "__main__":
    args = resolved_args(defaults=discover_minydra_defaults(), strict=False)

    if args.mode not in {"run_jobs", "print_commands"}:
        raise ValueError("Unknown args.mode: " + str(args.mode))
    if args.mode == "print_commands":
        os.environ["WANDB_SILENT"] = "true"
    else:
        args.pretty_print()

    jobs = []
    name = args.name or make_wandb_sweep_name()
    parameters = read_sweep_params(args.params)
    parameters["wandb_tag"] = {"values": [name]}
    sweep_configuration = {
        "name": name,
        "method": args.method,
        "parameters": parameters,
    }
    sweep_id = wandb.sweep(sweep_configuration)

    fpath = get_sweep_out_path(name)
    fpath.parent.mkdir(parents=True, exist_ok=True)

    pre_exists = fpath.exists()

    if args.mode != "python_command":
        print("Writing to:", str(fpath))
        with open(fpath, "a") as f:
            if pre_exists:
                f.write("\n\n" + "█" * 120)
                f.write("\n" + "█" * 120)
                f.write("\n" + "█" * 120)
                f.write("\n" + "█" * 120)
                f.write("\n" + "█" * 120 + "\n\n")
            f.write(" ".join(sys.argv))

    count = args.count
    mode = args.mode
    del args.name
    del args.method
    del args.params
    del args.count
    del args.mode

    wandb.agent(
        sweep_id, function=make_sbatch_exec(args, fpath, jobs, mode), count=count
    )
    if mode != "print_commands":

        with open(fpath, "r") as f:
            lines = f.read()

        pre_existing_jobs = re.findall(r" ► (\d+)", lines)
        lines = lines.split("\n")

        new_f = []
        delete = False
        for l in lines:
            if "░ ░ ░ ░ ░" in l and not delete:
                delete = True
            if "░ ░ ░ ░ ░" in l and delete:
                delete = False
            if not delete:
                new_f.append(l)

        with open(fpath, "w") as f:
            f.write("\n".join(new_f))
            f.write("\n\n" + ("░ ░ ░ " * 10)[:-1] + "\n")
            f.write("   Sweep Job IDs\n")
            if jobs:
                f.write("\n".join([f" ► {j}" for j in (pre_existing_jobs + jobs)]))
            f.write("\n" + ("░ ░ ░ " * 10)[:-1] + "\n")

        print(f"\nSweep File:\n{str(fpath)}")
