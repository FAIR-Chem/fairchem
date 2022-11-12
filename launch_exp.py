from minydra import resolved_args
from pathlib import Path
from yaml import safe_load
import os
from sbatch import now


def find_exp(name):
    exp_dir = Path(__file__).parent / "configs" / "exps"
    exp_file = exp_dir / f"{name}.yml"
    if exp_file.exists():
        return exp_file
    exp_file = exp_dir / f"{name}.yaml"
    if exp_file.exists():
        return exp_file

    raise ValueError(f"Could not find experiment {name}")


def seconds_to_time_str(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def cli_arg(args, key=""):
    s = ""
    for k, v in args.items():
        parent = "" if not key else f"{key}."
        if isinstance(v, dict):
            s += cli_arg(v, key=f"{parent}{k}")
        else:
            if " " in str(v):
                v = f"'{v}'"
            s += f" --{parent}{k}={v}"
    return s


if __name__ == "__main__":
    args = resolved_args()
    assert "exp" in args

    exp_name = args.exp.replace(".yml", "").replace(".yaml", "")
    exp_file = find_exp(exp_name)

    exp = safe_load(exp_file.open("r"))
    if "time" in exp["job"]:
        exp["job"]["time"] = seconds_to_time_str(exp["job"]["time"])

    runs = exp["runs"]

    commands = []

    for run in runs:
        params = exp["default"].copy()
        params.update(run)

        if "wandb_tags" in params:
            params["wandb_tags"] += "," + exp_name
        else:
            params["wandb_tags"] = exp_name

        py_args = f'py_args="{cli_arg(params).strip()}"'

        sbatch_args = " ".join([f"{k}={v}" for k, v in exp["job"].items()])

        command = f"python sbatch.py {sbatch_args} {py_args}"
        commands.append(command)

    print(f"About to run {len(commands)} jobs:\n • " + "\n\n  • ".join(commands))

    confirm = input("Confirm? [y/n]")

    if "y" in confirm:
        outputs = [
            print(f"Launching job {c:3}", end="\r") or os.popen(command).read().strip()
            for c, command in enumerate(commands)
        ]
        outdir = Path(__file__).parent / "data" / "exp_outputs"
        outdir.mkdir(exist_ok=True, parents=True)
        outfile = outdir / f"{exp_name}_{now()}.txt"
        exp_separator = "\n" * 4 + "{'#' * 80}\n" * 4 + "\n" * 4
        outfile.write_text(exp_separator.join(outputs))
        print(f"Output written to {str(outfile)}")
    else:
        print("Aborting")
