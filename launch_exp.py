import os
import re
from pathlib import Path

from minydra import resolved_args
from yaml import safe_load

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
            if " " in str(v) or "," in str(v) or isinstance(v, str):
                if "'" in str(v) and '"' in str(v):
                    v = str(v).replace("'", "\\'")
                    v = f"'{v}'"
                elif "'" in str(v):
                    v = f'"{v}"'
                else:
                    v = f"'{v}'"
            s += f" --{parent}{k}={v}"
    return s


if __name__ == "__main__":
    args = resolved_args()
    assert "exp" in args
    regex = args.get("match", ".*")

    exp_name = args.exp.replace(".yml", "").replace(".yaml", "")
    exp_file = find_exp(exp_name)

    exp = safe_load(exp_file.open("r"))

    runs = exp["runs"]

    commands = []

    for run in runs:
        params = exp["default"].copy()
        job = exp["job"].copy()

        job.update(run.pop("job", {}))
        if run.pop("_no_exp_default_", False):
            params = {}
        params.update(run)
        if "time" in job:
            job["time"] = seconds_to_time_str(job["time"])

        if "wandb_tags" in params:
            params["wandb_tags"] += "," + exp_name
        else:
            params["wandb_tags"] = exp_name

        py_args = f'py_args="{cli_arg(params).strip()}"'

        sbatch_args = " ".join([f"{k}={v}" for k, v in job.items()])
        command = f"python sbatch.py {sbatch_args} {py_args}"
        commands.append(command)

    commands = [c for c in commands if re.findall(regex, c)]

    print(f"🔥 About to run {len(commands)} jobs:\n\n • " + "\n\n  • ".join(commands))

    confirm = input("\n🚦 Confirm? [y/n]")

    if confirm == "y":
        outputs = [
            print(f"Launching job {c:3}", end="\r") or os.popen(command).read().strip()
            for c, command in enumerate(commands)
        ]
        outdir = Path(__file__).resolve().parent / "data" / "exp_outputs" / exp_name
        outfile = outdir / f"{exp_name}_{now()}.txt"
        outfile.parent.mkdir(exist_ok=True, parents=True)
        exp_separator = "\n" * 4 + f"{'#' * 80}\n" * 4 + "\n" * 4
        text = exp_separator.join(outputs)
        jobs = [
            line.replace(sep, "").strip()
            for line in text.splitlines()
            if (sep := "Submitted batch job ") in line
        ]
        text += f"{exp_separator}All jobs launched: {' '.join(jobs)}"
        with outfile.open("w") as f:
            f.write(text)
        print(f"Output written to {str(outfile)}")
        print("All job launched:", " ".join(jobs))
    else:
        print("Aborting")
