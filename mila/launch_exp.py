import copy
import os
import re
import subprocess
import sys
from pathlib import Path

from minydra import resolved_args
from sbatch import now
from yaml import dump, safe_load

ROOT = Path(__file__).resolve().parent.parent


def util_strings(jobs, yaml_comments=False):
    s = "  • All jobs launched: " + ", ".join(jobs)
    s += "\n  • Cancel experiment:\n    $ scancel " + " ".join(jobs)
    s += "\n  • WandB query for dashboard:\n    (" + "|".join(jobs) + ")"
    s += (
        "\n  • Delete experiment run dirs:\n    $ "
        + 'exp_run_dirs="$SCRATCH/ocp/runs"; for jid in '
        + " ".join(jobs)
        + '; do rm -rf "$exp_run_dirs/$jid" && echo "Deleted $exp_run_dirs/$jid"; done;'
    )
    if yaml_comments:
        s = "\n".join(["# " + line for line in s.splitlines()])
    else:
        s = "\n  │ ".join(s.splitlines())
    return s


def merge_dicts(dict1: dict, dict2: dict):
    """Recursively merge two dictionaries.
    Values in dict2 override values in dict1. If dict1 and dict2 contain a dictionary
    as a value, this will call itself recursively to merge these dictionaries.
    This does not modify the input dictionaries (creates an internal copy).
    Additionally returns a list of detected duplicates.
    Adapted from https://github.com/TUM-DAML/seml/blob/master/seml/utils.py

    Parameters
    ----------
    dict1: dict
        First dict.
    dict2: dict
        Second dict. Values in dict2 will override values from dict1 in case they share
        the same key.

    Returns
    -------
    return_dict: dict
        Merged dictionaries.
    """
    if not isinstance(dict1, dict):
        raise ValueError(f"Expecting dict1 to be dict, found {type(dict1)}.")
    if not isinstance(dict2, dict):
        raise ValueError(f"Expecting dict2 to be dict, found {type(dict2)}.")

    return_dict = copy.deepcopy(dict1)

    for k, v in dict2.items():
        if k not in dict1:
            return_dict[k] = v
        else:
            if isinstance(v, dict) and isinstance(dict1[k], dict):
                return_dict[k] = merge_dicts(dict1[k], dict2[k])
            elif isinstance(v, list) and isinstance(dict1[k], list):
                if len(dict1[k]) != len(dict2[k]):
                    raise ValueError(
                        f"List for key {k} has different length in dict1 and dict2."
                        + " Use an empty dict {} to pad for items in the shorter list."
                    )
                return_dict[k] = [merge_dicts(d1, d2) for d1, d2 in zip(dict1[k], v)]
            else:
                return_dict[k] = dict2[k]

    return return_dict


def write_exp_yaml_and_jobs(exp_file, outfile, jobs):
    """
    Reads the exp_file, adds the jobs as comments in each run line and writes the
    resulting yaml file in the same directory as the outfile.

    Args:
        exp_file (Path): Path to the experimental yaml file
        outfile (Path): Path to the output txt file
        jobs (list[str]): List of jobs, one per run line in the yaml exp_file
    """
    lines = exp_file.read_text().splitlines()
    run_lines = [i for i, l in enumerate(lines) if l.strip().startswith("runs:")]
    if run_lines:
        run_line = run_lines[0]
        j = 0
        for i, line in enumerate(lines[run_line:]):
            if line.strip().startswith("- "):
                lines[run_line + i] = f"{line}  # {jobs[j]}"
                j += 1

    lines += [""] + util_strings(jobs, True).splitlines()
    yml_out = outfile.with_suffix(".yaml")
    yml_out.write_text("\n".join(lines))
    return yml_out


def get_commit():
    try:
        commit = (
            subprocess.check_output("git rev-parse --verify HEAD".split())
            .decode("utf-8")
            .strip()
        )
    except Exception:
        commit = "unknown"
    return commit


def find_exp(name):
    exp_dir = ROOT / "configs" / "exps"
    exp_file = exp_dir / f"{name}.yaml"
    if exp_file.exists():
        return exp_file

    raise ValueError(f"Could not find experiment {name}")


def seconds_to_time_str(seconds):
    try:
        seconds = int(seconds)
    except ValueError:
        return seconds
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
                if '"' in str(v):
                    v = str(v).replace('"', '\\"')
                    v = f"'{v}'"
                elif "'" in str(v):
                    v = f'\\"{v}\\"'
                else:
                    v = f"'{v}'"
            s += f" --{parent}{k}={v}"
            if "ads" in k:
                print(s.split(" --")[-1])
    return s


def get_args_or_exp(key, args, exp):
    value = None
    if key in args:
        if key in exp:
            print(f"Overriding orion.{key} from the command-line")
        value = args[key]
    elif key in exp:
        value = exp[key]
    else:
        raise ValueError(
            f"Must specify 'orion.{key}' "
            + f"in exp file or from the command-line `{key}=value`"
        )
    return value


if __name__ == "__main__":
    is_interrupted = False
    n_jobs = None
    args = resolved_args()
    assert "exp" in args

    regex = args.pop("match", ".*")
    exp_name = args.pop("exp").replace(".yml", "").replace(".yaml", "")
    no_confirm = args.pop("no_confirm", False)

    sbatch_overrides = args.to_dict()

    ts = now()

    exp_file = find_exp(exp_name)

    exp = safe_load(exp_file.open("r"))

    if "orion" in exp:
        orion_base = ROOT / "data" / "orion"
        assert "runs" not in exp, "Cannot use both Orion and runs"
        assert "space" in exp["orion"], "Must specify orion.space"
        assert "algorithms" in exp["orion"], "Must specify orion.algorithms"

        n_jobs = get_args_or_exp("n_jobs", args, exp["orion"])
        unique_exp_name = get_args_or_exp("unique_exp_name", args, exp["orion"])
        if "unique_exp_name" not in exp:
            exp["unique_exp_name"] = unique_exp_name

        search_path = orion_base / "search-spaces" / f"{ts}-{unique_exp_name}.yaml"
        job_dict = {
            "job_name": unique_exp_name,
        }

        if (exp["job"].get("partition") or "long") == "long":
            job_dict["partition"] = "long-grace"

        runs = [
            {
                "orion_exp_config_path": str(search_path),
                "job": job_dict,
            }
            for _ in range(n_jobs)
        ]
    else:
        runs = exp["runs"]

    commands = []

    for run in runs:
        params = exp["default"].copy()
        job = merge_dicts(exp["job"].copy(), run.pop("job", {}))
        if run.pop("_no_exp_default_", False):
            params = {}
        params = merge_dicts(params, run)
        if "time" in job:
            job["time"] = seconds_to_time_str(job["time"])

        if "job_name" not in job:
            job["job_name"] = exp_name

        if "wandb_tags" in params:
            params["wandb_tags"] += "," + exp_name
        else:
            params["wandb_tags"] = exp_name

        job = merge_dicts(job, sbatch_overrides)

        py_args = f'py_args="{cli_arg(params).strip()}"'

        sbatch_args = " ".join(
            [f"{k}={v}" for k, v in job.items()] + [f"exp_name={exp_name}"]
        )
        sbatch_path = str(ROOT / "mila" / "sbatch.py")
        command = f"python {sbatch_path} {sbatch_args} {py_args}"
        commands.append(command)

    commands = [c for c in commands if re.findall(regex, c)]

    print(
        f"🔥 About to run {len(commands)} jobs:\n\n • "
        + "\n\n  • ".join(commands if n_jobs is None else commands[:1])
    )

    separator = "\n" * 4 + f"{'#' * 80}\n" * 4 + "\n" * 4
    text = "<><><> Experiment command: $ " + " ".join(["python"] + sys.argv)
    text += "\n<><><> Experiment commit: " + get_commit()
    text += "\n<><><> Experiment config:\n\n-----" + exp_file.read_text() + "-----"
    text += "\n<><><> Experiment runs:\n\n • " + "\n\n  • ".join(commands) + separator

    confirm = no_confirm or "y" in input("\n🚦 Confirm? [y/n] : ")

    if confirm:
        try:
            if "orion" in exp:
                search_path.parent.mkdir(exist_ok=True, parents=True)
                assert not search_path.exists()
                search_path.write_text(dump(exp["orion"]))

            outputs = []
            print()
            for c, command in enumerate(commands):
                print(f"Launching job {c+1:3}", end="\r")
                outputs.append(os.popen(command).read().strip())
                if "Aborting" in outputs[-1]:
                    print("\nError submitting job", c + 1, ":", command)
                    print(outputs[-1].replace("Error while launching job:\n", ""))
                    print("\n")
                if " verbose=true" in command.lower():
                    print(outputs[-1])
        except KeyboardInterrupt:
            is_interrupted = True
        outdir = ROOT / "data" / "exp_outputs" / exp_name
        outfile = outdir / f"{exp_name.split('/')[-1]}_{ts}.txt"
        outfile.parent.mkdir(exist_ok=True, parents=True)
        text += separator.join(outputs)
        jobs = [
            line.replace(sep, "").strip()
            for line in text.splitlines()
            if (sep := "Submitted batch job ") in line
        ]

        if is_interrupted:
            print("\n💀 Interrupted. Kill jobs with:\n$ scancel" + " ".join(jobs))
        elif not jobs:
            print("\n❌ No jobs launched")
        else:
            text += f"{separator}All jobs launched: {' '.join(jobs)}"
            with outfile.open("w") as f:
                f.write(text)
            print("\n\n ✅ Done!")
            if jobs:
                print(util_strings(jobs))
                yml_out = write_exp_yaml_and_jobs(exp_file, outfile, jobs)
                print(
                    "  • Experiment summary YAML in ",
                    f"./{str(yml_out.relative_to(Path.cwd()))}",
                )
    else:
        print("Aborting")
