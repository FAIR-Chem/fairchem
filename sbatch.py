from minydra import resolved_args
from pathlib import Path
from datetime import datetime
import os
import subprocess
from shutil import copyfile
import sys
import re

template = """\
#!/bin/bash
{sbatch_params}

# {sbatch_command_line}
# git commit: {git_commit}
# cwd: {cwd}

{git_checkout}
{sbatch_py_vars}

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo "Master port $MASTER_PORT"

cd {code_loc}

{modules}

if {virtualenv}
then
    source {env}/bin/activate
else
    conda activate {env}
fi

srun --output={output} {python_command}
"""


def make_sbatch_params(params):
    """
    Make a string containing the sbatch parameters as
    #SBATCH --param=value

    Args:
        params (dict): dict of param/value to use to submit the job

    Returns:
        str: \n-joined string of #SBATCH --param=value
    """
    sps = []
    for k, v in params.items():
        if v:
            sps.append(f"#SBATCH --{k}={v}")
    return "\n".join(sps) + "\n"


def discover_minydra_defaults():
    """
    Returns a list containing:
    * the path to the shared configs/sbatch/defaults.yaml file
    * the path to configs/sbatch/$USER.yaml file if it exists


    Returns:
        list[pathlib.Path]: Path to the shared defaults and optionnally
            to a user-specific one if it exists
    """
    root = Path(__file__).resolve().parent
    defaults = [root / "configs" / "sbatch" / "defaults.yaml"]
    user_config = root / "configs" / "sbatch" / f"{os.environ['USER']}.yaml"
    if user_config.exists() and user_config.is_file():
        defaults.append(user_config)
    return defaults


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


def get_commit():
    """
    Get the git commit hash of the current repo

    Returns:
        str: Current git commit hash or "unknown" if not in a git repo or
             an error occurred
    """
    try:
        commit = (
            subprocess.check_output("git rev-parse --verify HEAD".split())
            .decode("utf-8")
            .strip()
        )
    except Exception:
        commit = "unknown"
    return commit


def make_sbatch_py_vars(sbatch_py_vars):
    """
    Turns a dict into a series of SBATCH_PY_{key.upper()}=value
    to be parsed by main.py

    Args:
        sbatch_py_vars (dict): sbatch.py-specific env variables

    Returns:
        str: \n-joined string of SBATCH_PY_{key.upper()}=value
    """
    s = ""
    for k, v in sbatch_py_vars.items():
        k = "SBATCH_PY_" + k.replace("-", "_").upper()
        s += k
        if v:
            s += f"={v}"
        else:
            s += "=true"
        s += "\n"
    return s[:-1]


def add_jobid_to_log(j, command_line, exp_name=None):
    """
    Stores the command into a log file. If an exp_name is provided, it will be appended
    tio the appropriate experiment: as a new item if the latest experiment has the
    same name, as a new experiment otherwise.

    Args:
        j (str): SLURM job id
        command_line (str): command-line ran to submit the job
        exp_name (str, optional): Optional experiment the job submission is part of.
            Defaults to None.
    """
    logfile = Path(__file__).resolve().parent / "data" / "sbatch_job_ids.txt"
    if not logfile.exists():
        logfile.touch()
    n = now()
    today = f">>> {n.split('_')[0]}"
    job_line = f"    [{n.split('_')[1]}] {j}\n    {command_line}\n"
    lines = logfile.read_text().splitlines()
    dates = {
        line.strip(): i
        for i, line in enumerate(lines)
        if re.search(r">>> \d{4}-\d{2}-\d{2}", line)
    }
    exp_line = f"  🧪 {exp_name}" if exp_name else ""
    if today in dates:
        day_jobs_line, jobs = [
            (i + dates[today], line)
            for i, line in enumerate(lines[dates[today] :])
            if "All day's jobs:" in line
        ][0]
        lines[day_jobs_line] = jobs + f" {j}"
        if exp_line:
            todays_exps = []
            for line in lines[dates[today] :]:
                if "🧪" in line:
                    todays_exps.append(line)
            if not todays_exps or todays_exps[-1] != exp_line:
                lines += [f"\n{exp_line}"]
        lines += [job_line]
    else:
        lines += [
            f"\n{'-'*len(today)}\n{today}\n{'-'*len(today)}",
            f"All day's jobs: {j}",
        ]
        if exp_line:
            lines += [f"\n{exp_line}"]
        lines += [job_line]

    logfile.write_text("\n".join(lines))


if __name__ == "__main__":
    # has the submission been successful?
    success = False
    sbatch_py_vars = {}
    is_narval = "narval.calcul.quebec" in os.environ.get("HOSTNAME", "")

    # repository root
    root = Path(__file__).resolve().parent
    # parse and resolve args.
    # defaults are loaded and overwritten from the command-line as `arg=value`
    args = resolved_args(defaults=discover_minydra_defaults())
    modules = (
        []
        if not args.modules
        else args.modules.split(",")
        if isinstance(args.modules, str)
        else args.modules
    )
    if args.verbose:
        args.pretty_print()

    # set n_tasks_per node from gres if none is provided
    if args.ntasks_per_node is None:
        if ":" not in args.gres:
            args.ntasks_per_node = 1
        else:
            try:
                args.ntasks_per_node = int(args.gres.split(":")[-1])
            except Exception as e:
                print("Could not parse ntasks_per_node from gres:", e)
                print("Setting to 1")
                args.ntasks_per_node = 1

    # distribute training
    if args.ntasks_per_node > 1 and "--distributed" not in args.py_args:
        if args.sweep:
            sbatch_py_vars["distributed"] = None
            sbatch_py_vars["num-nodes"] = args.nodes
            sbatch_py_vars["num-gpus"] = args.ntasks_per_node
        else:
            args.py_args += " --distributed --num-nodes {} --num-gpus {}".format(
                args.nodes, args.ntasks_per_node
            )

    # add logdir to main.py's command-line arguments
    if "--logdir" not in args.py_args and args.logdir:
        args.py_args += f" --logdir {args.logdir}"
    # add run-dir to main.py's command-line arguments
    if "--run-dir" not in args.py_args and args.logdir:
        args.py_args += f" --run-dir {args.logdir}"

    if "--note" not in args.py_args and args.note:
        note = args.note.replace('"', '\\"')
        args.py_args += f' --note "{note}"'

    git_checkout = f"git checkout {args.git_checkout}" if args.git_checkout else ""
    sbatch_command_line = " ".join(["python"] + sys.argv)

    if args.sweep:
        count = f" --count {args.count}" if args.count else ""
        python_command = f"wandb agent{count} {args.sweep}"
    else:
        python_command = f"python main.py {args.py_args}"

    # conda or pip
    if "virtualenv" in args and args.virtualenv is True:
        virtualenv = "true"
    else:
        virtualenv = "false"

    # create sbatch job submission parameters dictionary
    # to use with make_sbatch_params()
    sbatch_params = {
        "job-name": args.job_name,
        "nodes": args.nodes or 1,
        "ntasks-per-node": args.ntasks_per_node,
        "partition": args.partition,
        "cpus-per-task": args.cpus,
        "mem": args.mem,
        "gres": args.gres,
        "output": str(resolve(args.output)),
    }
    if args.time:
        sbatch_params["time"] = args.time
    if is_narval:
        del sbatch_params["partition"]
        sbatch_params["account"] = "rrg-bengioy-ad_gpu"

    if "a100" in args.env:
        modules += ["cuda/11.2"]

    # format string template with defaults + command-line args
    script = template.format(
        code_loc=(str(resolve(args.code_loc)) if args.code_loc else str(root)),
        cwd=str(Path.cwd()),
        debug_dir="$SCRATCH/ocp/runs/$SLURM_JOBID",
        env=args.env,
        git_checkout=git_checkout,
        git_commit=get_commit(),
        modules="\nmodule load ".join([""] + modules),
        output=str(resolve(args.output)),
        python_command=python_command,
        sbatch_command_line=sbatch_command_line,
        sbatch_params=make_sbatch_params(sbatch_params),
        sbatch_py_vars=make_sbatch_py_vars(sbatch_py_vars),
        virtualenv=virtualenv,
    )

    # default script path to execute `sbatch {script_path}/script_{now()}.sh`
    data_path = root / "data" / "sbatch_scripts"
    data_path.mkdir(parents=True, exist_ok=True)
    # write script in data_path or args.script_path if it has been provided
    script_path = args.script_path or data_path
    script_path = Path(script_path).resolve()

    # make script's parent dir if it does not exist
    if script_path.is_dir() and not script_path.exists():
        script_path.mkdir(parents=True)

    # add default name if a fodler path (not a file path) was provided
    if script_path.is_dir():
        script_path /= f"sbatch_script_{now()}.sh"

    # make parent directory if file path has been provided and its parent does not exist
    if script_path.is_file() and not script_path.parent.exists():
        script_path.parent.mkdir(parents=True)

    # write filled template to script text file
    with script_path.open("w") as f:
        f.write(script)

    # command to request the job
    array = f" --array={args.array}" if args.array else ""
    command = f"sbatch{array} {str(script_path)}"
    if args.verbose:
        print(f"Executing:\n{command}")
        print(f"\nFile content:\n{'=' * 50}\n{script}{'=' * 50}\n")

    # dev mode: don't actually exectue
    if args.dev:
        print("\nDev mode: not actually executing the command 🤓\n")
    else:
        # not dev mode: run the command, make directories
        out = subprocess.check_output(command.split(" ")).decode("utf-8").strip()
        jobid = out.split(" job ")[-1].strip()
        success = out.startswith("Submitted batch job")

        # make slurm output directory based on job id
        if "/%j/" in args.output and success:
            args.output = args.output.replace("/%j/", f"/{jobid.strip()}/")
            output_parent = resolve(args.output).parent
            if not output_parent.exists():
                if args.verbose:
                    print("Creating directory", str(output_parent))
                output_parent.mkdir(parents=True, exist_ok=True)
            copyfile(script_path, output_parent / script_path.name)
        if not args.verbose:
            print("Submitted batch job", jobid)
        add_jobid_to_log(jobid, sbatch_command_line, args.exp_name)

    if args.dev:
        pass
    else:
        # print command result
        if args.verbose:
            print(f"\n{out}\n")
        if success and not resolve(args.output).parent.exists():
            print(
                "\n >>>WARNING slurm output folder does not exist:",
                f"--output={args.output}",
            )
