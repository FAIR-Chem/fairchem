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
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --gres={gres}
#SBATCH --output={output}
{time}

# {sbatch_command_line}
# git commit: {git_commit}
# cwd: {cwd}

{git_checkout}
{sbatch_py_vars}

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo "Master port $MASTER_PORT"

cd {code_loc}

if {virtualenv}
then
    source {env}/bin/activate
else
    export LD_DEBUG=libs,files
    module load anaconda/3 &>> {debug_dir}/libtinfo-hunt.txt
    unset LD_DEBUG
    conda activate {env}
fi

srun --output={output} {python_command}
"""


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


def add_jobid_to_log(j, command_line):
    logfile = Path(__file__).resolve().parent / "data" / "sbatch_job_ids.txt"
    if not logfile.exists():
        logfile.touch()
    n = now()
    today = f">>> {n.split('_')[0]}"
    job_line = f"    [{n.split('_')[1]}] {j} | {command_line}"
    lines = logfile.read_text().splitlines()
    dates = {
        line.strip(): i
        for i, line in enumerate(lines)
        if re.search(r">>> \d{4}-\d{2}-\d{2}", line)
    }
    if today in dates:
        line, jobs = [
            (i, line)
            for i, line in enumerate(lines)
            if i > dates[today] and "All day's jobs:" in line
        ][0]
        lines[line] = jobs + f" {j}"
        lines += [job_line]
    else:
        lines += [
            f"\n{'-'*len(today)}\n{today}\n{'-'*len(today)}",
            f"All day's jobs: {j}",
            job_line,
        ]

    logfile.write_text("\n".join(lines))


if __name__ == "__main__":
    # has the submission been successful?
    success = False
    sbatch_py_vars = {}

    # repository root
    root = Path(__file__).resolve().parent
    # parse and resolve args.
    # defaults are loaded and overwritten from the command-line as `arg=value`
    args = resolved_args(defaults=discover_minydra_defaults())
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
            args.py_args += f" --distributed --num-nodes {args.nodes} --num-gpus {args.ntasks_per_node}"

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

    # format string template with defaults + command-line args
    script = template.format(
        cpus=args.cpus,
        cwd=str(Path.cwd()),
        env=args.env
        if "a100" not in args.env
        else f"{args.env}\n    module load cuda/11.2",
        git_commit=get_commit(),
        git_checkout=git_checkout,
        gres=args.gres,
        job_name=args.job_name,
        mem=args.mem,
        nodes=args.nodes or 1,
        ntasks_per_node=args.ntasks_per_node,
        ntasks=args.ntasks,
        output=str(resolve(args.output)),
        partition=args.partition,
        python_command=python_command,
        sbatch_command_line=" ".join(["python"] + sys.argv),
        sbatch_py_vars=make_sbatch_py_vars(sbatch_py_vars),
        time="" if not args.time else f"#SBATCH --time={args.time}",
        virtualenv=virtualenv,
        debug_dir="$SCRATCH/ocp/runs/$SLURM_JOBID",
        code_loc=(str(resolve(args.code_loc)) if args.code_loc else str(root)),
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
        add_jobid_to_log(jobid)

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
