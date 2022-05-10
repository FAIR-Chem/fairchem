from minydra import resolved_args
from pathlib import Path
from datetime import datetime
import os

template = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --ntasks={ntasks}
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --gres={gres}
#SBATCH --output={output}
#SBATCH --error={error}
{time}

module load anaconda/3
conda activate {env}
python main.py {py_args}
"""


def now():
    return str(datetime.now()).split(".")[0].replace(":", "-").replace(" ", "_")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    args = resolved_args(
        defaults=root / "configs" / "sbatch" / "defaults.yaml"
    ).pretty_print()

    script = template.format(
        cpus=args.cpus or 4,
        env=args.env or "ocp",
        error=args.error or "slurm_res/error-job.txt",
        gres=args.gres or "gpu:4",
        job_name=args.job_name or "ocp-script",
        mem=args.time or "32GB",
        ntasks=args.ntasks or 1,
        output=args.output or "slurm_res/output-job.txt",
        partition=args.partition or "long",
        py_args=args.py_args or "",
        time="" if not args.time else f"#SBATCH --time={args.time}",
    )

    data_path = root / "data" / "sbatch_scripts"
    data_path.mkdir(parents=True, exist_ok=True)
    script_path = args.script_path or data_path
    script_path = Path(script_path).resolve()

    if script_path.is_dir() and not script_path.exists():
        script_path.mkdir(parents=True)

    if script_path.is_dir():
        script_path /= f"script_{now()}.sh"

    if script_path.is_file() and not script_path.parent.exists():
        script_path.parent.mkdir(parents=True)

    with script_path.open("w") as f:
        f.write(script)

    command = f"sbatch {str(script_path)}"
    print(f"Executing:\n{command}")
    print(f"\nFile content:\n{'=' * 50}\n{script}{'=' * 50}\n")
    if args.dev:
        print("\nDev mode: not actually executing the command 🤓\n")
    else:
        os.system(command)
