from os.path import expandvars
from tqdm import tqdm
from pathlib import Path
from minydra import resolved_args
import re
from shutil import rmtree


def is_job_dir(path):
    return path.is_dir() and re.match(r"\d+", path.name)


if __name__ == "__main__":
    args = resolved_args(
        defaults={
            "path": "$SCRATCH/ocp/runs",
        }
    )

    path = Path(expandvars(args.path)).expanduser().resolve()
    dirs = list(path.iterdir())
    for folder in tqdm(dirs, desc="Cleaning checkpoints"):
        if not is_job_dir(folder):
            continue
        ckpts = list(folder.glob("checkpoints/*.pt"))
        if not ckpts:
            rmtree(folder)
            continue
        max_ckpt = max(
            ckpts, key=lambda x: int(x.stem.split("-")[-1] if "-" in x.stem else -1)
        )
        best_ckpt = "best_checkpoint.pt"

        for ckpt in ckpts:
            if ckpt.name != best_ckpt and ckpt != max_ckpt:
                ckpt.unlink()

        for f in folder.glob("**/*"):
            if "logs" in f.parts:
                continue
            if f.is_dir():
                continue
            with f.open("rb"):
                pass
