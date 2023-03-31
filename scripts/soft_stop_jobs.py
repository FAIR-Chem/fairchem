from minydra import resolved_args
from pathlib import Path
import os
import re

if __name__ == "__main__":
    args = resolved_args()
    assert "jobs" in args
    jobs = [
        j.strip()
        for j in str(args.jobs).replace(",", " ").replace("  ", " ").split(" ")
    ]
    runs = Path(os.environ["SCRATCH"]) / "ocp" / "runs"
    outs = [(runs / j / "output-0.txt") for j in jobs]
    confirmed = args.no_confirm or (
        "y"
        in input(f"\nAbout to early-stop jobs:\n {', '.join(jobs)}\nContinue? [y/n]: ")
    )
    if confirmed:
        for out in outs:
            if not out.exists():
                print(f"Output file for job {out.parent.name} not found")
                continue
            stop = re.findall(r"early_stopping_file: (.+)", out.read_text())
            if stop:
                Path(stop[0]).touch()
            else:
                print(f"Early stopping file not found in {str(out)}")
