import os
from sys import exit
from time import sleep

from minydra import resolved_args


def n_jobs():
    return len(os.popen("squeue -u $USER").read().splitlines()) - 1


if __name__ == "__main__":
    args = resolved_args()
    assert "exp" in args
    hours = args.get("hours", 1)
    min_jobs = args.get("min_jobs", 1)
    cmd = f"python launch_exp.py exp={args.exp} no_confirm='y'" + "n_jobs={new_jobs}"
    print(
        f"\nChecking every {hours} hours for new jobs to launch for exp {args.exp}",
        f"so that you always have at least {min_jobs} jobs running\n",
    )

    if "y" not in input("Continue? [y/n]: "):
        exit()

    i = 0

    try:
        while True:
            j = n_jobs()
            print(f"\nNumber of jobs at iteration {i}: {j}")
            if j < min_jobs:
                new_jobs = min_jobs - j
                print(f"  Launching {new_jobs} jobs at iteration {i}")
                os.system(cmd.format(new_jobs=new_jobs))
            i += 1
            sleep(hours * 60 * 60)
    except KeyboardInterrupt:
        print("Exiting...")
