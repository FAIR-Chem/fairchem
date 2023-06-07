import argparse
import logging
from pathlib import Path

from ll import Runner, Trainer

from ocpmodels.trainers.base import S2EFConfig, S2EFModule

logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--config", type=Path, help="Path to config file")
    # Add two actions: local and submit
    subparsers = parser.add_subparsers(dest="action", required=True)
    _ = subparsers.add_parser("local")
    submit = subparsers.add_parser("submit")
    _ = submit.add_argument(
        "--gpus", type=int, required=True, help="Number of GPUs"
    )
    _ = submit.add_argument(
        "--nodes", type=int, required=True, help="Number of nodes"
    )
    _ = submit.add_argument(
        "--partition",
        type=str,
        required=True,
        help="SLURM partition to submit to.",
    )
    _ = submit.add_argument(
        "--cpus-per-task",
        type=int,
        default=-1,
        help="Number of CPUs per task, or -1 to infer based on `num_workers` in the config.",
    )
    _ = submit.add_argument(
        "--snapshot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to take a snapshot of the current code and use it for the run.",
    )
    args = parser.parse_args()
    return args


def main():
    # Parse the arguments
    args = parse_args()

    # Parse the config
    config = S2EFConfig.from_file(args.config)

    # Define the runner main function.
    # This is called locally when using runner.local(config)
    # or on every GPU when using runner.submit([config, ...]).
    def run(config: S2EFConfig):
        model = S2EFModule(config)
        trainer = Trainer(config)
        trainer.fit(model)

    # Create the runner and run locally
    runner = Runner(run)
    match args.action:
        case "local":
            # Run locally
            runner.local(config)
        case "submit":
            # If cpus_per_task is not specified, use the number of workers
            cpus_per_task: int = args.cpus_per_task
            if cpus_per_task == -1:
                cpus_per_task = config.data.num_workers

            # Submit to SLURM
            jobs = runner.submit(
                [config],
                gpus=args.gpus,
                nodes=args.nodes,
                cpus_per_task=cpus_per_task,
                partition=args.partition,
                snapshot=args.snapshot,
            )

            # Print the job IDs
            for job in jobs:
                log.critical(
                    f"Submitted job {job.job_id} to {args.partition}."
                )

        case _:
            raise ValueError(f"Invalid action {args.action}")


if __name__ == "__main__":
    main()
