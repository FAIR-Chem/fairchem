import submitit

from ocpmodels.common import distutils
from ocpmodels.common.flags import flags
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    build_config,
    create_grid,
    save_experiment_log,
    setup_imports,
)


def main(config):
    setup_imports()
    trainer = registry.get_trainer_class(config.get("trainer", "simple"))(
        task=config["task"],
        model=config["model"],
        dataset=config["dataset"],
        optimizer=config["optim"],
        identifier=config["identifier"],
        is_debug=config.get("is_debug", False),
        is_vis=config.get("is_vis", False),
        print_every=config.get("print_every", 10),
        seed=config.get("seed", 0),
        logger=config.get("logger", "tensorboard"),
        local_rank=config["local_rank"]
    )
    trainer.train()
    distutils.synchronize()


def distributed_main(config):
    distutils.setup(config)
    main(config)
    distutils.cleanup()


if __name__ == "__main__":
    parser = flags.get_parser()
    args = parser.parse_args()
    config = build_config(args)

    if args.submit:  # Run on cluster
        if args.sweep_yml:  # Run grid search
            configs = create_grid(config, args.sweep_yml)
        else:
            configs = [config]

        print(f"Submitting {len(configs)} jobs")
        executor = submitit.AutoExecutor(folder=args.logdir / "%j")
        executor.update_parameters(
            name=args.identifier,
            mem_gb=args.slurm_mem,
            timeout_min=args.slurm_timeout * 60,
            slurm_partition=args.slurm_partition,
            gpus_per_node=args.num_gpus,
            cpus_per_task=(args.num_workers + 1),
            tasks_per_node=(args.num_gpus if args.distributed else 1),
            nodes=args.num_nodes,
        )
        if args.distributed:
            jobs = executor.map_array(distributed_main, configs)
        else:
            jobs = executor.map_array(main, configs)
        print("Submitted jobs:", ", ".join([job.job_id for job in jobs]))
        log_file = save_experiment_log(args, jobs, configs)
        print(f"Experiment log saved to: {log_file}")

    else:  # Run locally
        if args.distributed:
            distributed_main(config)
        else:
            main(config)
