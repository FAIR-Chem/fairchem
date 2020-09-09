import datetime
import glob
import os

import submitit
import torch

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
        model=config["model_attributes"],
        dataset=config["dataset"],
        optimizer=config["optim"],
        identifier=config["identifier"],
        run_dir=config.get("run_dir", "./"),
        is_debug=config.get("is_debug", False),
        is_vis=config.get("is_vis", False),
        print_every=config.get("print_every", 10),
        seed=config.get("seed", 0),
        logger=config.get("logger", "tensorboard"),
        local_rank=config["local_rank"],
    )
    trainer.load_pretrained(config["model_path"], ddp_to_dp=False)
    trainer.train()
    distutils.synchronize()


def distributed_main(config):
    distutils.setup(config)
    main(config)
    distutils.cleanup()


if __name__ == "__main__":
    parser = flags.get_parser()
    args = parser.parse_args()

    if args.submit:  # Run on cluster
        slurm_job_id = "29684753_"
        configs = []
        for job in range(9):
            trial = "schnet_all"
            log_path = glob.glob(
                f"/private/home/mshuaibi/baselines/logs/{trial}/{slurm_job_id}{job}/*.out"
            )[0]
            log = open(log_path, "r").read().splitlines()
            for i in log:
                if "checkpoint_dir" in i:
                    checkpoint_path = os.path.join(
                        log[4].strip().split()[1], "checkpoint.pt"
                    )
                    break
            assert os.path.isfile(checkpoint_path)
            checkpoint = torch.load(checkpoint_path)
            config = checkpoint["config"]
            config["dataset"]["src"] = os.path.join(
                "/private/home/mshuaibi/baselines/", config["dataset"]["src"]
            )
            config["dataset"] = [config["dataset"]]
            config["model_path"] = checkpoint_path
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            config["trainer"] = "dist_forces"
            config["model_attributes"].update({"name": config["model"]})
            config["submit"] = args.submit
            config["identifier"] = f"{args.identifier}_run{job}"
            config["seed"] = args.seed
            config["is_debug"] = args.debug
            config["run_dir"] = args.run_dir
            config["is_vis"] = args.vis
            config["dataset"].append(
                {
                    "src": "/private/home/mshuaibi/baselines/data/data/ocp_s2ef/val/is_1M/",
                    "normalize_labels": False,
                }
            )
            config["print_every"] = args.print_every
            if args.distributed:
                config["local_rank"] = args.local_rank
                config["distributed_port"] = args.distributed_port
                config["world_size"] = args.num_nodes * args.num_gpus
                config["distributed_backend"] = args.distributed_backend
            configs.append(config)

        executor = submitit.AutoExecutor(folder=args.logdir / "%j")
        executor.update_parameters(
            name=config["identifier"],
            mem_gb=args.slurm_mem,
            timeout_min=args.slurm_timeout * 60,
            slurm_partition=args.slurm_partition,
            gpus_per_node=args.num_gpus,
            cpus_per_task=(args.num_workers + 1),
            tasks_per_node=(args.num_gpus if args.distributed else 1),
            nodes=args.num_nodes,
            slurm_additional_parameters={"begin": f"now+{args.begin*3600}"},
            slurm_comment=args.slurm_comment,
        )
        if args.distributed:
            jobs = executor.map_array(distributed_main, configs)
        else:
            jobs = executor.map_array(main, configs)
        print("Submitted jobs:", ", ".join([job.job_id for job in jobs]))
        log_file = save_experiment_log(args, jobs, configs)
        print(f"Experiment log saved to: {log_file}")
