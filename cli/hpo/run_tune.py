import os

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from ocpmodels.common.flags import flags
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import build_config, setup_imports


# this function is general and should work for any ocp trainer
def ocp_trainable(config, checkpoint_dir=None) -> None:
    setup_imports()
    # trainer defaults are changed to run HPO
    trainer = registry.get_trainer_class(config.get("trainer", "energy"))(
        task=config["task"],
        model=config["model"],
        dataset=config["dataset"],
        optimizer=config["optim"],
        identifier=config["identifier"],
        run_dir=config.get("run_dir", "./"),
        is_debug=config.get("is_debug", False),
        is_vis=config.get("is_vis", False),
        is_hpo=config.get("is_hpo", True),  # hpo
        print_every=config.get("print_every", 10),
        seed=config.get("seed", 0),
        logger=config.get("logger", None),  # hpo
        local_rank=config["local_rank"],
        amp=config.get("amp", False),
        cpu=config.get("cpu", False),
    )
    # add checkpoint here
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        trainer.load_pretrained(checkpoint)
    # start training
    trainer.train()


# this section defines the hyperparameters to tune and all the Ray Tune settings
# current params/settings are an example for ForceNet
def main() -> None:
    # parse config
    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()
    config = build_config(args, override_args)
    # add parameters to tune using grid or random search
    config["model"].update(
        hidden_channels=tune.choice([256, 384, 512, 640, 704]),
        decoder_hidden_channels=tune.choice([256, 384, 512, 640, 704]),
        depth_mlp_edge=tune.choice([1, 2, 3, 4, 5]),
        depth_mlp_node=tune.choice([1, 2, 3, 4, 5]),
        num_interactions=tune.choice([3, 4, 5, 6]),
    )
    # define scheduler
    scheduler = ASHAScheduler(
        time_attr="steps",
        metric="val_loss",
        mode="min",
        max_t=100000,
        grace_period=2000,
        reduction_factor=4,
        brackets=1,
    )
    # ray init
    # for debug
    # ray.init(local_mode=True)
    # for slurm cluster
    ray.init(
        address="auto",
        _node_ip_address=os.environ["ip_head"].split(":")[0],
        _redis_password=os.environ["redis_password"],
    )
    # define command line reporter
    reporter = CLIReporter(
        print_intermediate_tables=True,
        metric="val_loss",
        mode="min",
        metric_columns={
            "steps": "steps",
            "epochs": "epochs",
            "training_iteration": "training_iteration",
            "val_loss": "val_loss",
            "val_forces_mae": "val_forces_mae",
        },
    )

    # define run parameters
    analysis = tune.run(
        ocp_trainable,
        resources_per_trial={"cpu": 8, "gpu": 1},
        config=config,
        fail_fast=False,
        local_dir=config.get("run_dir", "./"),
        num_samples=500,
        progress_reporter=reporter,
        scheduler=scheduler,
    )

    print(
        "Best config is:",
        analysis.get_best_config(
            metric="val_forces_mae", mode="min", scope="last"
        ),
    )


if __name__ == "__main__":
    main()
