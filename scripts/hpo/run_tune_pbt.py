import logging
import os

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import PopulationBasedTraining

from ocpmodels.common.flags import flags
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import build_config, setup_imports


# this function is general and should work for any ocp trainer
def ocp_trainable(trainer_config, checkpoint_dir=None):
    setup_imports()
    # update trainer_config for PBT learning rate
    trainer_config["optim"].update(lr_initial=trainer_config["lr"])
    # trainer defaults are changed to run HPO
    trainer = registry.get_trainer_class(trainer_config["trainer"])(
        **trainer_config,
        is_debug=trainer_config.get("is_debug", False),
        is_vis=trainer_config.get("is_vis", False),
        is_hpo=trainer_config.get("is_hpo", True),  # hpo
        print_every=trainer_config.get("print_every", 100),
        logger=trainer_config.get("logger", None),  # hpo
    )
    # add checkpoint here
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        trainer.load_pretrained(checkpoint)
    # set learning rate
    for g in trainer.optimizer.param_groups:
        g["lr"] = trainer_config["lr"]
    # start training
    trainer.train()


# this section defines all the Ray Tune run parameters
def main():
    # parse config
    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()
    config = build_config(args, override_args)
    # add parameters to tune using grid or random search
    config["lr"] = tune.loguniform(0.0001, 0.01)
    # define scheduler
    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="val_loss",
        mode="min",
        perturbation_interval=1,
        hyperparam_mutations={
            "lr": tune.loguniform(0.000001, 0.01),
        },
    )
    # ray init
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
            "act_lr": "act_lr",
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
        stop={"epochs": 12},
        # time_budget_s=28200,
        fail_fast=False,
        local_dir=config.get("run_dir", "./"),
        num_samples=8,
        progress_reporter=reporter,
        scheduler=scheduler,
    )

    print(
        "Best config is:",
        analysis.get_best_config(metric="val_forces_mae", mode="min", scope="last"),
    )


if __name__ == "__main__":
    main()
