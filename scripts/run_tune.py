import copy
import os
import time
from pathlib import Path

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
from ocpmodels.trainers import ForcesTrainer
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

def tune_trainable(config, checkpoint_dir=None):
    setup_imports()
    # define trainer defaults are changed to run HPO
    trainer = registry.get_trainer_class(
                config.get("trainer", "simple")
            )(
                task=config["task"],
                model=config["model"],
                dataset=config["dataset"],
                optimizer=config["optim"],
                identifier=config["identifier"],
                run_dir=config.get("run_dir", "./"),
                is_debug=config.get("is_debug", False),
                is_vis=config.get("is_vis", False),
                is_hpo=config.get("is_hpo", True), # hpo 
                print_every=config.get("print_every", 10),
                seed=config.get("seed", 0),
                logger=config.get("logger", None), # hpo
                local_rank=config["local_rank"],
                amp=config.get("amp", False),
                cpu=config.get("cpu", False),
            )
    # add checkpoint here
    # start training
    trainer.train()

def main():
    # parse config
    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()
    config = build_config(args, override_args)
    # add parameters to tune using grid/random search 
    #tune_config = {
    #    "model": {
    #        "hidden_channels": tune.choice([384, 512, 640])   # tune.randint(300, 524),
            #"depth_mlp_edge": tune.randint(1, 10)
    #    },
        #"optim": {
        #    "lr_initial": tune.loguniform(1e-4, 1e-1)
        #}
    #}
    config["model"].update(hidden_channels = tune.choice([256, 384, 512, 640, 704]), 
                           decoder_hidden_channels = tune.choice([256, 384, 512, 640, 704]), 
                           depth_mlp_edge = tune.choice([1, 2, 3, 4, 5]), 
                           depth_mlp_node = tune.choice([1, 2, 3, 4, 5]), 
                           num_interactions = tune.choice([3, 4, 5, 6]))
    #config.update(tune_config)
    #print(config)
    # define scheduler
    scheduler = ASHAScheduler(
        time_attr="steps",
        metric="val_loss",
        mode="min",
        max_t=100000,
        grace_period=2000,
        reduction_factor=4,
        brackets=1)
    # define run parameters
    analysis = tune.run(
        tune_trainable,
        resources_per_trial={"cpu": 8, "gpu": 1},
        config=config,
        fail_fast=True,
        local_dir="/project/projectdirs/m2755/bwood/ocp_hpo/forcenet_hpo_small",
        num_samples=750,
        scheduler=scheduler)
    
    print(
        "Best config is:",
        analysis.get_best_config(
            metric="val_forces_mae", mode="min", scope="last"
        ),
    )
    
if __name__ == "__main__":
    main()
      