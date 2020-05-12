import copy
import os
import sys

import yaml

import ray
from ocpmodels.common.flags import flags
from ocpmodels.common.registry import registry
from ocpmodels.trainers import TuneHPOTrainer
from ray import tune
from ray.tune.logger import CSVLogger, JsonLogger, TBXLogger
from ray.tune.schedulers import AsyncHyperBandScheduler
from wandb.ray import WandbLogger


# this function converts a tune string to a tune function
# it is necessary because of YAML safe_loading
def tune_str_to_func(config):
    hpo_config = copy.deepcopy(config)
    for key_i, dict_i in hpo_config.items():
        if isinstance(dict_i, dict):
            for key_j, val_j in dict_i.items():
                if isinstance(val_j, str) and "tune." in val_j:
                    dict_i.update({key_j: eval(val_j)})
    return hpo_config


if __name__ == "__main__":
    parser = flags.get_parser()
    args = parser.parse_args()

    # get cwd, assumes it to be in the scripts dir of this package
    cwd = os.getcwd()
    # load configs from yaml files.
    config = yaml.safe_load(open(args.config_yml, "r"))
    includes = config.get("includes", [])
    # update include paths from rel paths to abs paths
    includes = [
        os.path.abspath(os.path.join(cwd, "..", rel_path))
        for rel_path in includes
    ]
    if not isinstance(includes, list):
        raise AttributeError(
            "Includes must be a list, {} provided".format(type(includes))
        )

    for include in includes:
        include_config = yaml.safe_load(open(include, "r"))
        config.update(include_config)

    config.pop("includes")
    config["cmd"] = args.__dict__

    # converts tune str from YAML to tune functions
    hpo_config = tune_str_to_func(config)
    # update all remaining rel paths to abs paths
    hpo_config["dataset"]["src"] = os.path.abspath(
        os.path.join(cwd, "..", hpo_config["dataset"]["src"])
    )
    hpo_config["tune"]["local_dir"] = os.path.abspath(
        os.path.join(cwd, "..", hpo_config["tune"]["local_dir"])
    )

    # initialize and run Tune.
    if hpo_config["tune"]["ray_init_default"]:
        ray.init()
    else:
        ray.init(
            address="auto",
            node_ip_address=os.environ["ip_head"].split(":")[0],
            redis_password=os.environ["redis_password"],
        )
    # define the hpo scheduler ASHA or AHB depending on the number of brackets
    async_hb_scheduler = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        metric="validation_mae",
        mode="min",
        max_t=hpo_config["tune"]["max_t"],
        grace_period=1,
        reduction_factor=4,
        brackets=1,
    )
    analysis = tune.run(
        TuneHPOTrainer,
        scheduler=async_hb_scheduler,
        resources_per_trial=hpo_config["tune"]["resources_per_trial"],
        num_samples=hpo_config["tune"]["num_samples"],
        checkpoint_at_end=True,
        checkpoint_freq=hpo_config["tune"]["checkpoint_freq"],
        sync_on_checkpoint=False,
        reuse_actors=True,
        config=hpo_config,
        loggers=[JsonLogger, CSVLogger, TBXLogger, WandbLogger],
        local_dir=hpo_config["tune"]["local_dir"],
        resume=hpo_config["tune"]["resume"],
    )
    print(
        "Best config is:",
        analysis.get_best_config(
            metric="validation_mae", mode="min", scope="last"
        ),
    )
