import ray
import yaml
from ray import tune

from baselines.common.flags import flags
from baselines.trainers import TuneHPOTrainer

if __name__ == "__main__":
    parser = flags.get_parser()
    args = parser.parse_args()

    # Loads configs from yaml files.
    config = yaml.safe_load(open(args.config_yml, "r"))
    includes = config.get("includes", [])
    if not isinstance(includes, list):
        raise AttributeError(
            "Includes must be a list, {} provided".format(type(includes))
        )

    for include in includes:
        include_config = yaml.safe_load(open(include, "r"))
        config.update(include_config)

    config.pop("includes")
    config["cmd"] = args.__dict__

    # Params to sweep over. Follow the same format as in the yaml files.
    config_hyperparam_sweep = {
        "optim": {"batch_size": 80, "lr_initial": tune.uniform(0.005, 0.1)},
        "model_attributes": {
            "atom_embedding_size": tune.randint(46, 64),
            "num_graph_conv_layers": 6,
            "fc_feat_size": 128,  # projection layer after conv + pool layers
            "num_fc_layers": 4,
        },
    }
    config.update(config_hyperparam_sweep)

    # Initialize and run Tune.
    ray.init()
    analysis = tune.run(
        TuneHPOTrainer,
        stop={"training_iteration": 60},
        resources_per_trial={"cpu": 5, "gpu": 1},
        num_samples=25,
        config=config,
        # TODO(abhshkdz): include this summary directory in config.
        local_dir="/global/homes/b/bwood/machine_learning/hpo/results/H_4k_lr_atom_feat_25",
    )
    print(
        "Best config is:",
        analysis.get_best_config(
            metric="validation_mae", mode="min", scope="last"
        ),
    )
