# Documentation for model configurations

In the OCP baselines repository, **a config is the collection of all the necessary
settings to run a particular model for a given task**. In general, the config is made up of a number of constituents:
- task
- model
- dataset
- optimizer
- identifier
- run-dir
- is_debug
- is_vis
- print_every
- seed
- logger
- local_rank
- amp
- mode

Configs can be a dictionary read from a yaml file or given directly to the trainer of interest.

### task (dict)
key (str): dataset \
value (str): single_point_lmdb or trajectory_lmdb \
description: single_point_lmdb for single points used with `EnergyTrainer` and `ForcesTrainer`
and trajectory_lmdb for trajectories used with `ForcesTrainer`

key (str): description \
value (str) \
description: user description of the model/task being run

key (str): type \
value (str): regression \
description:

key (str): metric \
value (str): mae \
description:

key (str): labels \
value (list of str): potential energy \
description:

key (str): grad_input \
value (str): atomic forces \
description:

key (str): train_on_free_atoms \
value (bool) \
description:

key (str): eval_on_free_atoms \
value (bool) \
description:
