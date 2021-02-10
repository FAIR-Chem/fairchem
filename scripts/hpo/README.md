# Running Hyperparameter Optimization with Ray Tune

## Installation
Activate ocp-models conda env

`pip install -U [link to wheel] --upgrade-strategy only-if-needed`

link to OS dependent wheel can be found [here](https://docs.ray.io/en/master/installation.html#daily-releases-nightlies)

## Usage with Slurm

1. Make necessary changes to `run_tune.py` and `slurm/submit-ray-cluster.sbatch`

    Example `run_tune.py` updates
    - choose search and scheduler algorithms and set associated parameters (see [Ray Tune docs](https://docs.ray.io/en/master/tune/index.html) for details)
    - set the resources to use per individual trial

    Example `slurm/submit-ray-cluster.sbatch` updates
    - load modules or set conda env
    - change the total run time and resources to use

2. submit using `sbatch slurm/submit-ray-cluster.sbatch`

Slurm scripts taken from https://github.com/NERSC/slurm-ray-cluster

For usage with other cluster managers or cloud resources please refer to the
[Distributed Ray Docs](https://docs.ray.io/en/master/cluster/index.html#)

## Current Limitations (02/09/2021)

- Only single GPU training is supported
- All models should work but must use the forces_trainer
