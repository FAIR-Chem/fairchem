# Running Hyperparameter Optimization with Ray Tune

# Installation
`pip install ray ray[tune]`

## Model config considerations

The current Ray Tune implementation uses the standard OCP config. However, there are a number of config settings that require additional consideration.

```
logger: None
is_hpo: True

optim:
  …
  eval_every: (int) number of steps
  checkpoint_every: (int: optional) number of steps
```
The first two are easily set. The logger is set to None because Ray Tune internally handles the logging.

The `eval_every` setting is case specific and will likely require some experimentation. The `eval_every` flag sets how often the validation set is run in number of steps. Depending on the OCP model and dataset of interest, training for a single epoch can take a substantial amount of time. However, to take full advantage of HPO methods that minimize compute by terminating trials that are not promising, such as successive halving, communication of train and val metrics need to happen on shorter timescales. Paraphrasing the Ray Tune docs, `eval_every` should be set large enough to avoid overheads but short enough to report progress periodically — minutes timescale recommended.

The `eval_every` setting is only available for the force trainer so when using the energy trainer validation will be run and reporting to Ray Tune will occur on a per epoch basis.

The `checkpoint_every` setting determines how frequently, in steps, Ray Tune will write a checkpoint. Checkpointing can create a lot of overhead for certain HPO methods so do not do it too frequently. The default behavior is no checkpointing.

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

## Examples

1. Asynchronous Successive Halving — `ocp/scripts/hpo/run_tune.py`
2. Population Based Training — `ocp/scripts/hpo/run_tune_pbt.py`

## Testing/Debugging Ray Tune

- In `run_tune.py` set `ray.init(local_mode=True)`
- run `python path_to/run_tune.py --mode train --config-yml path_to/config --run_dir path_to_run_dir`
