# Open-Catalyst-Project Models

Implements the following baselines that take arbitrary chemical structures as
input to predict material properties:
- [Crystal Graph Convolutional Neural Networks (CGCNN)](https://link.aps.org/doi/10.1103/PhysRevLett.120.145301).
- Modified version of CGCNN as proposed in [Gu et al., 2020](https://pubs.acs.org/doi/abs/10.1021/acs.jpclett.0c00634).
- [Path-Augmented Graph Transformer Network](https://arxiv.org/abs/1905.12712).
Also related to [Graph Attention Networks](https://arxiv.org/abs/1710.10903) and
[Graph Transformer](https://openreview.net/forum?id=HJei-2RcK7).

##  Installation

[last updated May 21, 2020]

The easiest way of installing prerequisites is via [conda](https://conda.io/docs/index.html).
After installing [conda](http://conda.pydata.org/), run the following commands
to create a new [environment](https://conda.io/docs/user-guide/tasks/manage-environments.html)
named `ocp-models` and install dependencies:

### Pre-install step
Install `conda-merge`:
```bash
pip install conda-merge
```
If you're using system `pip`, then you may want to add the `--user` flag to avoid using `sudo`.
Check that you can invoke `conda-merge`:
```
$ conda-merge -h
usage: conda-merge [-h] files [files ...]

Tool to merge environment files of the conda package manager.

Given a list of environment files, print a unified environment file.
Usage: conda-merge file1 file2 ... [> unified-environment]

Merge strategy for each part of the definition:
  name: keep the last name, if any is given (according to the order of the files).
  channels: merge the channel priorities of all files and keep each file's priorities
    in the same order. If there is a collision between files, report an error.
  dependencies: merge the dependencies and remove duplicates, sorts alphabetically.
    conda itself can handle cases like [numpy, numpy=1.7] gracefully so no need
    to do that. You may beautify the dependencies by hand if you wish.
    The script also doesn't detect collisions, relying on conda to point that out.

positional arguments:
  files

optional arguments:
  -h, --help  show this help message and exit
```

### GPU machines

Instructions are for PyTorch 1.4, CUDA 10.0 specifically.

First, check that CUDA is in your `PATH` and `LD_LIBRARY_PATH`, e.g.
```
$ echo $PATH | tr ':' '\n' | grep cuda
/public/apps/cuda/10.0/bin
$ echo $LD_LIBRARY_PATH | tr ':' '\n' | grep cuda
/public/apps/cuda/10.0/lib64
```
The exact paths may differ on your system. Then install the dependencies:
```bash
conda-merge env.common.yml env.gpu.yml > env.yml
conda env create -f env.yml
```
Activate the conda environment with `conda activate ocp-models`.
Finally, install the pre-commit hooks:
```bash
pre-commit install
```

### CPU-only machines

Please skip the following if you completed the with-GPU installation from above.

```bash
conda-merge env.common.yml env.cpu.yml > env.yml
conda env create -f env.yml
conda activate ocp-models
pre-commit install
```

### Additional experiment-specific setup

Append `env.extras.yml` to the `conda-merge` command, e.g. for CPU machines:
```
conda-merge env.common.yml env.cpu.yml env.extras.yml > env.yml
```

The extras contain:
- [Kevin Tran](https://github.com/ktran9891)'s Convolution-Fed Gaussian Process
  (CFGP) pipeline requires `gpytorch`, installable via `conda install gpytorch -c conda-forge`.
- Hyperparameter optimization (HPO) requires `Tune`, installable via `pip install ray[tune]`.

## Usage

### Download the datasets

For now, we are working with the following datasets:
- `ulissigroup_co`: dataset of DFT results for CO adsorption on various slabs (shared by Junwoong Yoon) already in pytorch-geometric format.
- `gasdb`: tiny dataset of DFT results for CO, H, N, O, and OH adsorption on various slabs (shared by Kevin Tran) in raw ase format.

To download the datasets:

```
cd data
./download_data.sh
```

### Train models to predict energies from structures

To quickly get started with training a CGCNN model on the `gasdb` dataset
with reasonable defaults, take a look at
[scripts/train_example.py](https://github.com/Open-Catalyst-Project/baselines/blob/master/scripts/train_example.py)
(reproduced below):

```
from ocpmodels.trainers import SimpleTrainer

task = {
    "dataset": "gasdb",
    "description": "Binding energy regression on a dataset of DFT results for CO, H, N, O, and OH adsorption on various slabs.",
    "labels": ["binding energy"],
    "metric": "mae",
    "type": "regression",
}

model = {
    "name": "cgcnn",
    "atom_embedding_size": 64,
    "fc_feat_size": 128,
    "num_fc_layers": 4,
    "num_graph_conv_layers": 6,
}

dataset = {
    "src": "data/data/gasdb",
    "train_size": 800,
    "val_size": 100,
    "test_size": 100,
}

optimizer = {
    "batch_size": 10,
    "lr_gamma": 0.1,
    "lr_initial": 0.001,
    "lr_milestones": [100, 150],
    "max_epochs": 50,
    "warmup_epochs": 10,
    "warmup_factor": 0.2,
}

trainer = SimpleTrainer(
    task=task,
    model=model,
    dataset=dataset,
    optimizer=optimizer,
    identifier="my-first-experiment",
)

trainer.train()

predictions = trainer.predict("data/data/gasdb")
```

For more advanced usage and digging deeper into default parameters, take a look
at [`BaseTrainer`](https://github.com/Open-Catalyst-Project/baselines/blob/master/ocpmodels/trainers/base_trainer.py). To use `BaseTrainer` to train a CGCNN model
on the `ulissigroup_co` CO adsorption data to predict binding energy (with
default params):

```bash
python main.py --identifier my-first-experiment --config-yml configs/ulissigroup_co/cgcnn.yml
```

See [`configs/ulissigroup_co/base.yml`](https://github.com/Open-Catalyst-Project/baselines/blob/master/configs/ulissigroup_co/base.yml) and [`configs/ulissigroup_co/cgcnn.yml`](https://github.com/Open-Catalyst-Project/baselines/blob/master/configs/ulissigroup_co/cgcnn.yml) for dataset, model and optimizer parameters.

## Acknowledgements

- This codebase was initially forked from [CGCNN](https://github.com/txie-93/cgcnn)
by [Tian Xie](http://txie.me), but has undergone significant changes since.
- A lot of engineering ideas have been borrowed from [github.com/facebookresearch/pythia](https://github.com/facebookresearch/pythia).

## License

TBD
