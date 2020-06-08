# Open-Catalyst-Project Models

Implements the following baselines that take arbitrary chemical structures as
input to predict material properties:
- [Crystal Graph Convolutional Neural Networks (CGCNN)](https://link.aps.org/doi/10.1103/PhysRevLett.120.145301).
- Modified version of CGCNN as proposed in [Gu et al., 2020](https://pubs.acs.org/doi/abs/10.1021/acs.jpclett.0c00634).
- [Path-Augmented Graph Transformer Network](https://arxiv.org/abs/1905.12712).
Also related to [Graph Attention Networks](https://arxiv.org/abs/1710.10903) and
[Graph Transformer](https://openreview.net/forum?id=HJei-2RcK7).

##  Installation

[last updated June 8, 2020]

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
Check that you can invoke `conda-merge` by running `conda-merge -h`.

### GPU machines

Instructions are for PyTorch 1.5, CUDA 10.1 specifically.

First, check that CUDA is in your `PATH` and `LD_LIBRARY_PATH`, e.g.
```
$ echo $PATH | tr ':' '\n' | grep cuda
/public/apps/cuda/10.1/bin
$ echo $LD_LIBRARY_PATH | tr ':' '\n' | grep cuda
/public/apps/cuda/10.1/lib64
```
The exact paths may differ on your system. Then install the dependencies:
```bash
conda-merge env.common.yml env.gpu.yml > env.yml
conda env create -f env.yml
```
Activate the conda environment with `conda activate ocp-models`.
Install this package with `pip install -e .`.
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
pip install -e .
pre-commit install
```

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
