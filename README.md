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

### GPU machines

Instructions are for PyTorch 1.4, CUDA 10.0 specifically.

- `conda create -n ocp-models python=3.6`
- `conda activate ocp-models`
- `conda install pytorch=1.4 cudatoolkit=10.0 pyyaml pymatgen ase matplotlib tensorboard pre-commit tqdm -c pytorch -c conda-forge`
- Check if PyTorch is installed with CUDA support:
    - `python -c "import torch; print(torch.cuda.is_available())"` should return true
- Add CUDA to `$PATH` and `$CPATH`
    - `export PATH=/usr/local/cuda/bin:$PATH`
    - `export CPATH=/usr/local/cuda/include:$CPATH`
- Add CUDA to `$LD_LIBRARY_PATH`
    - `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH`
    - `export DYLD_LIBRARY_PATH=/usr/local/cuda/lib:$DYLD_LIBRARY_PATH`
- Ensure that PyTorch and system CUDA versions match
    - `python -c "import torch; print(torch.version.cuda)"` and `nvcc --version` should both return 10.0
- `pip install torch-scatter==latest+cu100 torch-sparse==latest+cu100 torch-cluster==latest+cu100 torch-spline-conv==latest+cu100 -f https://pytorch-geometric.com/whl/torch-1.4.0.html`
- `pip install torch-geometric demjson wandb`
- `pre-commit install`

### CPU-only machines

Please skip the following if you completed the with-GPU installation from above.

```bash
conda env create -f env.cpu.yml
conda activate ocp-models
pre-commit install
```

### Additional experiment-specific setup

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
