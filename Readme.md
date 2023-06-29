# FAENet

This repository contains the code used to *train* the models reported in [FAENet: Frame Averaging Equivariant GNNs for Materials Modeling](https://openreview.net/forum?id=HRDRZNxQXc).

![pipeline](assets/pipeline.png)

To re-use components in this work, we recommend using one of the following 2 derived packages

* [**`phast`**](https://github.com/vict0rsch/phast), from [PhAST: Physics-Aware, Scalable, and Task-specific GNNs for Accelerated Catalyst Design](https://arxiv.org/abs/2211.12020)
  * `phast.PhysEmbedding` that allows one to create an embedding vector from atomic numbers that is the concatenation of:
    * A learned embedding for the atom's group and one for the atom's period.
    * A fixed or learned embedding from a set of known physical properties, as reported by `mendeleev`
    * For the OC20 dataset, a learned embedding for the atom's `tag` (adsorbate, catalyst surface or catalyst sub-surface)
  * Tag-based graph rewiring strategies for the OC20 dataset:
    * `remove_tag0_nodes` deletes all nodes in the graph associated with a tag 0 and recomputes edges
    * `one_supernode_per_graph` replaces all tag 0 atoms with a single new atom
    * `one_supernode_per_atom_type` replaces all tag 0 atoms of a given element with its own super node*
* [**`faenet`**](https://github.com/vict0rsch/faenet) to reuse direct components from the FAENet paper:
  * `faenet.FAENet` implements our efficient GNN model
  * `faenet.FrameAveraging` and `faenet.model_forward` implement (Stochastic) Frame Averaging data transforms and a utility function to average predictions over frames.

## Installation

```bash
# (1.a) ICML version
$ pip install --upgrade torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
$ pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
# (1.b) Or more recent
$ pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
$ pip install torch_geometric==2.3.0
$ pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+cu116.html
# (2.) Then
$ pip install ase dive-into-graphs e3nn h5py mendeleev minydra numba orion Cython pymatgen rdkit rich scikit-learn sympy tqdm wandb tensorboard lmdb pytorch_warmup ipdb orjson
$ git clone https://github.com/icanswim/cosmosis.git cosmosis # For the QM7X dataset
```

## To run the code

### TL;DR

1. Update the data paths in `configs/models/tasks/${task}.yaml`
2. Check out flags in `ocpmodels/common/flags.py`, especially those related to Weights and Biases
3. Run `python main.py --config=${model}-${task}-${split}`
4. Have a look at the example `scripts/submit.sh` to run multi-GPU SLURM jobs

### Configuration

* Specify the base configuration to use from the command-line with **`--config=${model}-${task}-${split}`**
  * `${model}` must listed in `ocpmodels/models/*.py` and the name to use is specified by the `@registry.register_model(${model})`
  * `${task}` can be one of `{is2re, s2ef, qm7x, qm9}`
  * `${split}` is either a pre-defined split (in the case of OC20) or `all` for the `qm*` tasks
  * Examples
    * `--config=faenet-is2re-all`, `--config=faenet-s2ef-all`, `--config=schnet-qm7x-all` etc.

* The code will load hyperparameters from `configs/models`, by subsequently merging (deep merge) resulting dictionaries:

  1. An initial `dict` is created from the *default* flag values in `ocpmodels/common/flags.py`
  2. `tasks/${task}.yaml -> default:`
  3. `tasks/${task}.yaml -> ${split}:`
  4. `${model}.yaml -> default:`
  5. `${model}.yaml -> ${task}:default:`
  6. `${model}.yaml -> ${task}:${split}:`
  7. Lastly, any command-line arg will override the configuration.

* The default parameters for a given `${model}-${task}-${split}` reflect the results in the papers.
* **Override any hyperparameter from the command-line** (including nested keys) `--nested.key=value`
* The main namespaces for hyperparameters are:
  * `--model.*` to define the model specific HPs (`num_gaussians`, `num_interactions` etc.)
  * `--optim.*` to define the optimization's HPs (`batch_size`, `lr_initial` `max_epochs` etc.)
  * `--dataset.*` to define data attributes (`default_val`, `${split}.src` etc.)
