# Open-Catalyst-Project Models

Implements the following baselines that take arbitrary chemical structures as
input to predict material properties:
- [SchNet](https://arxiv.org/abs/1706.08566)
- [DimeNet](https://arxiv.org/abs/2003.03123)
- [Crystal Graph Convolutional Neural Networks (CGCNN)](https://link.aps.org/doi/10.1103/PhysRevLett.120.145301).

##  Installation

[last updated October 10, 2020]

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

Instructions are for PyTorch 1.6, CUDA 10.1 specifically.

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

Dataset download links can be found at [opencatalstproject.org](https://opencatalystproject.org) for the S2EF, IS2RS, and IS2RE tasks. IS2* datasets are stored as LMDB files and are ready to be used upon download. S2EF datasets require an additional preprocessing step.

### Preprocess datasets - S2EF only

1. Untar the downloaded dataset: `tar -xzvf sample_xyz_compressed.tar`
2. Uncompress the untarred directory contents: `baselines/scripts/uncompress.py --ipdir /path/to/sample_xyz_compressed --opdir raw_data/`
3. Run the LMDB preprocessing script: `scripts/preprocess_ef.py --data-path raw_data/ --out-path processed_lmdb/ --num-workers 32 --get-edges --ref-energy`; where
    - `--get-edges`: includes edge information in LMDBs (~10x storage requirement, ~3-5x slowdown), otherwise, compute edges on the fly (larger GPU memory requirement).
    - `--ref-energy`: uses referenced energies instead of raw energies.

### Train models for the the desired tasks

A detailed description of how to train, predict, and run ML-based relaxations can be found [here](https://github.com/Open-Catalyst-Project/baselines/blob/release/docs/source/tutorials/training.rst).

## Acknowledgements

- This codebase was initially forked from [CGCNN](https://github.com/txie-93/cgcnn)
by [Tian Xie](http://txie.me), but has undergone significant changes since.
- A lot of engineering ideas have been borrowed from [github.com/facebookresearch/mmf](https://github.com/facebookresearch/mmf).

## License

This code is MIT licensed, as found in the [LICENSE file](https://github.com/Open-Catalyst-Project/baselines/blob/master/LICENSE.md).
