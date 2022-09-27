## Installation

The easiest way to install prerequisites is via [conda](https://conda.io/docs/index.html).

After installing [conda](http://conda.pydata.org/), run the following commands
to create a new [environment](https://conda.io/docs/user-guide/tasks/manage-environments.html)
named `ocp-models` and install dependencies.

### Pre-install step

Install `conda-merge`:
```bash
pip install conda-merge
```
If you're using system `pip`, then you may want to add the `--user` flag to avoid using `sudo`.
Check that you can invoke `conda-merge` by running `conda-merge -h`.

### GPU machines

Instructions are for PyTorch 1.9.0, CUDA 10.2 specifically.

First, check that CUDA is in your `PATH` and `LD_LIBRARY_PATH`, e.g.
```bash
$ echo $PATH | tr ':' '\n' | grep cuda
/public/apps/cuda/10.2/bin

$ echo $LD_LIBRARY_PATH | tr ':' '\n' | grep cuda
/public/apps/cuda/10.2/lib64
```

The exact paths may differ on your system.

Then install the dependencies:
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

#### Ampere GPUs

NVIDIA Ampere cards require a CUDA version >= 11.1 to function properly, modify the lines [here](https://github.com/Open-Catalyst-Project/ocp/blob/master/env.gpu.yml#L6-L8) to
```
- cudatoolkit=11.1
- -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
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

### Mac CPU-only machines

Only run the following if installing on a CPU only machine running Mac OS X.

```
conda env create -f env.common.yml
conda activate ocp-models
MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ pip install torch-cluster torch-scatter torch-sparse torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.9.0+cpu.html
pip install -e .
pre-commit install
```
