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

Instructions are for PyTorch 1.13.1, CUDA 11.6 specifically.

First, check that CUDA is in your `PATH` and `LD_LIBRARY_PATH`, e.g.
```bash
$ echo $PATH | tr ':' '\n' | grep cuda
/public/apps/cuda/11.6/bin

$ echo $LD_LIBRARY_PATH | tr ':' '\n' | grep cuda
/public/apps/cuda/11.6/lib64
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

### CPU-only machines

Please skip the following if you completed the with-GPU installation from above.

```bash
conda-merge env.common.yml env.cpu.yml > env.yml
conda env create -f env.yml
conda activate ocp-models
pip install -e .
pre-commit install
```
