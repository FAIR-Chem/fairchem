# Installation

## pip (fast, easy to get started)

Installing the OCP package and necessary dependencies is now as easy as:

### GPU enabled machines
```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install pyg_lib torch_scatter torch_sparse --no-index -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install -i https://test.pypi.org/simple/ ocp-models
```

Note the `--no-index` in the second line - it seems unnecessary, but is important to make sure the versions come from that file and not the main pip index!

### CPU-only install (slower training/inference!)
```
pip install torch==1.13.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu # install CPU torch
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-1.13.1+cpu.html
pip install -i https://test.pypi.org/simple/ ocp-models
```

## Conda (preferred for model training & development)

- We'll use `conda` to install dependencies and set up the environment.
We recommend using the [Python 3.9 Miniconda installer](https://docs.conda.io/en/latest/miniconda.html#linux-installers).
- After installing `conda`, install [`mamba`](https://mamba.readthedocs.io/en/latest/) to the base environment. `mamba` is a faster, drop-in replacement for `conda`:
    ```bash
    conda install mamba -n base -c conda-forge
    ```
- Also install `conda-merge` to the base environment:
    ```bash
    conda install conda-merge -n base -c conda-forge
    ```

Next, follow the instructions for [GPU](#gpu-machines) or [CPU](#cpu-only-machines) machines depending on your hardware to create a new environment named `ocp-models` and install dependencies.

### GPU machines

Instructions are for PyTorch 1.13.1, CUDA 11.6 specifically.

- First, check that CUDA is in your `PATH` and `LD_LIBRARY_PATH`, e.g.
    ```bash
    $ echo $PATH | tr ':' '\n' | grep cuda
    /public/apps/cuda/11.6/bin

    $ echo $LD_LIBRARY_PATH | tr ':' '\n' | grep cuda
    /public/apps/cuda/11.6/lib64
    ```
    The exact paths may differ on your system.
- Then install the dependencies:
    ```bash
    conda-merge env.common.yml env.gpu.yml > env.yml
    mamba env create -f env.yml
    ```
    Activate the conda environment with `conda activate ocp-models`.
- Install the `ocp` package with `pip install -e .`.
- Finally, install the pre-commit hooks:
    ```bash
    pre-commit install
    ```

### CPU-only machines

Please skip the following if you completed the with-GPU installation from above.

```bash
conda-merge env.common.yml env.cpu.yml > env.yml
mamba env create -f env.yml
conda activate ocp-models
pip install -e .
pre-commit install
```
