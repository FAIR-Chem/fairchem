## Installation

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
