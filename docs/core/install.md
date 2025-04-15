# Installation & License

To install `fairchem-core` you will need to setup the `fairchem-core` environment (using [conda](#Conda) or [pip](#PyPi))
and then either install `fairchem-core` package [directly](#Install-fairchem-core) or install a [development version](#Development-install) from our git repository.

## Environment

You can install the environment using either conda or pip


### PyPi
1. Make a clean conda environment, or virtualenv, or 
2. You can install `pytorch` and `torch_geometric` dependencies from PyPI. A one-line install that works well and will install everything needed to build the docs and show the examples in the documentation is:
```
pip install --upgrade --force-reinstall torch==2.4.1 torchvision torchaudio -f https://download.pytorch.org/whl/cu121 \
            pyg_lib torch_spline_conv torch_geometric torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.4.1+cu121.html \ 
            fairchem-core[docs,adsorbml,quacc] fairchem-demo-ocpapi fairchem-data-oc
```
If you're operating in an environment alongside other packages, then you should probably run the command above with `--dry-run` first to make sure what it's about to do makes sense!

If you need other versions of python/cuda/pytorch/etc (untested, be careful!):
1. Install `pytorch` by selecting your installer, OS and CPU or CUDA version following the official
[Pytorch docs](https://pytorch.org/get-started/locally/)
2. Install `torch_geometric` and the `torch_scatter`, `torch_sparse`, and `torch_cluster` optional dependencies
   similarly by selecting the appropriate versions in the official
   [PyG docs](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
3. pip install the fairchem packages with the selections you need
```bash
pip install fairchem-core
```


### Conda

We do not have official conda recipes (yet!); in the meantime you can use the
following environment yaml files to setup on CPU or GPU. If conda is too slow for you, please consider using [mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html)

1. Create an environment to install *fairchem*

   a. **GPU**

      The default environment uses cuda 11.8, if you need a different version you will have to edit *pytorch-cuda* version
      accordingly.
      ```bash
      wget https://raw.githubusercontent.com/FAIR-Chem/fairchem/main/packages/env.gpu.yml
      conda env create -f env.gpu.yml
      ```

   b. **CPU**
      ```bash
      wget https://raw.githubusercontent.com/FAIR-Chem/fairchem/main/packages/env.cpu.yml
      conda env create -f env.cpu.yml
      ```

2. Activate the environment
   ```bash
   conda activate fair-chem
   ```


## Standard installation of fairchem-core
Install `fairchem-core` from PyPi


### Additional packages
`fairchem` is a namespace package, meaning all packages are installed seperately. If you need
to install other packages you can do so by:
```bash
pip install fairchem-{package-to-install}
```
Available `fairchem` packages are `fairchem-core`,`fairchem-data-oc`,`fairchem-demo-ocpapi`,`fairchem-applications-cattsunami`

## Development installation
If you plan to make contributions you will need to fork and clone (for windows user please see next section) the repo,
set up the environment, and install fairchem-core from source in editable mode with dev dependencies,
```bash
git clone https://github.com/FAIR-Chem/fairchem.git
cd fairchem
pip install -e packages/fairchem-core[dev]
pytest tests/core
```

And similarly for any other namespace package:
```bash
pip install -e packages/fairchem-{package-to-install}
```

### Cloning and installing the git repository on windows

Our build system requires the use of symlinks which are not available by default on windows. To properly build fairchem packages you must enable symlinks and clone the repository with them enabled.

1) When installing git on your machine make sure "enable symbolic links" is checked  ([download git installer](https://git-scm.com/download/win)) ([see here](https://stackoverflow.com/a/65563980) for detailed instructions )

2) Enable developer mode ([instructions](https://learn.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development)) or run the following commands as administrator

3) Run the git clone command with symlinks enabled
```
git clone -c core.symlinks=true https://github.com/FAIR-Chem/fairchem.git
```

# Workflow Setup and Huggingface tokens

We're going to use [`quacc`](https://quantum-accelerators.github.io/quacc/index.html) along with FAIR chem calculators for some simple recipes to calculate elastic and phonon properties. `quacc` has the nice property that you can also use many different workflow managers like jobflow, dask, or prefect to scale and parallelize many calculations, including both ML and DFT calculations. 

## Quick setup (without a workflow backend)

1. First, make sure you installed `fairchem[quacc]` to pick up an appropriate version of quacc and the phonon dependencies. 
2. We're going to use OMat24 machine learning interatomic potentials (MLIPs), which require approval and a license agreement. 
    1. Navigate to the [OMat24 model repo](https://huggingface.co/fairchem/OMAT24), login with your HuggingFace account, and request access to the gated models if you haven't already
    2. Make sure `huggingface_hub` has access to the repo by running `huggingface-cli login` and following the instructions:
        a. Navigate to [HuggingFace tokens](https://huggingface.co/settings/tokens)
        b. Click "+Create new token"
        c. type OMat24 in the Repositories permissions field
        d. Click create token
        e. Type the token into the CLI, or alternatively you can run `huggingface-cli login --token YOUR_TOKEN_HERE`. You can also set the environment variable `HF_TOKEN=YOUR_TOKEN_HERE` if that's more convenient.
3. Set up quacc to use your favorite workflow tool (dask, prefect, etc) by setting your [quacc configuration](https://quantum-accelerators.github.io/quacc/user/basics/wflow_overview.html#__tabbed_1_4). 
    a. Tip: If you just want to run like this tutorial is, write a simple file with `WORKFLOW_ENGINE: null` in `~/.quacc.yaml`. This is by far the easiest way to get started if you don't want to worry about parallelization/etc


# License

## Repository software

The software in this repo is licensed under an MIT license unless otherwise specified. 

```
MIT License

Copyright (c) Meta, Inc. and its affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Model checkpoints

The Open Catalyst Project and OpenDAC checkpoints are licensed under a CC-by license. The OMat24-trained checkpoints are available for use under a custom license available at https://huggingface.co/fairchem/OMAT24
