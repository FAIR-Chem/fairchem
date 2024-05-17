# Installation
 
To install `fairchem-core` you will need to setup the `fairchem-core` environment (using [conda](#Conda) or [pip](#PyPi)) and then either install `fairchem-core` package [directly](#Install-fairchem-core) or install [development version](#Development-install) from our git repository.

## Environment 

You can install the environment using either conda or pip

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

### PyPi
You can also install `pytorch` and `torch_geometric` dependencies from PyPI to select specific CPU or CUDA versions.

1. Install `pytorch` by selecting your installer, OS and CPU or CUDA version following the official
[Pytorch docs](https://pytorch.org/get-started/locally/)

2. Install `torch_geometric` and the `torch_scatter`, `torch_sparse`, and `torch_cluster` optional dependencies
   similarly by selecting the appropriate versions in the official
   [PyG docs](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

## Install fairchem-core
Install `fairchem-core` from PyPi
```bash
pip install fairchem-core
```

## Additional packages

`fairchem` is a namespace package, meaning all packages are installed seperately. If you need
to install other packages you can do so by:
```bash
pip install fairchem-{package-to-install}
```

## Development install

If you plan to make contributions you will need to fork and clone (for windows user please see next section) the repo, set up the environment, and install from source.
`fairchem-core` in editable mode with dev
dependencies,
```bash
git clone https://github.com/FAIR-Chem/fairchem.git
cd fairchem
pip install -e packages/fairchem-core[dev]
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
