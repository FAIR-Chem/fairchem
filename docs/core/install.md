# Installation

## conda or better yet [mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html) - easy

We do not have official conda recipes (yet!), so to install with conda or mamba you will need to clone the
[fairchem](https://github.com/FAIR-Chem/fairchem) and run the following from inside the repo directory to create an environment with all the
necessary dependencies.

1. Create a *fairchem* environment
   1. **GPU**

      The default environment uses cuda 11.8, if you need a different version you will have to edit *pytorch-cuda* version
      accordingly.
      ```bash
      conda env create -f packages/env.gpu.yml
      ```

   2. **CPU**
      ```bash
      conda env create -f packages/env.cpu.yml
      ```

2. Activate the environment and install `fairchem-core`
   ```bash
   conda activate fair-chem
   pip install packages/fairchem-core
   ```

## PyPi - flexible
1. Install `pytorch` by selecting your installer, OS and CPU or CUDA version following the official
[Pytorch docs](https://pytorch.org/get-started/locally/)

2. Install `torch_geometric` and the `torch_scatter`, `torch_sparse`, and `torch_cluster` optional dependencies
   similarly by selecting the appropriate versions in the official
   [PyG docs](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

3. Install `fairchem-core`
   1. From test-PyPi (until we have our official release on PyPi soon!)
      ```bash
      pip install -i https://test.pypi.org/simple/fairchem-core
      ```
   2. Or by cloning the repo and then using pip
      ```bash
      pip install packages/fairchem-core
      ```

## Additional packages

`fairchem` is a namespace package, meaning all packages are installed seperately. If you need
to install other packages you can do so by:
```bash
pip install -e pip install packages/fairchem-{package-to-install}
```

## Dev install

If you plan to make contributions you will need to clone the repo and install `fairchem-core` in editable mode with dev
dependencies,
```bash
pip install -e pip install packages/fairchem-core[dev]
```
