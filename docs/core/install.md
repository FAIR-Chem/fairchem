# Installation

## conda or better yet [mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html) - easy

We do not have official conda recipes (yet!), so install with conda or mamba you will need to clone the
[ocp repo](https://github.com/Open-Catalyst-Project/ocp) and run the following from inside the repo directory to
create an environment with all dependencies.

1. Create an *ocp-models* environment
   1. **GPU**

      The default environment uses cuda 11.8, if you need a different version you will have to edit *pytorch-cuda* version
      accordingly.
      ```bash
      conda create -f env.gpu.yml
      ```

   2. **CPU**
      ```bash
      conda create -f env.cpu.yml
      ```

2. Activate the environment and install `ocpmodels`
   ```bash
   conda activate ocp-models
   pip install .
   ```

## PyPi - flexible
1. Install `pytorch` by selecting your installer, OS and CPU or CUDA version follow the official
[Pytorch docs](https://pytorch.org/get-started/locally/)

2. Install `torch_geometric` and the `torch_scatter`, `torch_sparse`, and `torch_cluster` optional dependencies
   similarly by selecting the appropriate versions in the official
   [PyG docs](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

3. Install `ocpmodels`
   1. From test-PyPi (until we have our official release on PyPi soon!)
      ```bash
      pip install -i https://test.pypi.org/simple/ ocp-models
      ```
   2. Or by cloning the repo and then using pip
      ```bash
      pip install .
      ```


## Dev install

If you plan to make contributions you will need to clone the repo and install `ocp-models` with dev dependencies,
```bash
pip install -e .[dev]
```
