#!/bin/bash
# Setup virtual environment
module purge
module load cuda/10.2
module load python/3.8
python -m virtualenv ~/.virtualenvs/ocp-torch1110cuda102
source ~/.virtualenvs/ocp-torch1110cuda102/bin/activate
# Update pip
python -m pip install --upgrade pip
# Install PyTorch family
python -m pip --no-cache-dir install torch==1.11.0
# Check PyTorch installation
python -c "import torch; print(torch.__version__, torch.version.cuda)"
# Install rest of PyTorch family
python -m pip --no-cache-dir install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu102.html
# Check installation
python -c "import torch; x=torch.eye(5).cuda(); x=x@x; print(x.sum()); import torch_sparse"
# Install pymatgen
python -m pip install Cython
python -m pip install pymatgen=="2020.4.2"
# Check pymatgen installation
python -c "import pymatgen"
# Other packages
python -m pip install git+https://github.com/SUNCAT-Center/CatKit.git ase==3.19.* lmdb six submitit wandb tensorboard numba tqdm wandb PyYAML ipdb jupyter flake8 black minydra
# ocpmodels: run from the repository root
python -m pip install .


