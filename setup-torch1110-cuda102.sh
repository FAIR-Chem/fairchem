#!/bin/bash
# Setup virtual environment
module purge
module load cuda/10.2
module load python/3.8
python -m virtualenv ~/.virtualenvs/ocp-torch1110cuda102
source ~/.virtualenvs/ocp-torch1110cuda102/bin/activate
# Update pip
python -m pip install --upgrade pip
# Install torch family
python -m pip install torch==1.11.0
python -m pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu102.html
# Check installation
python -c "import torch; x=torch.eye(5).cuda(); x=x@x; print(x.sum()); import torch_sparse"
# pymatgen
python -m pip install Cython
python -m pip install pymatgen
# Other
python -m pip install ase==3.21.* lmdb==1.1.1 six
# ocpmodels: run from the repository root
python -m pip install .


