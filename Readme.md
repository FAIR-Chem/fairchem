# FAENet

## Installation

### To run the code

```bash
# (1.a) ICML version
$ pip install --upgrade torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
$ pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
# (1.b) Or more recent
$ pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
$ pip install torch_geometric==2.3.0
$ pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+cu116.html
# (2.) Then
$ pip install ase dive-into-graphs e3nn h5py mendeleev minydra numba orion Cython pymatgen rdkit rich scikit-learn sympy tqdm wandb tensorboard lmdb pytorch_warmup ipdb orjson
$ git clone https://github.com/icanswim/cosmosis.git cosmosis # For the QM7X dataset
```

### Individual components

#### FAENet model
#### Graph rewiring transforms (PhAST)

#### Frame Averaging transforms

