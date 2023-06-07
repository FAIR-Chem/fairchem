mamba create -n ocp20 python=3.10
mamba activate ocp20
mamba install -y -c conda-forge -c pytorch -c nvidia -c pyg numpy matplotlib seaborn sympy pandas numba scikit-learn ipykernel ipywidgets tqdm pyyaml networkx \
    pytorch torchtriton pytorch-cuda=11.8 \
    pyg pytorch-scatter pytorch-sparse pytorch-cluster \
    lightning torchmetrics einops wandb \
    ase python-lmdb h5py \
    cloudpickle \
    frozendict wrapt varname typing-extensions lovely-tensors lovely-numpy requests pytest nbval

pip install simplejson
