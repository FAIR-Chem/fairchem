# Open-Catalyst-Project Models

Implements the following baselines that take arbitrary chemical structures as
input to predict material properties:
- [Crystal Graph Convolutional Neural Networks (CGCNN)](https://link.aps.org/doi/10.1103/PhysRevLett.120.145301).
- Modified version of CGCNN as proposed in [Gu et al., 2020](https://pubs.acs.org/doi/abs/10.1021/acs.jpclett.0c00634).
- [Path-Augmented Graph Transformer Network](https://arxiv.org/abs/1905.12712).
Also related to [Graph Attention Networks](https://arxiv.org/abs/1710.10903) and
[Graph Transformer](https://openreview.net/forum?id=HJei-2RcK7).

##  Prerequisites

This package requires:

- [PyTorch](http://pytorch.org)
- [scikit-learn](http://scikit-learn.org/stable/)
- [pymatgen](http://pymatgen.org)

The easiest way of installing the prerequisites is via [conda](https://conda.io/docs/index.html).
After installing [conda](http://conda.pydata.org/), run the following command to
create a new [environment](https://conda.io/docs/user-guide/tasks/manage-environments.html)
named `ocp-models` and install all prerequisites:

```bash
conda upgrade conda
conda create -n ocp-models python=3.6

conda activate ocp-models
pip install -r requirements.txt
pre-commit install
```

This creates a conda environment and installs necessary python packages for
running various models. Activate the environment by:

```bash
conda activate ocp-models
```

Then you can test if all the prerequisites are installed properly by running:

```bash
python main.py -h
```

This should display the help messages for `main.py`. If you find no error messages, it means that the prerequisites are installed properly.

After you are done, exit the environment by:

```bash
conda deactivate
```

## Usage

### Download the datasets

For now, we are working with the following datasets:
- ulissigroup_co: dataset of DFT results for CO adsorption on various slabs (shared by [Jun](http://ulissigroup.cheme.cmu.edu/2017-11-28-junwoong-yoon/))
- [Materials Project](https://materialsproject.org): subset used in [Xie and Grossman](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301)
- [QM9](http://www.quantum-machine.org/datasets/)
- [ISO17](http://www.quantum-machine.org/datasets/)

To download the datasets:

```
cd data
./download_data.sh
```

#### ulissigroup_co

`docs_energy.pkl` has the dataset prepackaged for training. Make sure `configs/ulissigroup_co/base.yml` has the correct path to it.

`docs.pkl` is the original (raw) dataset from Jun. Each entry in `docs.pkl` has the following keys:

`['mongo_id', 'adsorbate', 'mpid', 'miller', 'shift', 'top', 'coordination', 'neighborcoord', 'energy', 'atoms', 'results', 'calc', 'initial_configuration’]`

`energy` is a target energy we want to fit, this is what we call binding energy.

`atoms` contains information of atoms mainly atom types and atom positions (x,y,z).

`results` shows the energy, E(slab+adsorbate) of the system and forces of each atom in the system in the FINAL DFT configuration.
The energy in `results` is different from `energy` above.
`energy` was calculated as E(slab+adsorbate) - E(slab) - E(adsorbate).
E(slab) and E(adsorbate) are calculated separately and not shown in our dataset, and they are constant.

`initial_configuration` contains `atoms` and `results` for INITIAL DFT configuration.

### Train a CGCNN model

To train a model on the CO adsorption data to predict energy (with default params):

```bash
python main.py --identifier my-first-experiment --config-yml configs/ulissigroup_co/cgcnn.yml
```

See `configs/ulissigroup_co/base.yml` and `configs/ulissigroup_co/cgcnn.yml` for dataset, model and optimizer parameters.

## Acknowledgements

- This codebase was initially forked from [CGCNN](https://github.com/txie-93/cgcnn)
by [Tian Xie](http://txie.me), but has undergone significant changes since.
- A lot of engineering ideas have been borrowed from [github.com/facebookresearch/pythia](https://github.com/facebookresearch/pythia).

If you use the CGCNN implementation for your research, consider citing:

```
@article{PhysRevLett.120.145301,
  title = {Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties},
  author = {Xie, Tian and Grossman, Jeffrey C.},
  journal = {Phys. Rev. Lett.},
  volume = {120},
  issue = {14},
  pages = {145301},
  numpages = {6},
  year = {2018},
  month = {Apr},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevLett.120.145301},
  url = {https://link.aps.org/doi/10.1103/PhysRevLett.120.145301}
}
```

## License

TBD
