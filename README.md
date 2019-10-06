# Crystal Graph Convolutional Neural Networks

Forked from [ulissigroup/cgnn](https://github.com/ulissigroup/cgcnn).

This software package implements the Crystal Graph Convolutional Neural Networks (CGCNN) that takes an arbitary crystal structure to predict material properties.

The following paper describes the details of the CGCNN framework:

[Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties](https://link.aps.org/doi/10.1103/PhysRevLett.120.145301)

##  Prerequisites

This package requires:

- [PyTorch](http://pytorch.org)
- [scikit-learn](http://scikit-learn.org/stable/)
- [pymatgen](http://pymatgen.org)

If you are new to Python, the easiest way of installing the prerequisites is via [conda](https://conda.io/docs/index.html). After installing [conda](http://conda.pydata.org/), run the following command to create a new [environment](https://conda.io/docs/user-guide/tasks/manage-environments.html) named `cgcnn` and install all prerequisites:

```bash
conda upgrade conda
conda create -n cgcnn python=3.6

conda activate cgcnn
pip install -r requirements.txt
```

*Note: since PyTorch introduced some breaking changes in v0.4.0, this code only works up to v0.3.1*

This creates a conda environment for running CGCNN. Before using CGCNN, activate the environment by:

```bash
conda activate cgcnn
```

Then, in directory `cgcnn`, you can test if all the prerequisites are installed properly by running:

```bash
python main.py -h
```

This should display the help messages for `main.py`. If you find no error messages, it means that the prerequisites are installed properly.

After you are done using CGCNN, exit the environment by:

```bash
conda deactivate
```

## Usage

### Download the dataset

#### 2019/10/05

For now, we are working with a dataset of DFT results for CO adsorption on various slabs (shared by [Jun](http://ulissigroup.cheme.cmu.edu/2017-11-28-junwoong-yoon/)).

To download the dataset:

```
cd data
./download_data.sh
```

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
python main.py --identifier my-first-experiment
```

See `configs/ulissigroup_co/base.yml` and `configs/ulissigroup_co/cgcnn.yml` for dataset, model and optimizer parameters.

## Acknowledgements

This codebase is based on [CGCNN](https://github.com/txie-93/cgcnn) by [Tian Xie](http://txie.me) who was advised by [Prof. Jeffrey Grossman](https://dmse.mit.edu/faculty/profile/grossman).

If you use this codebase for your research, consider citing:

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

CGCNN is released under the MIT License.
