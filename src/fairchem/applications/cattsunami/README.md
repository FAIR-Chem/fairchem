## CatTSunami: Accelerating Transition State Energy Calculations with Pre-trained Graph Neural Networks

![summary](https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/applications/cattsunami/summary_fig.png)

CatTSunami is a framework for high-throughput enumeration of nudged elastic band (NEB) frame sets. It was built for use with machine learned (ML) models trained on [OC20](https://arxiv.org/abs/2010.09990), which were demonstrated to be performant on this auxiliary task. To train your own model or obtain pre-trained checkpoints, please see [`fairchem-core`](https://github.com/FAIR-Chem/fairchem/tree/cattsunami-package/src/fairchem/core).

This repository contains the validation dataset, framework for enumeration, and accompanying code to run ML-accelerated NEBs and validate new models. For more information, please read the manuscript [paper](https://arxiv.org/abs/2405.02078).

### Getting started
Configured for use:
1. Install fairchem-core and fairchem-data-oc [instructions](https://fair-chem.github.io/core/install.html)
2. Pip innstall fairchem-applications-cattsunami 
3. Check out the [tutorial notebook](https://github.com/FAIR-Chem/fairchem/tree/main/src/fairchem/applications/cattsunami/tutorial/workbook.ipynb)
```
pip install fairchem-applications-cattsunami
```

Configured for local development:
1. Clone the [fairchem repo](https://github.com/FAIR-Chem/fairchem/tree/main) 
2. Install `fairchem-data-oc` and `fairchem-core`:  [instructions](https://fair-chem.github.io/core/install.html)
3. Install this repository `pip install -e packages/fairchem-applications-cattsunami`
4. Check out the [tutorial notebook](https://github.com/FAIR-Chem/fairchem/tree/main/src/fairchem/applications/cattsunami/tutorial/workbook.ipynb)


### Validation Dataset
The validation dataset is comprised of 932 DFT NEB calculations to assess model performance on this important task. There are 3 different reaction classes considered: desorptions, dissociations, and transfers. For more information see the [dataset markdown file](https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/applications/cattsunami/DATASET.md).

|Splits |Size of compressed version (in bytes)  |Size of uncompressed version (in bytes)    | MD5 checksum (download link)   |
|---    |---    |---    |---    |
|ASE Trajectories   |1.5G  |6.3G   | [52af34a93758c82fae951e52af445089](https://dl.fbaipublicfiles.com/opencatalystproject/data/oc20neb/oc20neb_dft_trajectories_04_23_24.tar.gz)   |



## Citing this work

If you use this codebase in your work, please consider citing:

```bibtex
@article{wander2024cattsunami,
  title={CatTSunami: Accelerating Transition State Energy Calculations with Pre-trained Graph Neural Networks},
  author={Wander, Brook and Shuaibi, Muhammed and Kitchin, John R and Ulissi, Zachary W and Zitnick, C Lawrence},
  journal={arXiv preprint arXiv:2405.02078},
  year={2024}
}
```
