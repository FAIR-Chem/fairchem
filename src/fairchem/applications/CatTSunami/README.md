## CatTSunami: Accelerating Transition State Energy Calculations with Pre-trained Graph Neural Networks

![summary](https://github.com/Open-Catalyst-Project/CatTSunami/blob/master/summary_fig.png)

CatTSunami is a framework for high-throughput enumeration of nudged elastic band (NEB) frame sets. It was built for use with machine learned (ML) models trained on [OC20](https://arxiv.org/abs/2010.09990), which were demonstrated to be performant on this auxiliary task. To train your own model or obtain pre-trained checkpoints, please see [`ocp`](https://github.com/Open-Catalyst-Project/ocp).

This repository contains the validation dataset, framework for enumeration, and accompanying code to run ML-accelerated NEBs and validate new models. For more information, please read the manuscript [paper](https://arxiv.org/abs/2405.02078).

### Getting started
1. Install [`Open-Catalyst-Dataset`](https://github.com/Open-Catalyst-Project/Open-Catalyst-Dataset) and [`ocp`](https://github.com/Open-Catalyst-Project/ocp)
2. Clone this repository
3. `cd CatTSunami && python setup.py develop`
4. Check out the [tutorial notebook](https://github.com/Open-Catalyst-Project/CatTSunami/blob/master/tutorial/workbook.ipynb) 


### Validation Dataset
The validation dataset is comprised of 932 DFT NEB calculations to assess model performance on this important task. There are 3 different reaction classes considered: desorptions, dissociations, and transfers. For more information see the [dataset markdown file](https://github.com/Open-Catalyst-Project/CatTSunami/blob/master/DATASET.md).

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
