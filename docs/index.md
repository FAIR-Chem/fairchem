
<h2 align="center"> <code>fairchem</code> by FAIR Chemistry </h2>

<p align="center">
  <img width="559" height="200" src="https://github.com/FAIR-Chem/fairchem/assets/45150244/5872c21c-8f39-41af-b703-af9817f0affe"?
</p>

<h4 align="center">

![tests](https://github.com/FAIR-Chem/fairchem/actions/workflows/test.yml/badge.svg?branch=main)
![documentation](https://github.com/FAIR-Chem/fairchem/actions/workflows/deploy_docs.yml/badge.svg?branch=main)
![Static Badge](https://img.shields.io/badge/python-3.9%2B-blue)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new/FAIR-Chem/fairchem?quickstart=1)

</h4>

#### FAIR-Chem overview

`fairchem` is the [FAIR](https://ai.meta.com/research/) Chemistry's centralized repository of all its data, models, demos, and application efforts
for materials science and quantum chemistry. Collaborative projects that contribute or use the models and approaches in
this repo:
* [Open Catalyst Project (OCP)](https://opencatalystproject.org/)
* [Open Direct Air Capture (OpenDAC)](https://open-dac.github.io/)

```{note}
We re-organized and rebranded the repository in 2024 (previously the `fairchem` repo) to reflect the increasingly
general usability of these models beyond catalysis, including things like direct air capture.
```

#### Datasets in `fairchem`:
`fairchem` provides training and evaluation code for tasks and models that take arbitrary
chemical structures as input to predict energies / forces / positions / stresses,
and can be used as a base scaffold for research projects. For an overview of
tasks, data, and metrics, please read the documentations and respective papers:
 - [OC20](core/datasets/oc20)
 - [OC22](core/datasets/oc22)
 - [ODAC23](core/datasets/odac)
 - [OC20Dense](core/datasets/oc20dense)
 - [OC20NEB](core/datasets/oc20neb)
 - [OMat24](core/datasets/omat24)

#### Projects and models built on `fairchem`:

- SchNet [[`arXiv`](https://arxiv.org/abs/1706.08566)] [[`code`](https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/models/schnet.py)]
- DimeNet++ [[`arXiv`](https://arxiv.org/abs/2011.14115)] [[`code`](https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/models/dimenet_plus_plus.py)]
- GemNet-dT [[`arXiv`](https://arxiv.org/abs/2106.08903)] [[`code`](https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/models/gemnet)]
- PaiNN [[`arXiv`](https://arxiv.org/abs/2102.03150)] [[`code`](https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/models/painn)]
- Graph Parallelism [[`arXiv`](https://arxiv.org/abs/2203.09697)] [[`code`](https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/models/gemnet_gp)]
- GemNet-OC [[`arXiv`](https://arxiv.org/abs/2204.02782)] [[`code`](https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/models/gemnet_oc)]
- SCN [[`arXiv`](https://arxiv.org/abs/2206.14331)] [[`code`](https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/models/scn)]
- AdsorbML [[`arXiv`](https://arxiv.org/abs/2211.16486)] [[`code`](https://github.com/FAIR-Chem/fairchem/tree/main/src/fairchem/applications/AdsorbML)]
- eSCN [[`arXiv`](https://arxiv.org/abs/2302.03655)] [[`code`](https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/models/escn)]
- EquiformerV2 [[`arXiv`](https://arxiv.org/abs/2306.12059)] [[`code`](https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/models/equiformer_v2)]

Older model implementations that are no longer supported:

- CGCNN [[`arXiv`](https://arxiv.org/abs/1710.10324)] [[`code`](https://github.com/FAIR-Chem/fairchem/blob/e7a8745eb307e8a681a1aa9d30c36e8c41e9457e/ocpmodels/models/cgcnn.py)]
- DimeNet [[`arXiv`](https://arxiv.org/abs/2003.03123)] [[`code`](https://github.com/FAIR-Chem/fairchem/blob/e7a8745eb307e8a681a1aa9d30c36e8c41e9457e/ocpmodels/models/dimenet.py)]
- SpinConv [[`arXiv`](https://arxiv.org/abs/2106.09575)] [[`code`](https://github.com/FAIR-Chem/fairchem/blob/e7a8745eb307e8a681a1aa9d30c36e8c41e9457e/ocpmodels/models/spinconv.py)]
- ForceNet [[`arXiv`](https://arxiv.org/abs/2103.01436)] [[`code`](https://github.com/FAIR-Chem/fairchem/blob/e7a8745eb307e8a681a1aa9d30c36e8c41e9457e/ocpmodels/models/forcenet.py)]

### Discussion

For all non-codebase related questions and to keep up-to-date with the latest OCP
announcements, please join the [discussion board](https://discuss.opencatalystproject.org/).

All code-related questions and issues should be posted directly on our
[issues page](https://github.com/FAIR-Chem/fairchem/issues).

### Acknowledgements

- This codebase was initially forked from [CGCNN](https://github.com/txie-93/cgcnn)
by [Tian Xie](http://txie.me), but has undergone significant changes since.
- A lot of engineering ideas have been borrowed from [github.com/facebookresearch/mmf](https://github.com/facebookresearch/mmf).
- The DimeNet++ implementation is based on the [author's Tensorflow implementation](https://github.com/klicperajo/dimenet) and the [DimeNet implementation in Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/models/dimenet.py).
- It was then developed as the OCP repo, and includes many contributions from the community and collaborators.
- Much of the documentation was developed for various papers or as part of a comprehensive tutorial for the 2023 ACS Fall Chemistry conference.

### License

`fairchem` is released under the [MIT](https://github.com/FAIR-Chem/fairchem/blob/main/LICENSE.md) license.

### Citing `fairchem`

If you use this codebase in your work, please consider citing:

```bibtex
@article{ocp_dataset,
    author = {Chanussot*, Lowik and Das*, Abhishek and Goyal*, Siddharth and Lavril*, Thibaut and Shuaibi*, Muhammed and Riviere, Morgane and Tran, Kevin and Heras-Domingo, Javier and Ho, Caleb and Hu, Weihua and Palizhati, Aini and Sriram, Anuroop and Wood, Brandon and Yoon, Junwoong and Parikh, Devi and Zitnick, C. Lawrence and Ulissi, Zachary},
    title = {Open Catalyst 2020 (OC20) Dataset and Community Challenges},
    journal = {ACS Catalysis},
    year = {2021},
    doi = {10.1021/acscatal.0c04525},
}
```
