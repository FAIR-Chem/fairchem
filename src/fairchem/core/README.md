<h1 align="center"> <code>fairchem</code> by FAIR Chemistry </h1>

<p align="center">
  <img width="559" height="200" src="https://github.com/FAIR-Chem/fairchem/assets/45150244/5872c21c-8f39-41af-b703-af9817f0affe"?
</p>


<h4 align="center">

![tests](https://github.com/FAIR-Chem/fairchem/actions/workflows/test.yml/badge.svg?branch=main)
[![documentation](https://github.com/FAIR-Chem/fairchem/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/FAIR-Chem/fairchem/actions/workflows/docs.yml)
[![Static Badge](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)

</h4>

`fairchem` is the [FAIR](https://ai.meta.com/research/) Chemistry's centralized repository of all its data, models, demos, and application efforts for materials science and quantum chemistry.

`fairchem` provides training and evaluation code for tasks and models that take arbitrary
chemical structures as input to predict energies / forces / positions / stresses,
and can be used as a base scaffold for research projects. For an overview of
tasks, data, and metrics, please read the documentations and respective papers:
 - [OC20](https://fair-chem.github.io/core/datasets/oc20.html)
 - [OC22](https://fair-chem.github.io/core/datasets/oc22.html)
 - [ODAC23](https://fair-chem.github.io/core/datasets/odac.html)

## Acknowledgements

- This codebase was initially forked from [CGCNN](https://github.com/txie-93/cgcnn)
by [Tian Xie](http://txie.me), but has undergone significant changes since.
- A lot of engineering ideas have been borrowed from [github.com/facebookresearch/mmf](https://github.com/facebookresearch/mmf).
- The DimeNet++ implementation is based on the [author's Tensorflow implementation](https://github.com/klicperajo/dimenet) and the [DimeNet implementation in Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/models/dimenet.py).

## License

`fairchem` is released under the [MIT](https://github.com/FAIR-Chem/fairchem/blob/main/LICENSE.md) license.

## Citing `fairchem`

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
