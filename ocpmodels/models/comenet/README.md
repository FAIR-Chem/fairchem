# ComENet: Towards Complete and Efficient Message Passing for 3D Molecular Graphs

Limei Wang, Yi Liu, Yuchao Lin, Haoran Liu, Shuiwang Ji

[`arXiv:2206.08515`](https://arxiv.org/abs/2206.08515)

ComENet OC20 IS2RE Performance ("All" dataset, direct energy predictions, no pretraining or auxiliary task)
![OC20 IS2RE Performance](ComENetData.jpg)
ComENet can be trained in under 20 minutes per epoch on a single NVIDIA 2080ti GPU and predictions
take less than one minute per test/validation split.

## Citing

If you use ComENet in your work, please consider citing:

```bibtex
@article{ComENet,
  title = {{ComENet: Towards Complete and Efficient Message Passing for 3D Molecular Graphs}},
  author = {Wang, Limei and Liu, Yi and Lin, Yuchao and Liu, Haoran and Ji, Shuiwang},
  journal = {Neural Information Processing Systems (NeurIPS)},
  year = {2022},
}
```