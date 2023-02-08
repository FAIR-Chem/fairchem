# Polarizable Atom Interaction Neural Network (PaiNN)

Kristof T. Schütt, Oliver T. Unke, Michael Gastegger

[[`arXiv:2102.03150`](https://arxiv.org/abs/2102.03150)]

This is our independent reimplementation of the original PaiNN architecture
with the difference that forces are predicted directly from vectorial features
via a gated equivariant block instead of gradients of the energy output.
This breaks energy conservation but is essential for good performance on OC20.

All PaiNN models were trained without AMP, as using AMP led to unstable training.

## IS2RE

Trained only using IS2RE data, no auxiliary losses and/or S2EF data.

| Model | Val ID Energy MAE | Test metrics | Download |
| ----- | ----------------- | ------------ | -------- |
| painn_h1024_bs4x8 | 0.5728 | [IS2RE](https://evalai.s3.amazonaws.com/media/submission_files/submission_200972/45d289fc-8de9-45cc-aed4-6cd1753cb56d.json) | [config](https://github.com/Open-Catalyst-Project/ocp/tree/main/configs/is2re/all/painn/painn_h1024_bs8x4.yml) \| [checkpoint](https://dl.fbaipublicfiles.com/opencatalystproject/models/2022_05/is2re/painn_h1024_bs4x8_is2re_all.pt) |

## S2EF

| Model | Val ID 30k Force MAE | Val ID 30k Energy MAE | Val ID 30k Force cos | Test metrics | Download |
| ----- | -------------------- | --------------------- | -------------------- | ------------ | -------- |
| painn_h512 | 0.02945 | 0.2459 | 0.5143 | [S2EF](https://evalai.s3.amazonaws.com/media/submission_files/submission_200711/2f487981-051d-445e-a7cd-6eb00ebe0735.json) \| [IS2RE](https://evalai.s3.amazonaws.com/media/submission_files/submission_200710/7fe29c4c-c203-434d-a6d4-9ea992d3bb5c.json) \| [IS2RS](https://evalai.s3.amazonaws.com/media/submission_files/submission_200700/8fd419e6-bab3-49be-a936-ae31979b4866.json) | [config](https://github.com/Open-Catalyst-Project/ocp/tree/main/configs/s2ef/all/painn/painn_h512.yml) \| [checkpoint](https://dl.fbaipublicfiles.com/opencatalystproject/models/2022_05/s2ef/painn_h512_s2ef_all.pt) |

## Citing

If you use PaiNN in your work, please consider citing the original paper:

```bibtex
@inproceedings{schutt_painn_2021,
  title = {Equivariant message passing for the prediction of tensorial properties and molecular spectra},
  author = {Sch{\"u}tt, Kristof and Unke, Oliver and Gastegger, Michael},
  booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
  year = {2021},
}
```
