# Pre-trained CGCNN models

This directory includes several pre-trained CGCNN models that one can use to predict the material properties of new crystals. We encourage users to report their own CGCNN models for other properties which might benefit the community.

## Pre-trained models

### Regression

| File                        | Property         | Units    | Data Ref.                                                    | Model Ref.                                                   |
| --------------------------- | ---------------- | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `formation-energy-per-atom` | Formation Energy | eV/atom  | [Jain et al.](https://aip.scitation.org/doi/10.1063/1.4812323) | [Xie et al.](https://link.aps.org/doi/10.1103/PhysRevLett.120.145301) |
| ` final-energy-per-atom`    | Absolute Energy  | eV/atom  | [Jain et al.](https://aip.scitation.org/doi/10.1063/1.4812323) | [Xie et al.](https://link.aps.org/doi/10.1103/PhysRevLett.120.145301) |
| `band-gap`                  | Band Gap         | eV       | [Jain et al.](https://aip.scitation.org/doi/10.1063/1.4812323) | [Xie et al.](https://link.aps.org/doi/10.1103/PhysRevLett.120.145301) |
| `efermi`                    | Fermi Energy     | eV/atom  | [Jain et al.](https://aip.scitation.org/doi/10.1063/1.4812323) | [Xie et al.](https://link.aps.org/doi/10.1103/PhysRevLett.120.145301) |
| `bulk-moduli`               | Bulk Moduli      | log(GPa) | [Jain et al.](https://aip.scitation.org/doi/10.1063/1.4812323) | [Xie et al.](https://link.aps.org/doi/10.1103/PhysRevLett.120.145301) |
| `shear-moduli`              | Shear Moduli     | log(GPa) | [Jain et al.](https://aip.scitation.org/doi/10.1063/1.4812323) | [Xie et al.](https://link.aps.org/doi/10.1103/PhysRevLett.120.145301) |
| `poisson-ratio`             | Poisson Ratio    | —        | [Jain et al.](https://aip.scitation.org/doi/10.1063/1.4812323) | [Xie et al.](https://link.aps.org/doi/10.1103/PhysRevLett.120.145301) |

### Classification

| File                        | Positive | Negative      | Data Ref.                                                    | Model Ref.                                                   |
| --------------------------- | -------- | ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `semi-metal-classification` | Metal    | Semiconductor | [Jain et al.](https://aip.scitation.org/doi/10.1063/1.4812323) | [Xie et al.](https://link.aps.org/doi/10.1103/PhysRevLett.120.145301) |

## Before using pre-trained models

- CGCNN models (and machine learning models in general) can only generalize to crystals from the same ditribution as training data. It is adviced to check **Data Ref.** before using pre-trained models. For instance, Materials Project uses [ICSD](https://icsd.fiz-karlsruhe.de/search/index.xhtml;jsessionid=E3291AF7E25ED34B31B9AD5A9CBF80A1) structures as input, which includes experimentally synthesized crystal structures. Significant errors can be expected if a CGCNN model trained on Materials Project is used to predict the properties of imaginary, thermadynamically unstable crystals.
- CGCNN models have prediction errors.  It is adviced to check **Model Ref.** to understand their accuracy before using pre-trained models.

## How to cite

If you used any pre-trained models, please cite both **Data Ref.** and **Model Ref.** because data and model are equally important for a successful machine learning model! 

## How to share your pre-trained models

Please send an email to txie@mit.edu if you want to share your own pre-trained models. Since we don't have time to check the validity of your model, we only accept **peer reviewed** works.

To submit, be sure to include:

1. A `pth.tar` file storing the CGCNN model.
2. The type of the model and the target property.
3. The links to the data reference and model reference.

