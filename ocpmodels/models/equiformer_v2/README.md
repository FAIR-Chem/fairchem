# EquiformerV2: Improved Equivariant Transformer for Scaling to Higher-Degree Representations

Yi-Lun Liao, Brandon Wood, Abhishek Das*, Tess Smidt*

[[`arXiv:2306.12059`](https://arxiv.org/abs/2306.12059)]

## Running EquiformerV2

These instructions are for running EquiformerV2 from within the OCP codebase with minimal code duplication.
Please refer to the [official EquiformerV2 codebase](https://github.com/atomicarchitects/equiformer_v2)
for installation instructions.

* Clone the EquiformerV2 repo: `git clone git@github.com:atomicarchitects/equiformer_v2.git`
* Symlink the model to within the current repo: `ln -s path/to/equiformer_v2/nets ocpmodels/models/equiformer_v2/nets`
* Run training / evaluation as usual, e.g:
  ```bash
  python main.py \
    --config-yml configs/s2ef/2M/equiformer_v2/83M.yml \
    --checkpoint path/to/checkpoint.pt \
    --mode validate
  ```

## Checkpoints and config

We provide model weights for EquiformerV2 trained on S2EF-2M dataset for 30 epochs,
EquiformerV2 (31M) trained on S2EF-All+MD, and EquiformerV2 (153M) trained on S2EF-All+MD.

|Model	|Training Split	|Download	|val force MAE (meV / Å) |val energy MAE (meV) |
|---	|---	|---	|---	|---	|
|EquiformerV2	|2M	|[checkpoint](https://dl.fbaipublicfiles.com/opencatalystproject/models/2023_06/oc20/s2ef/eq2_83M_2M.pt) \| [config](configs/s2ef/2M/equiformer_v2/83M.yml)	|19.4 | 278 |
|EquiformerV2 (31M)|All+MD |[checkpoint](https://dl.fbaipublicfiles.com/opencatalystproject/models/2023_06/oc20/s2ef/eq2_31M_ec4_allmd.pt) \| [config](configs/s2ef/all/equiformer_v2/31M.yml) |16.3 | 232 |
|EquiformerV2 (153M) |All+MD | [checkpoint](https://dl.fbaipublicfiles.com/opencatalystproject/models/2023_06/oc20/s2ef/eq2_153M_ec4_allmd.pt) \| [config](configs/s2ef/all/equiformer_v2/153M.yml) |15.0 | 227 |

## Citing

If you use EquiformerV2 in your work, please consider citing:

```bibtex
@article{equiformer_v2,
  title={{EquiformerV2: Improved Equivariant Transformer for Scaling to Higher-Degree Representations}},
  author={Yi-Lun Liao and Brandon Wood and Abhishek Das* and Tess Smidt*},
  journal={arxiv preprint arxiv:2306.12059},
  year={2023},
}
```
