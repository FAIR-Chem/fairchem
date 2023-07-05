# EquiformerV2: Improved Equivariant Transformer for Scaling to Higher-Degree Representations

Yi-Lun Liao, Brandon Wood, Abhishek Das*, Tess Smidt*

[[`arXiv:2306.12059`](https://arxiv.org/abs/2306.12059)]

NOTE: Please refer to the [official EquiformerV2 codebase](https://github.com/atomicarchitects/equiformer_v2)
for installation instructions and for up-to-date code that reproduces numbers in
the paper.

The version of EquiformerV2 code within this OCP repository is meant to make it
easier to use EquiformerV2 as part of the OCP toolkit and to ease future
development.

## OC20 checkpoints and configs

We provide model weights for EquiformerV2 trained on S2EF-2M dataset for 30 epochs,
EquiformerV2 (31M) trained on S2EF-All+MD, and EquiformerV2 (153M) trained on S2EF-All+MD.

|Model	|Training Split	|Download	| S2EF val force MAE (meV / Å) | S2EF val energy MAE (meV) | Test results
|---	|---	|---	|---	|---	|--- |
|EquiformerV2 (83M)	|2M	|[checkpoint](https://dl.fbaipublicfiles.com/opencatalystproject/models/2023_06/oc20/s2ef/eq2_83M_2M.pt) \| [config](https://github.com/Open-Catalyst-Project/ocp/blob/main/configs/s2ef/2M/equiformer_v2/equiformer_v2_N@12_L@6_M@2.yml)	|19.4 | 278 | - |
|EquiformerV2 (31M)|All+MD |[checkpoint](https://dl.fbaipublicfiles.com/opencatalystproject/models/2023_06/oc20/s2ef/eq2_31M_ec4_allmd.pt) \| [config](https://github.com/Open-Catalyst-Project/ocp/blob/main/configs/s2ef/all/equiformer_v2/equiformer_v2_N@8_L@4_M@2_31M.yml) |16.3 | 232 | [S2EF](https://evalai.s3.amazonaws.com/media/submission_files/submission_289655/7208829e-f32b-4b61-aab3-a1c26b3e67da.json) \| [IS2RE](https://evalai.s3.amazonaws.com/media/submission_files/submission_289660/4b4da09a-9d67-4e83-9a3a-8e9c0e4b763f.json) \| [IS2RS](https://evalai.s3.amazonaws.com/media/submission_files/submission_289662/d38ac10a-e692-4354-a8c1-5af169f35640.json) |
|EquiformerV2 (153M) |All+MD | [checkpoint](https://dl.fbaipublicfiles.com/opencatalystproject/models/2023_06/oc20/s2ef/eq2_153M_ec4_allmd.pt) \| [config](https://github.com/Open-Catalyst-Project/ocp/blob/main/configs/s2ef/all/equiformer_v2/equiformer_v2_N@20_L@6_M@3_153M.yml) |15.0 | 227 | [S2EF](https://evalai.s3.amazonaws.com/media/submission_files/submission_277316/064d8657-4901-4c8b-89d2-5b13a171188d.json) \| [IS2RE](https://evalai.s3.amazonaws.com/media/submission_files/submission_277553/61652a78-539b-457d-927d-43a1f756d3a5.json) \| [IS2RS](https://evalai.s3.amazonaws.com/media/submission_files/submission_277562/c573bba6-156e-48c6-8a4e-e1293e1ce99b.json) |

## Running EquiformerV2

* If you haven't trained OCP models before and are specifically interested in EquiformerV2,
the training / validation scripts provided in the [official EquiformerV2 codebase](https://github.com/atomicarchitects/equiformer_v2/tree/main)
might be easier to get started.
* We provide a [slightly modified trainer](https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/models/equiformer_v2/trainers/forces_trainer.py) and LR scheduler. The differences
from the parent `forces` trainer are the following:
    - Different way of setting up model parameters with no weight decay.
    - Support for cosine LR scheduler.
    - When using the LR scheduler, it first converts the epochs into number of
      steps and then passes it to the scheduler. That way in the config
      everything can be specified in terms of epochs.
* To run training ([similar workflow as other OCP models](https://github.com/Open-Catalyst-Project/ocp/blob/main/TRAIN.md#structure-to-energy-and-forces-s2ef)):
  ```bash
  python main.py \
    --config-yml configs/s2ef/2M/equiformer_v2/equiformer_v2_N@12_L@6_M@2.yml \
    --mode train
  ```
* To run validation with a pretrained model checkpoint:
  ```bash
  python main.py \
    --config-yml configs/s2ef/2M/equiformer_v2/equiformer_v2_N@12_L@6_M@2.yml \
    --checkpoint path/to/checkpoint.pt \
    --mode validate
  ```

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
