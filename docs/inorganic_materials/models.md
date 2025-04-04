
# Pretrained models

* All config files for the OMat24 models are available in the [`configs/omat24`](https://github.com/FAIR-Chem/fairchem/tree/main/configs/omat24) directory.
* All models are equiformerV2 S2EFS models

**Note** in order to download any of the model checkpoints from the links below, you will need to first request access
through the [OMMAT24 Hugging Face page](https://huggingface.co/fairchem/OMAT24).

### OMat pretrained models

These checkpoints are trained on OMat24 only. Note that predictions are *not* Materials Project compatible.

| Model Name            | Checkpoint	| Config                                                                                      |
|-----------------------|--------------|---------------------------------------------------------------------------------------------|
| EquiformerV2-31M-OMat | [checkpoint](https://huggingface.co/fairchem/OMAT24/blob/main/eqV2_31M_omat.pt) | [config](https://github.com/FAIR-Chem/fairchem/tree/main/configs/omat24/all/eqV2_31M.yml)   |
| EquiformerV2-86M-OMat | [checkpoint](https://huggingface.co/fairchem/OMAT24/blob/main/eqV2_86M_omat.pt) | [config](https://github.com/FAIR-Chem/fairchem/tree/main/configs/omat24/all/eqV2_86M.yml)   |
| EquiformerV2-153M-OMat | [checkpoint](https://huggingface.co/fairchem/OMAT24/blob/main/eqV2_153M_omat.pt) | [config](https://github.com/FAIR-Chem/fairchem/tree/main/configs/omat24/all/eqV2_153M.yml)  |


### MPTrj only models
These models are trained only on the [MPTrj]() dataset.

| Model Name                | Checkpoint	| Config                                                                          |
|---------------------------|--------------|---------------------------------------------------------------------------------|
| EquiformerV2-31M-MP       | [checkpoint](https://huggingface.co/fairchem/OMAT24/blob/main/eqV2_31M_mp.pt) | [config](https://github.com/FAIR-Chem/fairchem/tree/main/configs/omat24/mptrj/eqV2_31M_mptrj.yml) |
| EquiformerV2-31M-DeNS-MP  | [checkpoint](https://huggingface.co/fairchem/OMAT24/blob/main/eqV2_dens_31M_mp.pt) | [config](https://github.com/FAIR-Chem/fairchem/tree/main/configs/omat24/mptrj/eqV2_31M_dens_mptrj.yml) |
| EquiformerV2-86M-DeNS-MP  | [checkpoint](https://huggingface.co/fairchem/OMAT24/blob/main/eqV2_dens_86M_mp.pt) | [config](https://github.com/FAIR-Chem/fairchem/tree/main/configs/omat24/mptrj/eqV2_86M_dens_mptrj.yml) |
| EquiformerV2-153M-DeNS-MP | [checkpoint](https://huggingface.co/fairchem/OMAT24/blob/main/eqV2_dens_153M_mp.pt) | [config](https://github.com/FAIR-Chem/fairchem/tree/main/configs/omat24/mptrj/eqV2_153M_dens_mptrj.yml) |


### Finetuned OMat models
These models are finetuned from the OMat pretrained checkpoints using MPTrj or MPTrj and sub-sampled trajectories
from the 3D PBE Alexandria dataset, which we call Alex.

| Model Name                     | Checkpoint	| Config                                                                             |
|--------------------------------|--------------|------------------------------------------------------------------------------------|
| EquiformerV2-31M-OMat-Alex-MP  | [checkpoint](https://huggingface.co/fairchem/OMAT24/blob/main/eqV2_31M_omat_mp_salex.pt) | [config](https://github.com/FAIR-Chem/fairchem/tree/main/configs/omat24/finetune/eqV2_31M_ft_salexmptrj.yml) |
| EquiformerV2-86M-OMat-Alex-MP  | [checkpoint](https://huggingface.co/fairchem/OMAT24/blob/main/eqV2_86M_omat_mp_salex.pt) | [config](https://github.com/FAIR-Chem/fairchem/tree/main/configs/omat24/finetune/eqV2_86M_ft_salexmptrj.yml) |
| EquiformerV2-153M-OMat-Alex-MP | [checkpoint](https://huggingface.co/fairchem/OMAT24) | [config](https://github.com/FAIR-Chem/fairchem/tree/main/configs/omat24/finetune/eqV2_153M_ft_salexmptrj.yml) |


Please consider citing the following work if you use OMat24 models in your work,
```bibtex
@article{barroso-luqueOpenMaterials20242024,
    title = {Open Materials 2024 (OMat24) Inorganic Materials Dataset and Models},
    author = {Barroso-Luque, Luis and Shuaibi, Muhammed and Fu, Xiang and Wood, Brandon M. and Dzamba, Misko and Gao, Meng and Rizvi, Ammar and Zitnick, C. Lawrence and Ulissi, Zachary W.},
    date = {2024-10-16},
    eprint = {2410.12771},
    eprinttype = {arXiv},
    doi = {10.48550/arXiv.2410.12771},
    url = {http://arxiv.org/abs/2410.12771},
}
```
