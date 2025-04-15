
## Open Direct Air Capture 2023 (ODAC23)

### Structure to Energy and Forces (S2EF) task

We provide precomputed LMDBs for train, validation, and the various test sets that can be used directly with the dataloaders provided in our code. The LMDBs contain input structures from all points in relaxation trajectories along with the energy of the structure and the atomic forces. The dataset contains an in-domain test set and 4 out-of-domain test sets (ood-large, ood-linker, ood-topology, and ood-linker & topology). All LMDbs  are compressed into a single `.tar.gz` file.

|Splits    |Size of compressed version (in bytes)    |Size of uncompressed version (in bytes)    | MD5 checksum (download link)    |
|---    |---    |---    |---    |
|Train + Validation + Test (all splits)    |  172G  |  476G  | [162f0660b2f1c9209c5b57f7b9e545a7](https://dl.fbaipublicfiles.com/large_objects/dac/datasets/odac23_s2ef.tar.gz )  |
|    |    |    |    |

The train and val splits are also available in `extxyz` formats. Each trajectory is in stored in a different `extxyz` file.

|Splits    |Size of compressed version (in bytes)    |Size of uncompressed version (in bytes)    | MD5 checksum (download link)    |
|---    |---    |---    |---    |
|Train    |  232G  |  781G  | [381e72fd8b9c055065fd3afff6b0945b](https://dl.fbaipublicfiles.com/large_objects/dac/datasets/extxyz_train.tar.gz )  |
|Val    |  5.1G  |  18G  | [09913759c6e0f8d649f7ec9dff9e0e8b](https://dl.fbaipublicfiles.com/dac/datasets/extxyz_val.tar.gz )  |
|    |    |    |    |

### Initial Structure to Relaxed Structure (IS2RS) / Relaxed Energy (IS2RE) tasks

For IS2RE / IS2RS training, validation and test sets, we provide precomputed LMDBs that can be directly used with dataloaders provided in our code. The LMDBs contain input initial structures and the output relaxed structures and energies. The dataset contains an in-domain test set and 4 out-of-domain test sets (ood-large, ood-linker, ood-topology, and ood-linker & topology). All LMDBs are compressed into a single `.tar.gz` file.


|Splits    |Size of compressed version (in bytes)    |Size of uncompressed version (in bytes)    | MD5 checksum (download link)    |
|---    |---    |---    |---    |
|Train + Validation + Test (all splits)    |  809M | 2.2G |  [f7f2f58669a30abae8cb9ba1b7f2bcd2](https://dl.fbaipublicfiles.com/dac/datasets/odac23_is2r.tar.gz )  |
|    |    |    |    |

### DDEC Charges

We provide DDEC charges computed for all MOFs in the ODAC23 dataset. A small number of MOFs (~2%) are missing these charges because the DDEC calcuations failed for them.

|Size of compressed version (in bytes)    |Size of uncompressed version (in bytes)    | MD5 checksum (download link)    |
|---    |---    |---    |
|  147M | 534M |  [81927b78d9e4184cc3c398e79760126a](https://dl.fbaipublicfiles.com/dac/datasets/ddec.tar.gz )  |
|    |    |    |


### Citing ODAC23

The OpenDAC 2023 (ODAC23) dataset is licensed under a [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/legalcode).

Please consider citing the following paper in any research manuscript using the ODAC23 dataset:


```bibtex
@article{odac23_dataset,
    author = {Anuroop Sriram and Sihoon Choi and Xiaohan Yu and Logan M. Brabson and Abhishek Das and Zachary Ulissi and Matt Uyttendaele and Andrew J. Medford and David S. Sholl},
    title = {The Open DAC 2023 Dataset and Challenges for Sorbent Discovery in Direct Air Capture},
    year = {2023},
    journal={arXiv preprint arXiv:2311.00341},
}
```
