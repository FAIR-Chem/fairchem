# Open Catalyst datasets

* [Open Catalyst 2020 (OC20)](#open-catalyst-2020-oc20)
    * [Scripts to download and preprocess the data](#download-and-preprocess-the-dataset)
    * [Structure to Energy and Forces (S2EF)](#structure-to-energy-and-forces-s2ef-task)
    * [Initial Structure to Relaxed Structure (IS2RS) / Relaxed Energy (IS2RE)](#initial-structure-to-relaxed-structure-is2rs-and-initial-structure-to-relaxed-energy-is2re-tasks)
    * [Relaxation Trajectories](#relaxation-trajectories)
    * [Bader charge data](#bader-charge-data)
    * [OC20 metadata](#oc20-mappings)
    * [Changelog](#dataset-changelog)
    * [License and bibtex](#citing-oc20)
* [Open Catalyst 2022 (OC22)](#open-catalyst-2022-oc22)
    * [Structure to Total Energy and Forces (S2EF-total)](#structure-to-total-energy-and-forces-s2ef-total-task)
    * [Initial Structure to Relaxed Structure (IS2RS) / Relaxed Total Energy (IS2RE-total)](#initial-structure-to-relaxed-structure-is2rs-and-initial-structure-to-relaxed-total-energy-is2re-total-tasks)
    * [Relaxation Trajectories](#relaxation-trajectories-1)
    * [OC22 metadata](#oc22-mappings)
    * [License and bibtex](#citing-oc22)

* * *

## Open Catalyst 2020 (OC20)

*NOTE: Data files for all tasks / splits were updated on Feb 10, 2021 due to minor bugs (affecting < 1% of the data) in earlier versions. If you downloaded data before Feb 10, 2021, please re-download the data.*


### Download and preprocess the dataset

IS2* datasets are stored as LMDB files and are ready to be used upon download.
S2EF train+val datasets require an additional preprocessing step.

For convenience, a self-contained script can be found [here](https://github.com/Open-Catalyst-Project/ocp/blob/main/scripts/download_data.py) to download, preprocess, and organize the data directories to be readily usable by the existing [configs](https://github.com/Open-Catalyst-Project/ocp/tree/main/configs).

For IS2*, run the script as:



```
python scripts/download_data.py --task is2re
```



For S2EF train/val, run the script as:



```
python scripts/download_data.py --task s2ef --split SPLIT_SIZE --get-edges --num-workers WORKERS --ref-energy
```





* `--split`: split size to download: `"200k", "2M", "20M", "all", "val_id", "val_ood_ads", "val_ood_cat", or "val_ood_both"`.
* `--get-edges`: includes edge information in LMDBs (~10x storage requirement, ~3-5x slowdown), otherwise, compute edges on the fly (larger GPU memory requirement).
* `--num-workers`: number of workers to parallelize preprocessing across.
* `--ref-energy`: uses referenced energies instead of raw energies.

For S2EF test, run the script as:


```
python scripts/download_data.py --task s2ef --split test
```



To download and process the dataset in a directory other than your local `ocp/data` folder, add the following command line argument `--data-path`.

Note that the baseline [configs](https://github.com/Open-Catalyst-Project/ocp/tree/main/configs)
expect the data to be found in `ocp/data`, make sure you symlink your directory or
modify the paths in the configs accordingly.

The following sections list dataset download links and sizes for various S2EF
and IS2RE/IS2RS task splits. If you used the above `download_data.py` script to
download and preprocess the data, you are good to go and can stop reading here!


### Structure to Energy and Forces (S2EF) task

For this task’s train and validation sets, we provide compressed trajectory files with the input structures and output energies and forces.  We provide precomputed LMDBs for the test sets. To use the train and validation datasets, first download the files and uncompress them. The uncompressed files are used to generate LMDBs, which are in turn used by the dataloaders to train the ML models. Code for the dataloaders and generating the LMDBs may be found in the Github repository.

Four training datasets are provided with different sizes. Each is a subset of the other, i.e., the 2M dataset is contained in the 20M and all datasets.

Four datasets are provided for validation set. Each dataset corresponds to a subsplit used to evaluate different types of extrapolation, in domain (id, same distribution as the training dataset), out of domain adsorbate (ood_ads, unseen adsorbate), out of domain catalyst (ood_cat, unseen catalyst composition), and out of domain both (ood_both, unseen adsorbate and catalyst composition).

For the test sets, we provide precomputed LMDBs for each of the 4 subsplits (In Domain, OOD Adsorbate, OOD Catalyst, OOD Both).

Each tarball has a README file containing details about file formats, number of structures / trajectories, etc.

|Splits |Size of compressed version (in bytes)  |Size of uncompressed version (in bytes)    | MD5 checksum (download link)   |
|---    |---    |---    |---    |
|Train  |   |   |   |   |
|all    |225G   |1.1T   | [12a7087bfd189a06ccbec9bc7add2bcd](https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_all.tar)   |
|20M    |34G    |165G   | [863bc983245ffc0285305a1850e19cf7](https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_20M.tar)   |
|2M |3.4G   |17G    | [953474cb93f0b08cdc523399f03f7c36](https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_2M.tar)   |
|200K   |344M   |1.7G   | [f8d0909c2623a393148435dede7d3a46](https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_200K.tar)   |
|   |   |   |   |   |
|Validation |   |   |   |   |
|val_id |1.7G   |8.3G   | [f57f7f5c1302637940f2cc858e789410](https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_id.tar)   |
|val_ood_ads    |1.7G   |8.2G   | [431ab0d7557a4639605ba8b67793f053](https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_ood_ads.tar)   |
|val_ood_cat    |1.7G   |8.3G   | [532d6cd1fe541a0ddb0aa0f99962b7db](https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_ood_cat.tar)   |
|val_ood_both   |1.9G   |9.5G   | [5731862978d80502bbf7017d68c2c729](https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_ood_both.tar)   |
|   |   |   |   |   |
|Test (LMDBs for all splits)    |30G    |415G   | [bcada432482f6e87b24e14b6b744992a](https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_test_lmdbs.tar.gz)   |
|   |   |   |   |   |
|Rattled data   |29G    |136G   | [40431149b27b64ce1fb40cac4e2e064b](https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_rattled.tar)   |
|   |   |   |   |   |
|MD data    |42G    |306G   | [9fed845aaab8fb4bf85e3a8db57796e0](https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_md.tar)   |
|   |   |   |   |




### Initial Structure to Relaxed Structure (IS2RS) and Initial Structure to Relaxed Energy (IS2RE) tasks

For the IS2RS and IS2RE tasks, we are providing:



* One `.tar.gz` file with precomputed LMDBs which once downloaded and uncompressed, can be used directly to train ML models. The LMDBs contain the input initial structures and the output relaxed structures and energies. Training datasets are split by size, with each being a subset of the larger splits, similar to S2EF. The validation and test datasets are broken into subsplits based on different extrapolation evaluations (In Domain, OOD Adsorbate, OOD Catalyst, OOD Both).
* underlying ASE relaxation trajectories for the adsorbate+catalyst in the entire training and validation sets for the IS2RE and IS2RS tasks. These are **not** required to download for training ML models, but are available for interested users.

Each tarball has README file containing details about file formats, number of structures / trajectories, etc.

|Splits    |Size of compressed version (in bytes)    |Size of uncompressed version (in bytes)    | MD5 checksum (download link)    |
|---    |---    |---    |---    |
|Train (all splits) + Validation (all splits) + test (all splits)    |8.1G    |97G    | [cfc04dd2f87b4102ab2f607240d25fb1](https://dl.fbaipublicfiles.com/opencatalystproject/data/is2res_train_val_test_lmdbs.tar.gz)    |
|Test-challenge 2021 ([challenge details](https://opencatalystproject.org/challenge.html)) |1.3G   |17G    | [aed414cdd240fbb5670b5de6887a138b](https://dl.fbaipublicfiles.com/opencatalystproject/data/is2re_test_challenge_2021.tar.gz)   |
|    |    |    |    |






### Relaxation Trajectories

#### Adsorbate+catalyst system trajectories (optional download)

|Split     |Size of compressed version (in bytes)    |Size of uncompressed version (in bytes)    | MD5 checksum (download link)    |
|---    |---    |---    |---    |
|All IS2RE/S training (~466k trajectories)    |109G    |841G    | [9e3ed4d1e497bfdce4472ee70455edef](https://dl.fbaipublicfiles.com/opencatalystproject/data/is2res_train_trajectories.tar)    |
|    |    |    |    |
|IS2RE/S Validation    |    |    |    |
|val_id (~25K trajectories)    |5.9G    |46G    | [fcb71363018fb1e7127db2500e39e11a](https://dl.fbaipublicfiles.com/opencatalystproject/data/is2res_val_id_trajectories.tar)    |
|val_ood_ads (~25K trajectories)    |5.7G    |44G    | [5ced8ea84584aa229d31e693e0fb090f](https://dl.fbaipublicfiles.com/opencatalystproject/data/is2res_val_ood_ads_trajectories.tar)    |
|val_ood_cat (~25K trajectories)    |6.0G    |46G    | [88dcc02fd8c174a72d2c416878fc44ff](https://dl.fbaipublicfiles.com/opencatalystproject/data/is2res_val_ood_cat_trajectories.tar)    |
|val_ood_both (~25K trajectories)    |4.4G    |35G    | [bc74b6474a13542cc56eaa97bd51adfc](https://dl.fbaipublicfiles.com/opencatalystproject/data/is2res_val_ood_both_trajectories.tar)    |



##### Per-adsorbate trajectories (optional download)

Adsorbate+catalyst trajectories on a per adsorbate basis are provided [here](./DATASET_PER_ADSORBATE.md) to avoid having to download all systems. Note - a few adsorbates are intentionally left out for the test splits.



#### Catalyst system trajectories (optional download)

|Number    |Size of compressed version (in bytes)    |Size of uncompressed version (in bytes)    |MD5 checksum (download link)    |
|---    |---    |---    |---    |
|294k systems    |20G    |151G    | [347f4183465810e9b384e7a033baefc7](https://dl.fbaipublicfiles.com/opencatalystproject/data/slab_trajectories.tar)    |


### Bader charge data
We provide Bader charge data for all final frames of our train + validation systems in OC20 (for both S2EF and IS2RE/RS tasks). A `.tar.gz` file, when downloaded and uncompressed will contain several directories with unique system-ids (of the format `random<XYZ>` where `XYZ` is an integer). Each directory will contain raw Bader charge analysis outputs. For more details on the Bader charge analysis, see https://theory.cm.utexas.edu/henkelman/research/bader/.

Downloadable link: https://dl.fbaipublicfiles.com/opencatalystproject/data/oc20_bader_data.tar (MD5 checksum: `aecc5e23542de49beceb4b7e44c153b9`)

### OC20 mappings

#### Data mapping information

We provide a Python pickle file containing information about the slab and adsorbates for each of the systems in OC20 dataset. Loading the pickle file will load a Python dictionary. The keys of this dictionary are the adsorbate+catalyst system-ids (of the format `random<XYZ>`  where `XYZ` is an integer), and the corresponding value of each key is a dictionary with information about:



* `bulk_mpid` : Materials Project ID of the bulk system used corresponding the the catalyst surface
* `bulk_symbols`  Chemical composition of the bulk counterpart
* `ads_symbols`  Chemical composition of the adsorbate counterpart
* `ads_id` : internal unique identifier, one for each of the 82 adsorbates used in the dataset
* `bulk_id` : internal unique identifier one for each of the 11500 bulks used in the dataset
* `miller_index`: 3-tuple of integers indicating the Miller indices of the surface
* `shift`: c-direction shift used to determine cutoff for the surface (c-direction is following the nomenclature from Pymatgen)
* `top`: boolean indicating whether the chosen surface was at the top or bottom of the originally enumerated surface
* `adsorption_site`: A tuple of 3-tuples containing the Cartesian coordinates of each binding adsorbate atom
* `class`: integer indicating the class of materials the system's slab is part of, where:
* 0 - intermetallics
* 1 - metalloids
* 2 - non-metals
* 3 - halides
* `anomaly`: integer indicating possible anomalies (based off general heuristics, not to be taken as perfect classifications), where:
* 0 - no anomaly
* 1 - adsorbate dissociation
* 2 - adsorbate desorption
* 3 - surface reconstruction
* 4 - incorrect CHCOH placement, appears to be CHCO with a lone, uninteracting, H far off in the unit cell

Downloadable link: https://dl.fbaipublicfiles.com/opencatalystproject/data/oc20_data_mapping.pkl (MD5 checksum: `01c879067a05b4288055a1fdf821e068`)

An example entry is



```
'random2181546': {'bulk_id': 6510,
  'ads_id': 69,
  'bulk_mpid': 'mp-22179',
  'bulk_symbols': 'Si2Ti2Y2',
  'ads_symbols': '*N2',
  'miller_index': (2, 0, 1),
  'shift': 0.145,
  'top': True,
  'adsorption_site': ((4.5, 12.85, 16.13),),
  'class': 1,
  'anomaly': 0}
```





### Adsorbate-catalyst system to catalyst system mapping information

We provide a Python pickle file containing information about the mapping from adsorbate-catalyst systems to their corresponding catalyst systems. Loading the pickle file will load a Python dictionary. The keys of this dictionary are the adsorbate+catalyst system-ids (of the format `random<XYZ>`  where `XYZ` is an integer), and values will be the catalyst system-ids (of the format `random<PQR>` where `PQR` is an integer).

Downloadable link: https://dl.fbaipublicfiles.com/opencatalystproject/data/mapping_adslab_slab.pkl (MD5 checksum: `079041076c3f15d18ecb5d17c509cdfe`)

An example entry is



```
'random1981709': 'random533137'
```





### Dataset changelog

#### September 2021

* Released IS2RE `test-challenge` data for the [Open Catalyst Challenge 2021](https://opencatalystproject.org/challenge.html)

#### March 2021

* Modified the pickle corresponding to data mapping information. Now the pickle includes extra information about `miller_index`, `shift`, `top` and `adsorption_site`.
* Added Molecular Dynamics (MD) and rattled data for S2EF task.

#### Version 2, Feb 2021

Modifications:



* Removed slab systems which had single frame checkpoints, this led to modifications of reference frame energies of 350k frames out of 130M.
* Fixed stitching of checkpoints in adsorbate+catalyst trajectories.
* Added release of slab trajectories.

Below are actual updates numbers, of the form `old` → `new`

Total S2EF frames:



* train: 133953162 → 133934018
* validation:
    * val_id : 1000000 → 999866
    * val_ood_ads: 1000000 → 999838
    * val_ood_cat: 1000000 → 999809
    * val_ood_both: 1000000 →  999944
* test:
    * test_id: 1000000 → 999736
    * test_ood_ads: 1000000 → 999859
    * test_ood_cat: 1000000 → 999826
    * test_ood_both: 1000000 → 999973

Total IS2RE and IS2RS systems:



* train: 461313 → 460328
* validation:
    * val_id : 24946 → 24943
    * val_ood_ads: 24966 → 24961
    * val_ood_cat: 24988 → 24963
    * val_ood_both: 24963 → 24987
* test:
    * test_id: 24951 → 24948
    * test_ood_ads: 24931 → 24930
    * test_ood_cat: 24967 → 24965
    * test_ood_both: 24986 → 24985

#### Version 1, Oct 2020

Total S2EF frames:



* train: 133953162
* validation:
    * val_id : 1000000
    * val_ood_ads: 1000000
    * val_ood_cat: 1000000
    * val_ood_both: 1000000
* test:
    * test_id: 1000000
    * test_ood_ads: 1000000
    * test_ood_cat: 1000000
    * test_ood_both: 1000000

Total IS2RE and IS2RS systems:



* train: 461313
* validation:
    * val_id : 24936
    * val_ood_ads: 24966
    * val_ood_cat: 24988
    * val_ood_both: 24963
* test:
    * test_id: 24951
    * test_ood_ads: 24931
    * test_ood_cat: 24967
    * test_ood_both: 24986

### Citing OC20

The Open Catalyst 2020 (OC20) dataset is licensed under a [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/legalcode).

Please consider citing the following paper in any research manuscript using the OC20 dataset:



```
@article{ocp_dataset,
    author = {Chanussot*, Lowik and Das*, Abhishek and Goyal*, Siddharth and Lavril*, Thibaut and Shuaibi*, Muhammed and Riviere, Morgane and Tran, Kevin and Heras-Domingo, Javier and Ho, Caleb and Hu, Weihua and Palizhati, Aini and Sriram, Anuroop and Wood, Brandon and Yoon, Junwoong and Parikh, Devi and Zitnick, C. Lawrence and Ulissi, Zachary},
    title = {Open Catalyst 2020 (OC20) Dataset and Community Challenges},
    journal = {ACS Catalysis},
    year = {2021},
    doi = {10.1021/acscatal.0c04525},
}
```



## Open Catalyst 2022 (OC22)

### Structure to Total Energy and Forces (S2EF-Total) task

For this task’s train, validation and test sets, we provide precomputed LMDBs that can be directly used with dataloaders provided in our code. The LMDBs contain input structures from all points in relaxation trajectories along with the energy of the structure and the atomic forces. The validation and test datasets are broken into subsplits based on in-distribution and out-of-distribution materials relative to the training dataset. All LMDBs are compressed into a single `.tar.gz` file.


|Splits    |Size of compressed version (in bytes)    |Size of uncompressed version (in bytes)    | MD5 checksum (download link)    |
|---    |---    |---    |---    |
|Train (all splits) + Validation (all splits) + test (all splits)    |   20G |  71G  | [ebea523c6f8d61248a37b4dd660b11e6](https://dl.fbaipublicfiles.com/opencatalystproject/data/oc22/s2ef_total_train_val_test_lmdbs.tar.gz)
|    |    |    |    |




### Initial Structure to Relaxed Structure (IS2RS) and Initial Structure to Relaxed Total Energy (IS2RE-Total) tasks

For IS2RE-Total / IS2RS training, validation and test sets, we provide precomputed LMDBs that can be directly used with dataloaders provided in our code. The LMDBs contain input initial structures and the output relaxed structures and energies. The validation and test datasets are broken into subsplits based on in-distribution and out-of-distribution materials relative to the training dataset. All LMDBs are compressed into a single `.tar.gz` file.


|Splits    |Size of compressed version (in bytes)    |Size of uncompressed version (in bytes)    | MD5 checksum (download link)    |
|---    |---    |---    |---    |
|Train (all splits) + Validation (all splits) + test (all splits)    |  109M |  424M  |  [b35dc24e99ef3aeaee6c5c949903de94](https://dl.fbaipublicfiles.com/opencatalystproject/data/oc22/is2res_total_train_val_test_lmdbs.tar.gz)  |
|    |    |    |    |








### Relaxation Trajectories

#### System trajectories (optional download)


We provide relaxation trajectories for all systems used in train and validation sets of S2EF-Total and IS2RE-Total/RS task:


|Number    |Size of compressed version (in bytes)    |Size of uncompressed version (in bytes)    | MD5 checksum (download link)    |
|---    |---    |---    |---    |
| S2EF and IS2RE (both train and validation)   | 34G   |   80G  |  [977b6be1cbac6864e63c4c7fbf8a3fce](https://dl.fbaipublicfiles.com/opencatalystproject/data/oc22/oc22_trajectories.tar.gz)  |
|    |    |    |    |





### OC22 Mappings

#### Data mapping information



We provide a Python pickle file containing information about the slab and adsorbates for each of the systems in OC22 dataset. Loading the pickle file will load a Python dictionary. The keys of this dictionary are the system-ids (of the format `XYZ`  where `XYZ` is an integer, corresponding to the `sid` in the LMDB Data object), and the corresponding value of each key is a dictionary with information about:


* `bulk_id`: Materials Project ID of the bulk system used corresponding the the catalyst surface
* `bulk_symbols`: Chemical composition of the bulk counterpart
* `miller_index`: 3-tuple of integers indicating the Miller indices of the surface
* `traj_id`: Identifier associated with the accompanying raw trajectory (if available)
* `slab_sid`: Identifier associated with the corresponding slab (if available)
* `ads_symbols`: Chemical composition of the adsorbate counterpart (adosrbate+slabs only)
* `nads`: Number of adsorbates present



Downloadable link:  https://dl.fbaipublicfiles.com/opencatalystproject/data/oc22/oc22_metadata.pkl (MD5 checksum: `13dc06c6510346d8a7f614d5b26c8ffa` )


An example adsorbate+slab entry:

```
 6877: {'bulk_id': 'mp-559112',
  'miller_index': (1, 0, 0),
  'nads': 1,
  'traj_id': 'K2Zn6O7_mp-559112_RyQXa0N0uc_ohyUKozY3G',
  'bulk_symbols': 'K4Zn12O14',
  'slab_sid': 30859,
  'ads_symbols': 'O2'},
```

An example slab entry:

```
 34815: {'bulk_id': 'mp-18793',
  'miller_index': (1, 2, 1),
  'nads': 0,
  'traj_id': 'LiCrO2_mp-18793_clean_3HDHBg6TIz',
  'bulk_symbols': 'Li2Cr2O4'},
```

####

#### OC20 reference information


In order to train models on OC20 total energy, we provide a Python pickle file containing the energy necessary to convert adsorption energy values to total energy. Loading the pickle file will load a Python dictionary. The keys of this dictionary are the system-ids (of the format `random<XYZ>`  where `XYZ` is an integer, corresponding to the `sid` in the LMDB Data object), and the corresponding value of each key is the energy to be added to OC20 energy values. To train on total energies for OC20, specify the path to this pickle file in your training configs.

Downloadable link:  https://dl.fbaipublicfiles.com/opencatalystproject/data/oc22/oc20_ref.pkl (MD5 checksum: `043e1e0b0cce64c62f01a8563dbc3178`)
####

### Citing OC22

The Open Catalyst 2022 (OC22) dataset is licensed under a [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/legalcode).

Please consider citing the following paper in any research manuscript using the OC22 dataset:


```
@article{oc22_dataset,
    author = {Tran*, Richard and Lan*, Janice and Shuaibi*, Muhammed and Wood*, Brandon and Goyal*, Siddharth and Das, Abhishek and Heras-Domingo, Javier and Kolluru, Adeesh and Rizvi, Ammar and Shoghi, Nima and Sriram, Anuroop and Ulissi, Zachary and Zitnick, C. Lawrence},
    title = {The Open Catalyst 2022 (OC22) Dataset and Challenges for Oxide Electrocatalysis},
    year = {2022},
    journal={arXiv preprint arXiv:2206.08917},
}
```
