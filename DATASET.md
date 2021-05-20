# Open Catalyst 2020 (OC20) Dataset

*NOTE: Data files for all tasks / splits were updated on Feb 10, 2021 due to minor bugs (affecting < 1% of the data) in earlier versions. If you downloaded data before Feb 10, 2021, please re-download the data.*

This page summarizes the dataset download links for S2EF and IS2RE/IS2RS tasks and various splits. The main project website is https://opencatalystproject.org/


## Structure to Energy and Forces (S2EF) task

For this task’s train and validation sets, we provide compressed trajectory files with the input structures and output energies and forces.  We provide precomputed LMDBs for the test sets. To use the train and validation datasets, first download the files and uncompress them. The uncompressed files are used to generate LMDBs, which are in turn used by the dataloaders to train the ML models. Code for the dataloaders and generating the LMDBs may be found in the Github repository.

Four training datasets are provided with different sizes. Each is a subset of the other, i.e., the 2M dataset is contained in the 20M and all datasets.

Four datasets are provided for validation set. Each dataset corresponds to a subsplit used to evaluate different types of extrapolation, in domain (id, same distribution as the training dataset), out of domain adsorbate (ood_ads, unseen adsorbate), out of domain catalyst (ood_cat, unseen catalyst composition), and out of domain both (ood_both, unseen adsorbate and catalyst composition).

For the test sets, we provide precomputed LMDBs for each of the 4 subsplits (In Domain, OOD Adsorbate, OOD Catalyst, OOD Both).

[*Update March 29,2021*]: We provide structures corresponding to molecular dynamics (MD) and rattled data as well.

Each tarball has a README file containing details about file formats, number of structures / trajectories, etc.

|Splits |Size of compressed version (in bytes)  |Size of uncompressed version (in bytes)    |Downloadable link  |MD5 checksum   |
|---    |---    |---    |---    |---    |
|Train  |   |   |   |   |
|all    |225G   |1.1T   |https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_all.tar |12a7087bfd189a06ccbec9bc7add2bcd   |
|20M    |34G    |165G   |https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_20M.tar |953474cb93f0b08cdc523399f03f7c36   |
|2M |3.4G   |17G    |https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_2M.tar  |863bc983245ffc0285305a1850e19cf7   |
|200K   |344M   |1.7G   |https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_200K.tar    |f8d0909c2623a393148435dede7d3a46   |
|   |   |   |   |   |
|Validation |   |   |   |   |
|val_id |1.7G   |8.3G   |https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_id.tar    |f57f7f5c1302637940f2cc858e789410   |
|val_ood_ads    |1.7G   |8.2G   |https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_ood_ads.tar   |431ab0d7557a4639605ba8b67793f053   |
|val_ood_cat    |1.7G   |8.3G   |https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_ood_cat.tar   |532d6cd1fe541a0ddb0aa0f99962b7db   |
|val_ood_both   |1.9G   |9.5G   |https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_ood_both.tar  |5731862978d80502bbf7017d68c2c729   |
|   |   |   |   |   |
|Test (LMDBs for all splits)    |30G    |415G   |https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_test_lmdbs.tar.gz |bcada432482f6e87b24e14b6b744992a   |
|   |   |   |   |   |
|Rattled data   |29G    |136G   |https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_rattled.tar   |40431149b27b64ce1fb40cac4e2e064b   |
|   |   |   |   |   |
|MD data    |42G    |306G   |https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_md.tar    |9fed845aaab8fb4bf85e3a8db57796e0   |
|   |   |   |   |   |




## Initial Structure to Relaxed Structure (IS2RS) and Initial Structure to Relaxed Energy (IS2RE) tasks

For the IS2RS and IS2RE tasks, we are providing:

* One `.tar.gz` file with precomputed LMDBs which once downloaded and uncompressed, can be used directly to train ML models. The LMDBs contain the input initial structures and the output relaxed structures and energies. Training datasets are split by size, with each being a subset of the larger splits, similar to S2EF. The validation and test datasets are broken into subsplits based on different extrapolation evaluations (In Domain, OOD Adsorbate, OOD Catalyst, OOD Both).
* underlying ASE relaxation trajectories for the adsorbate+catalyst in the entire training and validation sets for the IS2RE and IS2RS tasks. These are **not** required to download for training ML models, but are available for interested users.


Each tarball has README file containing details about file formats, number of structures / trajectories, etc.


### LMDBs

|Splits	|Size of compressed version (in bytes)	|Size of uncompressed version (in bytes)	|Downloadable link	|MD5 checksum	|
|---	|---	|---	|---	|---	|
|Train (all splits) + Validation (all splits) + test (all splits)	|8.1G	|97G	|https://dl.fbaipublicfiles.com/opencatalystproject/data/is2res_train_val_test_lmdbs.tar.gz	|cfc04dd2f87b4102ab2f607240d25fb1	|
|	|	|	|	|	|




## Relaxation Trajectories

### Adsorbate+catalyst system trajectories (optional download)

|Split 	|Size of compressed version (in bytes)	|Size of uncompressed version (in bytes)	|Downloadable link	|MD5 checksum	|
|---	|---	|---	|---	|---	|
|All IS2RE/S training (~466k trajectories)	|109G	|841G	|https://dl.fbaipublicfiles.com/opencatalystproject/data/is2res_train_trajectories.tar	|9e3ed4d1e497bfdce4472ee70455edef	|
|	|	|	|	|	|
|IS2RE/S Validation	|	|	|	|	|
|val_id (~25K trajectories)	|5.9G	|46G	|https://dl.fbaipublicfiles.com/opencatalystproject/data/is2res_val_id_trajectories.tar	|fcb71363018fb1e7127db2500e39e11a	|
|val_ood_ads (~25K trajectories)	|5.7G	|44G	|https://dl.fbaipublicfiles.com/opencatalystproject/data/is2res_val_ood_ads_trajectories.tar	|5ced8ea84584aa229d31e693e0fb090f	|
|val_ood_cat (~25K trajectories)	|6.0G	|46G	|https://dl.fbaipublicfiles.com/opencatalystproject/data/is2res_val_ood_cat_trajectories.tar	|88dcc02fd8c174a72d2c416878fc44ff	|
|val_ood_both (~25K trajectories)	|4.4G	|35G	|https://dl.fbaipublicfiles.com/opencatalystproject/data/is2res_val_ood_both_trajectories.tar	|bc74b6474a13542cc56eaa97bd51adfc	|

#### Per-adsorbate trajectories (optional download)
Adsorbate+catalyst trajectories on a per adsorbate basis are provided [here](./DATASET_PER_ADSORBATE.md) to avoid having to download all systems. Note - a few adsorbates are intentionally left out for the test splits.

### Catalyst system trajectories (optional download)

|Number	|Size of compressed version (in bytes)	|Size of uncompressed version (in bytes)	|Downloadable link	|MD5 checksum	|
|---	|---	|---	|---	|---	|
|294k systems	|20G	|151G	|https://dl.fbaipublicfiles.com/opencatalystproject/data/slab_trajectories.tar	|347f4183465810e9b384e7a033baefc7	|

## OC20 mappings

### Data mapping information

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

Downloadable link: https://dl.fbaipublicfiles.com/opencatalystproject/data/oc20_data_mapping.pkl (MD5 checksum: `71705204c12f8710ff43e71fbc6ba29b`)

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
  'adsorption_site': ((4.5, 12.85, 16.13),)}

```

### Adsorbate-catalyst system to catalyst system mapping information

We provide a Python pickle file containing information about the mapping from adsorbate-catalyst systems to their corresponding catalyst systems. Loading the pickle file will load a Python dictionary. The keys of this dictionary are the adsorbate+catalyst system-ids (of the format `random<XYZ>`  where `XYZ` is an integer), and values will be the catalyst system-ids (of the format `random<PQR>` where `PQR` is an integer).

Downloadable link: https://dl.fbaipublicfiles.com/opencatalystproject/data/mapping_adslab_slab.pkl (MD5 checksum: `079041076c3f15d18ecb5d17c509cdfe`)

An example entry is

```
'random1981709': 'random533137'
```

## Dataset changelog

### March 2021

* Modified the pickle corresponding to data mapping information. Now the pickle includes extra information about `miller_index`, `shift`, `top` and `adsorption_site`.
* Added MD and rattled data for S2EF task.

### Version 2, Feb 2021

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

### Version 1, Oct 2020

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

## Citation

The Open Catalyst 2020 (OC20) dataset is licensed under a [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/legalcode). Please cite the following paper in any research manuscript using the OC20 dataset:


```bibtex
@article{ocp_dataset,
author = {Chanussot*, Lowik and Das*, Abhishek and Goyal*, Siddharth and Lavril*, Thibaut and Shuaibi*, Muhammed and Riviere, Morgane and Tran, Kevin and Heras-Domingo, Javier and Ho, Caleb and Hu, Weihua and Palizhati, Aini and Sriram, Anuroop and Wood, Brandon and Yoon, Junwoong and Parikh, Devi and Zitnick, C. Lawrence and Ulissi, Zachary},
title = {Open Catalyst 2020 (OC20) Dataset and Community Challenges},
journal = {ACS Catalysis},
volume = {0},
number = {0},
pages = {6059-6072},
year = {0},
doi = {10.1021/acscatal.0c04525},
URL = {https://doi.org/10.1021/acscatal.0c04525},
}
```
