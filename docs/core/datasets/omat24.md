# Open Materials 2024 (OMat24)

## Overview
The OMat24 dataset contains a mix of single point calculations of non-equilibrium structures and
structural relaxations. The dataset contains structures labeled with total energy (eV), forces (eV/A)
and stress (eV/A^3). The dataset is provided in ASE DB compatible lmdb files.

The OMat24 train and val splits are fully compatible with the Matbench-Discovery benchmark test set.
   1. The splits do not contain any structure that has a protostructure label present in the initial or relaxed
      structures of the WBM dataset.
   2. The splits do not include any structure that was generated starting from an Alexandria relaxed structure with
      protostructure lable in the intitial or relaxed structures of the  WBM datset.

## Subdatasets
OMat24 is made up of X subdatasets based on how the structures were generated. The subdatasets included are:
1. rattled-1000-subsampled & rattled-1000
2. rattled-500-subsampled & rattled-300
3. rattled-300-subsampled & rattled-500
4. aimd-from-PBE-1000-npt
5. aimd-from-PBE-1000-nvt
6. aimd-from-PBE-3000-npt
7. aimd-from-PBE-3000-nvt
8. rattled-relax

**Note** There are two subdatasets for the rattled-< T > datasets. Both subdatasets in each pair were generated with the
same procedure as described in our manuscript.

## File contents and downloads
| Splits | Size of compressed version (in bytes) | Size of uncompressed version (in bytes) | Download link |
|--------|---------------------------------------|-----------------------------------------|---------------|
| train  | X GB                                  | X GB                                    | https://huggingface.co/datasets/fairchem/OMAT24 |
| val    | X GB                                  | X GB                                    | https://huggingface.co/datasets/fairchem/OMAT24 |
|        |                                       |                                         | https://huggingface.co/datasets/fairchem/OMAT24 |

## Getting ASE atoms objects
Dataset files are written as `AseLMDBDatabase` objects which are an implementation of an [ASE Database](https://wiki.fysik.dtu.dk/ase/ase/db/db.html),
in LMDB format. A single **.aselmdb* file can be read and queried like any other ASE DB.

You can also read many DB files at once and access atoms objects using the `AseDBDataset` class.

For example to read the **rattled-relax** subdataset,
```python
from fairchem.core.datasets import AseDBDataset

dataset_path = "/path/to/omat24/train/rattled-relax"
config_kwargs = {}  # see tutorial on additiona configuration

dataset = AseDBDataset(config=dict(src=dataset_path, **config_kwargs))

# atoms objects can be retrieved by index
atoms = dataset.get_atoms(0)
```

To read more than one subdataset you can simply pass a list of subdataset paths,
```python
from fairchem.core.datasets import AseDBDataset

config_kwargs = {}  # see tutorial on additiona configuration
dataset_paths = [
    "/path/to/omat24/train/rattled-relax",
    "/path/to/omat24/train/rattled-1000-subsampled",
    "/path/to/omat24/train/rattled-1000",
]
dataset = AseDBDataset(config=dict(src=dataset_paths, **config_kwargs))
```
To read all of the OMat24 training or validations splits simply pass the paths to all subdatasets.

### Citing OMat24

The OMat24 dataset is licensed under a [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/legalcode).

Please consider citing the following paper in any publication that uses this dataset:


```bibtex

```
