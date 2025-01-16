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

### OMat24 train split
|       Sub-dataset        | No. structures | File size |                                                                    Download                                                                     |
|:------------------------:|:--------------:|:---------:|:-----------------------------------------------------------------------------------------------------------------------------------------------:|
|      rattled-1000        |    122,937     |   21 GB   |            [rattled-1000.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-1000.tar.gz)            |
| rattled-1000-subsampled  |     41,786     |  7.1 GB   | [rattled-1000-subsampled.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-1000-subsampled.tar.gz) |
|       rattled-500        |     75,167     |   13 GB   |             [rattled-500.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-500.tar.gz)             |
|  rattled-500-subsampled  |     43,068     |  7.3 GB   |  [rattled-500-subsampled.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-500-subsampled.tar.gz)  |
|       rattled-300        |     68,593     |   12 GB   |             [rattled-300.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-300.tar.gz)             |
|  rattled-300-subsampled  |     37,393     |  6.4 GB   |  [rattled-300-subsampled.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-300-subsampled.tar.gz)  |
|  aimd-from-PBE-1000-npt  |    223,574     |   26 GB   |  [aimd-from-PBE-1000-npt.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/aimd-from-PBE-1000-npt.tar.gz)  |
|  aimd-from-PBE-1000-nvt  |    215,589     |   24 GB   |  [aimd-from-PBE-1000-nvt.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/aimd-from-PBE-1000-nvt.tar.gz)  |
|  aimd-from-PBE-3000-npt  |     65,244     |   25 GB   |  [aimd-from-PBE-3000-npt.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/aimd-from-PBE-3000-npt.tar.gz)  |
|  aimd-from-PBE-3000-nvt  |     84,063     |   32 GB   |  [aimd-from-PBE-3000-nvt.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/aimd-from-PBE-3000-nvt.tar.gz)  |
|      rattled-relax       |     99,968     |   12 GB   |           [rattled-relax.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-relax.tar.gz)           |
|          Total           |   1,077,382    | 185.8 GB  |

### OMat24 val split (this is a 1M subset used to train eqV2 models from the 5M val split)
**_NOTE:_** The original validation sets contained a duplicated structures. Corrected validation sets were uploaded on 20/12/24. Please see this [issue](https://github.com/FAIR-Chem/fairchem/issues/942)
for more details, an re-download the correct version of the validation sets if needed.

|       Sub-dataset       |   Size    | File Size |                                                                                                                                      Download |
|:-----------------------:|:---------:|:---------:|----------------------------------------------------------------------------------------------------------------------------------------------:|
|      rattled-1000       |  117,004  |  218 MB   |                       [rattled-1000.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-1000.tar.gz) |
| rattled-1000-subsampled |  39,785   |   77 MB   | [rattled-1000-subsampled.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-1000-subsampled.tar.gz) |
|       rattled-500       |  71,522   |  135 MB   |                         [rattled-500.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-500.tar.gz) |
| rattled-500-subsampled  |  41,021   |   79 MB   |   [rattled-500-subsampled.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-500-subsampled.tar.gz) |
|       rattled-300       |  65,235   |  122 MB   |                         [rattled-300.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-300.tar.gz) |
| rattled-300-subsampled  |  35,579   |   69 MB   |   [rattled-300-subsampled.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-300-subsampled.tar.gz) |
| aimd-from-PBE-1000-npt  |  212,737  |  261 MB   |   [aimd-from-PBE-1000-npt.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/aimd-from-PBE-1000-npt.tar.gz) |
| aimd-from-PBE-1000-nvt  |  205,165  |  251 MB   |   [aimd-from-PBE-1000-nvt.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/aimd-from-PBE-1000-nvt.tar.gz) |
| aimd-from-PBE-3000-npt  |  62,130   |  282 MB   |   [aimd-from-PBE-3000-npt.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/aimd-from-PBE-3000-npt.tar.gz) |
| aimd-from-PBE-3000-nvt  |  79,977   |  364 MB   |   [aimd-from-PBE-3000-nvt.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/aimd-from-PBE-3000-nvt.tar.gz) |
|      rattled-relax      |  95,206   |  118 MB   |                     [rattled-relax.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-relax.tar.gz) |
|          Total          | 1,025,361 |  1.98 GB  |


### sAlex Dataset
We also provide the sAlex dataset used for fine-tuning of our OMat models. sAlex is a subsampled, Matbench-Discovery compliant, version of the original [Alexandria](https://alexandria.icams.rub.de/).
sAlex was created by removing structures matched in WBM and only sampling structure along a trajectory with an energy difference greater than 10 meV/atom. For full details,
please see the manuscript.

| Dataset | Split | No. Structures | File Size |                                                                                               Download |
|:-------:|:-----:|:--------------:|:---------:|-------------------------------------------------------------------------------------------------------:|
|  sAlex  | train |   10,447,765   |  7.6 GB   | [train.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/sAlex/train.tar.gz) |
|  sAlex  |  val  |    553,218     |  408 MB   |     [val.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/sAlex/val.tar.gz) |


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

Please consider citing the following paper in any publications that uses this dataset:


```bib
@article{barroso_omat24,
  title={Open Materials 2024 (OMat24) Inorganic Materials Dataset and Models},
  author={Barroso-Luque, Luis and Muhammed, Shuaibi and Fu, Xiang and Wood, Brandon, Dzamba, Misko, and Gao, Meng and Rizvi, Ammar and  Zitnick, C. Lawrence and Ulissi, Zachary W.},
  journal={arXiv preprint arXiv:2410.12771},
  year={2024}
}
@article{schmidt_2023_machine,
  title={Machine-Learning-Assisted Determination of the Global Zero-Temperature Phase Diagram of Materials},
  author={Schmidt, Jonathan and Hoffmann, Noah and Wang, Hai-Chen and Borlido, Pedro and Carri{\c{c}}o, Pedro JMA and Cerqueira, Tiago FT and Botti, Silvana and Marques, Miguel AL},
  journal={Advanced Materials},
  volume={35},
  number={22},
  pages={2210788},
  year={2023},
  url={https://onlinelibrary.wiley.com/doi/full/10.1002/adma.202210788},
  publisher={Wiley Online Library}
}
```
