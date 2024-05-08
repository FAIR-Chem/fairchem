---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Making LMDB Datasets (original format, deprecated for ASE LMDBs)

Storing your data in an LMDB ensures very fast random read speeds for the fastest supported throughput. This was the
recommended option for the majority of fairchem use cases, but has since been deprecated for [ASE LMDB files](ase_dataset_creation)

This notebook provides an overview of how to create LMDB datasets to be used with the OCP repo. This tutorial is intended
for those who wish to use OCP to train on their own datasets. Those interested in just using OCP data need not worry
about these steps as they've been automated as part of this
[download script](https://github.com/FAIR-Chem/fairchem/blob/master/src/core/scripts/download_data.py).

```{code-cell} ipython3
from fairchem.core.preprocessing import AtomsToGraphs
from fairchem.core.datasets import LmdbDataset
import ase.io
from ase.build import bulk
from ase.build import fcc100, add_adsorbate, molecule
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.optimize import BFGS
import matplotlib.pyplot as plt
import lmdb
import pickle
from tqdm import tqdm
import torch
import os
```

## Generate toy dataset: Relaxation of CO on Cu

```{code-cell} ipython3
adslab = fcc100("Cu", size=(2, 2, 3))
ads = molecule("CO")
add_adsorbate(adslab, ads, 3, offset=(1, 1))
cons = FixAtoms(indices=[atom.index for atom in adslab if (atom.tag == 3)])
adslab.set_constraint(cons)
adslab.center(vacuum=13.0, axis=2)
adslab.set_pbc(True)
adslab.set_calculator(EMT())
dyn = BFGS(adslab, trajectory="CuCO_adslab.traj", logfile=None)
dyn.run(fmax=0, steps=1000)
```

```{code-cell} ipython3
raw_data = ase.io.read("CuCO_adslab.traj", ":")
len(raw_data)
```

## Initial Structure to Relaxed Energy/Structure (IS2RE/IS2RS) LMDBs

IS2RE/IS2RS LMDBs utilize the SinglePointLmdb dataset. This dataset expects the data to be contained in a SINGLE LMDB file.
In addition to the attributes defined by AtomsToGraph, the following attributes must be added for the IS2RE/IS2RS tasks:

- pos_relaxed: Relaxed adslab positions
- sid: Unique system identifier, arbitrary
- y_init: Initial adslab energy, formerly Data.y
- y_relaxed: Relaxed adslab energy
- tags (optional): 0 - subsurface, 1 - surface, 2 - adsorbate


As a demo, we will use the above generated data to create an IS2R* LMDB file.

+++

### Initialize AtomsToGraph feature extractor

```{code-cell} ipython3
a2g = AtomsToGraphs(
    max_neigh=50,
    radius=6,
    r_energy=True,    # False for test data
    r_forces=True,    # False for test data
    r_distances=False,
    r_fixed=True,
)
```

### Initialize LMDB file

```{code-cell} ipython3
db = lmdb.open(
    "sample_CuCO.lmdb",
    map_size=1099511627776 * 2,
    subdir=False,
    meminit=False,
    map_async=True,
)
```

### Write data to LMDB

```{code-cell} ipython3
def read_trajectory_extract_features(a2g, traj_path):
    traj = ase.io.read(traj_path, ":")
    tags = traj[0].get_tags()
    images = [traj[0], traj[-1]]
    data_objects = a2g.convert_all(images, disable_tqdm=True)
    data_objects[0].tags = torch.LongTensor(tags)
    data_objects[1].tags = torch.LongTensor(tags)
    return data_objects
```

```{code-cell} ipython3
system_paths = ["CuCO_adslab.traj"]
idx = 0

for system in system_paths:
    # Extract Data object
    data_objects = read_trajectory_extract_features(a2g, system)
    initial_struc = data_objects[0]
    relaxed_struc = data_objects[1]

    initial_struc.y_init = initial_struc.y # subtract off reference energy, if applicable
    del initial_struc.y
    initial_struc.y_relaxed = relaxed_struc.y # subtract off reference energy, if applicable
    initial_struc.pos_relaxed = relaxed_struc.pos

    # Filter data if necessary
    # OCP filters adsorption energies > |10| eV

    initial_struc.sid = idx  # arbitrary unique identifier

    # no neighbor edge case check
    if initial_struc.edge_index.shape[1] == 0:
        print("no neighbors", traj_path)
        continue

    # Write to LMDB
    txn = db.begin(write=True)
    txn.put(f"{idx}".encode("ascii"), pickle.dumps(initial_struc, protocol=-1))
    txn.commit()
    db.sync()
    idx += 1

db.close()
```

```{code-cell} ipython3
dataset = LmdbDataset({"src": "sample_CuCO.lmdb"})
len(dataset)
```

```{code-cell} ipython3
dataset[0]
```

## Structure to Energy and Forces (S2EF) LMDBs

S2EF LMDBs utilize the TrajectoryLmdb dataset. This dataset expects a directory of LMDB files. In addition to the attributes defined by AtomsToGraph, the following attributes must be added for the S2EF task:

- tags (optional): 0 - subsurface, 1 - surface, 2 - adsorbate
- fid: Frame index along the trajcetory
- sid- sid: Unique system identifier, arbitrary

Additionally, a "length" key must be added to each LMDB file.

As a demo, we will use the above generated data to create an S2EF LMDB dataset

```{code-cell} ipython3
os.makedirs("s2ef", exist_ok=True)
db = lmdb.open(
    "s2ef/sample_CuCO.lmdb",
    map_size=1099511627776 * 2,
    subdir=False,
    meminit=False,
    map_async=True,
)
```

```{code-cell} ipython3
tags = raw_data[0].get_tags()
data_objects = a2g.convert_all(raw_data, disable_tqdm=True)


for fid, data in tqdm(enumerate(data_objects), total=len(data_objects)):
    #assign sid
    data.sid = torch.LongTensor([0])

    #assign fid
    data.fid = torch.LongTensor([fid])

    #assign tags, if available
    data.tags = torch.LongTensor(tags)

    # Filter data if necessary
    # OCP filters adsorption energies > |10| eV and forces > |50| eV/A

    # no neighbor edge case check
    if data.edge_index.shape[1] == 0:
        print("no neighbors", traj_path)
        continue

    txn = db.begin(write=True)
    txn.put(f"{fid}".encode("ascii"), pickle.dumps(data, protocol=-1))
    txn.commit()

txn = db.begin(write=True)
txn.put(f"length".encode("ascii"), pickle.dumps(len(data_objects), protocol=-1))
txn.commit()


db.sync()
db.close()
```

```{code-cell} ipython3
dataset = LmdbDataset({"src": "s2ef/"})
len(dataset)
```

```{code-cell} ipython3
dataset[0]
```

### Advanced usage

LmdbDataset supports multiple LMDB files because the need to highly parallelize the dataset construction process. With OCP's largest split containing 135M+ frames, the need to parallelize the LMDB generation process for these was necessary. If you find yourself needing to deal with very large datasets we recommend parallelizing this process.

+++

## Interacting with the LMDBs

Below we demonstrate how to interact with an LMDB to extract particular information.

```{code-cell} ipython3
dataset = LmdbDataset({"src": "s2ef/"})
```

```{code-cell} ipython3
data = dataset[0]
data
```

```{code-cell} ipython3
energies = torch.tensor([data.energy for data in dataset])
energies
```

```{code-cell} ipython3
plt.hist(energies, bins = 10)
plt.yscale("log")
plt.xlabel("Energies")
plt.show()
```
