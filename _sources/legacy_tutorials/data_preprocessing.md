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

### OCP Data Preprocessing Tutorial


This notebook provides an overview of converting ASE Atoms objects to PyTorch Geometric Data objects. To better understand the raw data contained within OC20, check out the following tutorial first: https://github.com/Open-Catalyst-Project/ocp/blob/master/docs/source/tutorials/data_visualization.ipynb

```{code-cell} ipython3
from ocpmodels.preprocessing import AtomsToGraphs
import ase.io
from ase.build import bulk
from ase.build import fcc100, add_adsorbate, molecule
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.optimize import BFGS
```

### Generate toy dataset: Relaxation of CO on Cu

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
print(len(raw_data))
```

### Convert Atoms object to Data object

The AtomsToGraphs class takes in several arguments to control how Data objects created:

- max_neigh (int):   Maximum number of neighbors a given atom is allowed to have, discarding the furthest
- radius (float):      Cutoff radius to compute nearest neighbors around
- r_energy (bool):    Write energy to Data object
- r_forces (bool):    Write forces to Data object
- r_distances (bool): Write distances between neighbors to Data object
- r_edges (bool):     Write neigbhor edge indices to Data object
- r_fixed (bool):     Write indices of fixed atoms to Data object

```{code-cell} ipython3
a2g = AtomsToGraphs(
    max_neigh=50,
    radius=6,
    r_energy=True,
    r_forces=True,
    r_distances=False,
    r_edges=True,
    r_fixed=True,
)
```

```{code-cell} ipython3
data_objects = a2g.convert_all(raw_data, disable_tqdm=True)
```

```{code-cell} ipython3
data = data_objects[0]
data
```

```{code-cell} ipython3
data.atomic_numbers
```

```{code-cell} ipython3
data.cell
```

```{code-cell} ipython3
data.edge_index #neighbor idx, source idx
```

```{code-cell} ipython3
from torch_geometric.utils import degree
# Degree corresponds to the number of neighbors a given node has. Note there is no more than max_neigh neighbors for
# any given node.

degree(data.edge_index[1]) 
```

```{code-cell} ipython3
data.fixed
```

```{code-cell} ipython3
data.force
```

```{code-cell} ipython3
data.pos
```

```{code-cell} ipython3
data.y
```

### Adding additional info to your Data objects

In addition to the above information, the OCP repo requires several other pieces of information for your data to work
with the provided trainers:

- sid (int): A unique identifier for a particular system. Does not affect your model performance, used for prediction saving 
- fid (int) (S2EF only): If training for the S2EF task, your data must also contain a unique frame identifier for atoms objects coming from the same system.
- tags (tensor): Tag information - 0 for subsurface, 1 for surface, 2 for adsorbate. Optional, can be used for training.


Other information may be added her as well if you choose to incorporate other information in your models/frameworks

```{code-cell} ipython3
data_objects = []
for idx, system in enumerate(raw_data):
    data = a2g.convert(system)
    data.fid = idx
    data.sid = 0 # All data points come from the same system, arbitrarly define this as 0
    data_objects.append(data)
```

```{code-cell} ipython3
data = data_objects[100]
data
```

```{code-cell} ipython3
data.sid
```

```{code-cell} ipython3
data.fid
```

Resources:

- https://github.com/Open-Catalyst-Project/ocp/blob/6604e7130ea41fabff93c229af2486433093e3b4/ocpmodels/preprocessing/atoms_to_graphs.py
- https://github.com/Open-Catalyst-Project/ocp/blob/master/scripts/preprocess_ef.py
