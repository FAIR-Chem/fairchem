---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Tutorial for using Fair Chemistry models to relax NEBs

```{code-cell} ipython3
from ase.optimize import BFGS
from ase.io import read

from fairchem.applications.cattsunami.core.autoframe import interpolate
from fairchem.applications.cattsunami.core import OCPNEB
from fairchem.core.models.model_registry import model_name_to_local_file

#Optional
from x3dase.x3d import X3D
import matplotlib.pyplot as plt
from pathlib import Path
import os
```

## Set up inputs

Shown here are the values used consistently throughout the paper.

```{code-cell} ipython3
fmax = 0.05 # [eV / ang]
delta_fmax_climb = 0.4 # this means that when the fmax is below 0.45 eV/Ang climbing image will be turned on
k = 1 # you may adjust this value as you see fit
cpu = True # set to False if you have a GPU


# NOTE: Change the checkpoint path to locally downloaded files as needed
checkpoint_path = model_name_to_local_file('EquiformerV2-31M-S2EF-OC20-All+MD', local_cache='/tmp/ocp_checkpoints/')
```

## If you have your own set of NEB frames

```{code-cell} ipython3
"""
Load your frames (change to the appropriate loading method)
The approach uses ase, so you must provide a list of ase.Atoms objects
with the appropriate constraints.
"""
path_ = Path(__file__).parents[2]
path_ = os.path.join(path_, "src", "fairchem", "applications", "cattsunami", "tutorial", "sample_traj.traj")
frame_set = read(path_, ":")[0:10] # Change to the path to your atoms of the frame set
```

```{code-cell} ipython3
neb = OCPNEB(
    frame_set,
    checkpoint_path=checkpoint_path,
    k=k,
    batch_size=8, # If you get a memory error, try reducing this to 4
    cpu = cpu,
)
optimizer = BFGS(
    neb,
    trajectory=f"your-neb.traj",
)
conv = optimizer.run(fmax=fmax + delta_fmax_climb, steps=200)
if conv:
    neb.climb = True
    conv = optimizer.run(fmax=fmax, steps=300)
```

## If you have a proposed initial and final frame

You may use the `interpolate` function we implemented which is very similar to idpp but not sensative to periodic boundary crossings. Alternatively you can adopt whatever interpolation scheme you prefer. The `interpolate` function lacks some of the extra protections implemented in the `interpolate_and_correct_frames` which is used in the CatTSunami enumeration workflow. Care should be taken to ensure the results are reasonable.

IMPORTANT NOTES: 
1. Make sure the indices in the initial and final frame map to the same atoms
2. Ensure you have the proper constraints on subsurface atoms

```{code-cell} ipython3
---
tags: ["skip-execution"]
---
"""
Load your initial and frames (change to the appropriate loading method)
The approach uses ase, so you must provide ase.Atoms objects
with the appropriate constraints (i.e. fixed subsurface atoms).
"""
initial_frame = read("path-to-your-initial-atoms.traj")
final_frame = read("path-to-your-final-atoms.traj")
num_frames = 10 # you may change this to whatever you like
```

```{code-cell} ipython3
---
tags: ["skip-execution"]
---
frame_set = interpolate(initial_frame, final_frame, num_frames)

neb = OCPNEB(
    frame_set,
    checkpoint_path=checkpoint_path,
    k=k,
    batch_size=8, # If you get a memory error, try reducing this to 4
    cpu = cpu,
)
optimizer = BFGS(
    neb,
    trajectory=f"your-neb.traj",
)
conv = optimizer.run(fmax=fmax + delta_fmax_climb, steps=200)
if conv:
    neb.climb = True
    conv = optimizer.run(fmax=fmax, steps=300)
```

## Visualize the results

```{code-cell} ipython3
---
tags: ["skip-execution"]
---
optimized_neb = read(f"your-neb.traj", ":")[-1*nframes:]
```

```{code-cell} ipython3
---
tags: ["skip-execution"]
---
es  = []
for frame in optimized_neb:
    frame.set_calculator(calc)
    es.append(frame.get_potential_energy())
```

```{code-cell} ipython3
---
tags: ["skip-execution"]
---
# Plot the reaction coordinate

es = [e - es[0] for e in es]
plt.plot(es)
plt.xlabel("frame number")
plt.ylabel("relative energy [eV]")
plt.title(f"Ea = {max(es):1.2f} eV")
plt.savefig("reaction_coordinate.png")
```

```{code-cell} ipython3
---
tags: ["skip-execution"]
---
# Make an interative html file of the optimized neb trajectory
x3d = X3D(optimized_neb)
x3d.write("your-neb.html")
```
