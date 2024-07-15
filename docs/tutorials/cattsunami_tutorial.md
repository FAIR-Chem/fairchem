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

# CatTSunami Tutorial

```{code-cell} ipython3
from fairchem.applications.cattsunami.core import Reaction
from fairchem.data.oc.core import Slab, Adsorbate, Bulk, AdsorbateSlabConfig
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from ase.optimize import BFGS
from x3dase.visualize import view_x3d_n
from ase.io import read
from x3dase.x3d import X3D
from fairchem.applications.cattsunami.databases import DISSOCIATION_REACTION_DB_PATH
from fairchem.data.oc.databases.pkls import ADSORBATE_PKL_PATH, BULK_PKL_PATH
from fairchem.core.models.model_registry import model_name_to_local_file
import matplotlib.pyplot as plt
from fairchem.applications.cattsunami.core.autoframe import AutoFrameDissociation
from fairchem.applications.cattsunami.core import OCPNEB
from ase.io import read

#Optional
from IPython.display import Image
from x3dase.x3d import X3D

#Set random seed
import numpy as np
np.random.seed(22)
```

## Do enumerations in an AdsorbML style

```{code-cell} ipython3
# Instantiate the reaction class for the reaction of interest
reaction = Reaction(reaction_str_from_db="*CH -> *C + *H",
                    reaction_db_path=DISSOCIATION_REACTION_DB_PATH,
                    adsorbate_db_path = ADSORBATE_PKL_PATH)
```

```{code-cell} ipython3
# Instantiate our adsorbate class for the reactant and product
reactant = Adsorbate(adsorbate_id_from_db=reaction.reactant1_idx, adsorbate_db_path=ADSORBATE_PKL_PATH)
product1 = Adsorbate(adsorbate_id_from_db=reaction.product1_idx, adsorbate_db_path=ADSORBATE_PKL_PATH)
product2 = Adsorbate(adsorbate_id_from_db=reaction.product2_idx, adsorbate_db_path=ADSORBATE_PKL_PATH)
```

```{code-cell} ipython3
# Grab the bulk and cut the slab we are interested in
bulk = Bulk(bulk_src_id_from_db="mp-33", bulk_db_path=BULK_PKL_PATH)
slab = Slab.from_bulk_get_specific_millers(bulk = bulk, specific_millers=(0,0,1))
```

```{code-cell} ipython3
# Perform site enumeration
# For AdsorbML num_sites = 100, but we use 5 here for brevity. This should be increased for practical use.
reactant_configs = AdsorbateSlabConfig(slab = slab[0], adsorbate = reactant,
                                       mode="random_site_heuristic_placement",
                                       num_sites = 5).atoms_list
product1_configs = AdsorbateSlabConfig(slab = slab[0], adsorbate = product1,
                                      mode="random_site_heuristic_placement",
                                      num_sites = 5).atoms_list
product2_configs = AdsorbateSlabConfig(slab = slab[0], adsorbate = product2,
                                      mode="random_site_heuristic_placement",
                                      num_sites = 5).atoms_list
```

```{code-cell} ipython3
# Instantiate the calculator
# NOTE: If you have a GPU, use cpu = False
# NOTE: Change the checkpoint path to locally downloaded files as needed
checkpoint_path = model_name_to_local_file('EquiformerV2-31M-S2EF-OC20-All+MD', local_cache='/tmp/ocp_checkpoints/')
cpu = True
calc = OCPCalculator(checkpoint_path = checkpoint_path, cpu = cpu)
```

```{code-cell} ipython3
# Relax the reactant systems
reactant_energies = []
for config in reactant_configs:
    config.calc = calc
    opt = BFGS(config)
    opt.run(fmax = 0.05, steps=200)
    reactant_energies.append(config.get_potential_energy())
```

```{code-cell} ipython3
# Relax the product systems
product1_energies = []
for config in product1_configs:
    config.calc = calc
    opt = BFGS(config)
    opt.run(fmax = 0.05, steps=200)
    product1_energies.append(config.get_potential_energy())
```

```{code-cell} ipython3
product2_energies = []
for config in product2_configs:
    config.calc = calc
    opt = BFGS(config)
    opt.run(fmax = 0.05, steps=200)
    product2_energies.append(config.get_potential_energy())
```

## Enumerate NEBs

```{code-cell} ipython3
Image(filename="dissociation_scheme.png")
```

```{code-cell} ipython3
af = AutoFrameDissociation(
            reaction = reaction,
            reactant_system = reactant_configs[reactant_energies.index(min(reactant_energies))],
            product1_systems = product1_configs,
            product1_energies = product1_energies,
            product2_systems = product2_configs,
            product2_energies = product2_energies,
            r_product1_max=2, #r1 in the above fig
            r_product2_max=3, #r3 in the above fig
            r_product2_min=1, #r2 in the above fig
)
```

```{code-cell} ipython3
nframes = 10
frame_sets, mapping_idxs = af.get_neb_frames(calc,
                               n_frames = nframes,
                               n_pdt1_sites=4, # = 5 in the above fig (step 1)
                               n_pdt2_sites = 4, # = 5 in the above fig (step 2)
                              )
```

## Run NEBs

```{code-cell} ipython3
## This will run all NEBs enumerated - to just run one, run the code cell below.
# On GPU, each NEB takes an average of ~1 minute so this could take around a half hour on GPU
# But much longer on CPU
# Remember that not all NEBs will converge -- the k, nframes would be adjusted to achieve convergence

# fmax = 0.05 # [eV / ang**2]
# delta_fmax_climb = 0.4
# converged_idxs = []

# for idx, frame_set in enumerate(frame_sets):
#     neb = OCPNEB(
#         frame_set,
#         checkpoint_path=checkpoint_path,
#         k=1,
#         batch_size=8,
#         cpu = cpu,
#     )
#     optimizer = BFGS(
#         neb,
#         trajectory=f"ch_dissoc_on_Ru_{idx}.traj",
#     )
#     conv = optimizer.run(fmax=fmax + delta_fmax_climb, steps=200)
#     if conv:
#         neb.climb = True
#         conv = optimizer.run(fmax=fmax, steps=300)
#         if conv:
#             converged_idxs.append(idx)
            
# print(converged_idxs)
```

```{code-cell} ipython3
# If you run the above cell -- dont run this one
fmax = 0.05 # [eV / ang**2]
delta_fmax_climb = 0.4
neb = OCPNEB(
    frame_sets[0],
    checkpoint_path=checkpoint_path,
    k=1,
    batch_size=8,
    cpu = cpu,
)
optimizer = BFGS(
    neb,
    trajectory=f"ch_dissoc_on_Ru_0.traj",
)
conv = optimizer.run(fmax=fmax + delta_fmax_climb, steps=200)
if conv:
    neb.climb = True
    conv = optimizer.run(fmax=fmax, steps=300)
```

## Visualize the results

```{code-cell} ipython3
optimized_neb = read(f"ch_dissoc_on_Ru_{converged_idxs[0]}.traj", ":")[-1*nframes:]
```

```{code-cell} ipython3
es  = []
for frame in optimized_neb:
    frame.set_calculator(calc)
    es.append(frame.get_potential_energy())
```

```{code-cell} ipython3
# Plot the reaction coordinate

es = [e - es[0] for e in es]
plt.plot(es)
plt.xlabel("frame number")
plt.ylabel("relative energy [eV]")
plt.title(f"CH dissociation on Ru(0001), Ea = {max(es):1.2f} eV")
plt.savefig("CH_dissoc_on_Ru_0001.png")
```

```{code-cell} ipython3
# Make an interative html file of the optimized neb trajectory
x3d = X3D(optimized_neb)
x3d.write("optimized_neb_ch_disoc_on_Ru0001.html")
```

```{code-cell} ipython3

```
