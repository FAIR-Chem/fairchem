# ocpapi

Python library for programmatic use of the [Open Catalyst Demo](https://open-catalyst.metademolab.com/). Users unfamiliar with the Open Catalyst Demo are encouraged to read more about it before continuing.

## Installation

Ensure you have Python 3.9.1 or newer, and install `ocpapi` using:

```sh
pip install ocpapi
```

## Quickstart

The following examples are used to search for *OH binding sites on Pt surfaces. They use the `find_adsorbate_binding_sites` function, which is a high-level workflow on top of other methods included in this library. Once familiar with this routine, users are encouraged to learn about lower-level methods and features that support more advanced use cases.

### Note about async methods

This package relies heavily on [asyncio](https://docs.python.org/3/library/asyncio.html). The examples throughout this document can be copied to a python repl launched with:
```sh
$ python -m asyncio
```
Alternatively, an async function can be run in a script by wrapping it with [asyncio.run()](https://docs.python.org/3/library/asyncio-runner.html#asyncio.run):
```python
import asyncio
from ocpapi import find_adsorbate_binding_sites

asyncio.run(find_adsorbate_binding_sites(...))
```

### Search over all surfaces

```python
from ocpapi import find_adsorbate_binding_sites

results = await find_adsorbate_binding_sites(
    adsorbate="*OH",
    bulk="mp-126",
)
```

Input to this function includes:

* The SMILES string of the adsorbate to place
* The Materials Project ID of the bulk structure from which surfaces will be generated

This function will perform the following steps:

1. Enumerate surfaces of the bulk material
2. On each surface, enumerate initial guesses for adorbate binding sites
3. Run local force-based relaxations of each adsorbate placement

In addition, this handles:

* Retrying failed calls to the Open Catalyst Demo API
* Retrying submission of relaxations when they are rate limited

This should take 5-10 minutes to finish while hundreds of individual adsorbate placements are relaxed over six unique surfaces of Pt. Each of the objects in the returned list includes (among other details):

* Information about the surface being searched, including its structure and Miller indices
* The initial positions of the adsorbate before relaxation
* The final structure after relaxation
* The predicted energy of the final structure
* The predicted force on each atom in the final structure


### Search over a subset of Miller indices

```python
from ocpapi import (
    find_adsorbate_binding_sites,
    keep_slabs_with_miller_indices,
)

results = await find_adsorbate_binding_sites(
    adsorbate="*OH",
    bulk="mp-126",
    slab_filter=keep_slabs_with_miller_indices([(1, 1, 0), (1, 1, 1)])
)
```

This example adds the `slab_filter` field, which takes a function that selects out generated surfaces that meet some criteria; in this case, keeping only the surfaces that have Miller indices of (1, 1, 0) or (1, 1, 1).


### Persisting results

**Results should be saved whenever possible in order to avoid expensive recomputation.**

Assuming `results` was generated with the `find_adsorbate_binding_sites` method used above, it is an `AdsorbateBindingSites` object. This can be saved to file with:

```python
with open("results.json", "w") as f:
    f.write(results.to_json())
```

Similarly, results can be read back from file to an `AdsorbateBindingSites` object with:

```python
from ocpapi import AdsorbateBindingSites

with open("results.json", "r") as f:
    results = AdsorbateBindingSites.from_json(f.read())
```

### Viewing results in the web UI

Relaxation results can be viewed in a web UI. For example, https://open-catalyst.metademolab.com/results/7eaa0d63-83aa-473f-ac84-423ffd0c67f5 shows the results of relaxing *OH on a Pt (1,1,1) surface; the uuid, "7eaa0d63-83aa-473f-ac84-423ffd0c67f5", is referred to as the `system_id`.

Extending the examples above, the URLs to visualize the results of relaxations on each Pt surface can be obtained with:
```python
urls = [
    slab.ui_url
    for slab in results.slabs
]
```

## Advanced usage

### Changing the model type

The API currently supports two models:
* `GEMNET_OC_BASE_S2EF_ALL_MD`: https://arxiv.org/abs/2204.02782
* `EQUIFORMER_V2_31M_S2EF_ALL_MD` (default): https://arxiv.org/abs/2306.12059

A specific model type can be requested with:
```python
from ocpapi import find_adsorbate_binding_sites, Model

results = await find_adsorbate_binding_sites(
    adsorbate="*OH",
    bulk="mp-126",
    model=Model.GEMNET_OC_BASE_S2EF_ALL_MD,
)
```

### Converting to [ase.Atoms](https://wiki.fysik.dtu.dk/ase/ase/atoms.html) objects

**Important! The `to_ase_atoms()` method described below will fail with an import error if [ase](https://wiki.fysik.dtu.dk/ase) is not installed.**

Two classes have support for generating [ase.Atoms](https://wiki.fysik.dtu.dk/ase/ase/atoms.html) objects:
* `ocpapi.Atoms.to_ase_atoms()`: Adds unit cell, atomic positions, and other structural information to the returned `ase.Atoms` object.
* `ocpapi.AdsorbateSlabRelaxationResult.to_ase_atoms()`: Adds the same structure information to the `ase.Atoms` object. Also adds the predicted forces and energy of the relaxed structure, which can be accessed with the `ase.Atoms.get_potential_energy()` and `ase.Atoms.get_forces()` methods.

For example, the following would generate an `ase.Atoms` object for the first relaxed adsorbate configuration on the first slab generated for *OH binding on Pt:
```python
from ocpapi import find_adsorbate_binding_sites

results = await find_adsorbate_binding_sites(
    adsorbate="*OH",
    bulk="mp-126",
)

ase_atoms = results.slabs[0].configs[0].to_ase_atoms()
```

### Converting to other structure formats

From an `ase.Atoms` object (see previous section), is is possible to [write to other structure formats](https://wiki.fysik.dtu.dk/ase/ase/io/io.html#ase.io.write). Extending the example above, the `ase_atoms` object could be written to a [VASP POSCAR file](https://www.vasp.at/wiki/index.php/POSCAR) with:
```python
from ase.io import write

write("POSCAR", ase_atoms, "vasp")
```

## License

`ocpapi` is released under the [MIT License](LICENSE).

## Citing `ocpapi`

If you use `ocpapi` in your research, please consider citing the [AdsorbML paper](https://www.nature.com/articles/s41524-023-01121-5) (in addition to the relevant datasets / models used):

```bibtex
@article{lan2023adsorbml,
  title={{AdsorbML}: a leap in efficiency for adsorption energy calculations using generalizable machine learning potentials},
  author={Lan*, Janice and Palizhati*, Aini and Shuaibi*, Muhammed and Wood*, Brandon M and Wander, Brook and Das, Abhishek and Uyttendaele, Matt and Zitnick, C Lawrence and Ulissi, Zachary W},
  journal={npj Computational Materials},
  year={2023},
}
```
