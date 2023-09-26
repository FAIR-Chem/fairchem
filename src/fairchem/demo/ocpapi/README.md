# ocpapi

Python library for programmatic use of the [Open Catalyst Demo](https://open-catalyst.metademolab.com/). Users unfamiliar with the Open Catalyst Demo are encouraged to read more about it at that link before continuing.

## Quickstart

The following examples are used to search for *OH binding sites on Pt surfaces. They use the `find_adsorbate_binding_sites` function, which is a high-level workflow on top of other methods included in this library. Once familiar with this routine, users are encouraged to learn about lower-level methods and features that support more advanced use cases.

### Search over all surfaces

```
from ocpapi import find_adsorbate_binding_sites, Model

import logging
logging.basicConfig(level=logging.INFO)

results = await find_adsorbate_binding_sites(
    adsorbate="*OH",
    bulk="mp-126",
    model=Model.GEMNET_OC_BASE_S2EF_ALL_MD,
)
```

Input to this function includes:

* The SMILES string of the adsorbate to place
* The Materials Project ID of the bulk structure from which surfaces will be generated
* The type of the model being used

This function will perform the following steps:

1. Enumerate surfaces of the bulk material
2. On each surface, enumerate initial guesses for adorbate binding sites
3. Run local force-based relaxations of each adsorbate placement using the specified model

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

```
from ocpapi import (
    find_adsorbate_binding_sites, 
    keep_slabs_with_miller_indices, 
    Model,
)

import logging
logging.basicConfig(level=logging.INFO)

results = await find_adsorbate_binding_sites(
    adsorbate="*OH",
    bulk="mp-126",
    model=Model.GEMNET_OC_BASE_S2EF_ALL_MD,
    slab_filter=keep_slabs_with_miller_indices([(1, 1, 0), (1, 1, 1)])
)
```

This example adds the `slab_filter` field, which takes a function that selects out generated surfaces that meet some criteria; in this case, keeping only the surfaces that have Miller indices of (1, 1, 0) or (1, 1, 1).


### Persisting results

**Results should be saved whenever possible in order to avoid expensive recomputation.**

Assuming `results` was generated with the `find_adsorbate_binding_sites` method used above, it contains a list of `AdsorbateSlabRelaxation` objects. Those results can be saved to file with:

```
from ocpapi import AdsorbateSlabRelaxation

with open("results.json", "w") as f:
    f.write(AdsorbateSlabRelaxation.schema().dumps(results, many=True))
```

Similarly, results can be read back from file to a list of `AdsorbateSlabRelaxation` objects with:

```
from ocpapi import AdsorbateSlabRelaxation

with open("results.json", "r") as f:
    results = AdsorbateSlabRelaxation.schema().loads(f.read(), many=True)
```
