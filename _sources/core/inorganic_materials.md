---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

Using FAIR chemistry models and workflow tools like quacc for inorganic materials science
------------------


We're going to use [`quacc`](https://quantum-accelerators.github.io/quacc/index.html) along with FAIR chem calculators for some simple recipes to calculate elastic and phonon properties. `quacc` has the nice property that you can also use many different workflow managers like jobflow, dask, or prefect to scale and parallelize many calculations, including both ML and DFT calculations. 

## Quick setup (without a workflow backend)

1. First, make sure you installed `fairchem[quacc]` to pick up an appropriate version of quacc and the phonon dependencies. 
2. We're going to use OMat24 machine learning interatomic potentials (MLIPs), which require approval and a license agreement. 
    1. Navigate to the [OMat24 model repo](https://huggingface.co/fairchem/OMAT24), login with your HuggingFace account, and request access to the gated models if you haven't already
    2. Make sure `huggingface_hub` has access to the repo by running `huggingface-cli login` and following the instructions:
        a. Navigate to [HuggingFace tokens](https://huggingface.co/settings/tokens)
        b. Click "+Create new token"
        c. type OMat24 in the Repositories permissions field
        d. Click create token
        e. Type the token into the CLI, or alternatively you can run `huggingface-cli login --token YOUR_TOKEN_HERE`. You can also set the environment variable `HF_TOKEN=YOUR_TOKEN_HERE` if that's more convenient.
3. Set up quacc to use your favorite workflow tool (dask, prefect, etc) by setting your [quacc configuration](https://quantum-accelerators.github.io/quacc/user/basics/wflow_overview.html#__tabbed_1_4). 
    a. Tip: If you just want to run like this tutorial is, write a simple file with `WORKFLOW_ENGINE: null` in `~/.quacc.yaml`. This is by far the easiest way to get started if you don't want to worry about parallelization/etc

## Example of a simple relaxation

In `quacc`, a job is a fundamental block of work that can be run in parallel with other jobs. A flow is a sequence of jobs that need to be run to accomplish a goal. See the [quacc docs](https://quantum-accelerators.github.io/quacc/user/basics/wflow_decorators.html) for more information about how this works!

We're going to start simple here - let's run a local relaxation (optimize the unit cell and positions) using a pre-trained EquiformerV2-31M-OMAT24-MP-sAlex checkpoint. This checkpoint has a few fun properties
1. It's a relatively small (31M) parameter model
2. It was pre-trained on the OMat24 dataset, and then fine-tuned on the MPtrj and Alexandria datasets, so it should emit energies and forces that are consistent with the MP GGA (PBE/PBE+U) level of theory

This code will download the appropriate checkpoint from huggingface_hub automatically; if you don't have the right access token specified, you'll hit an permission or 401 error.

```{code-cell} ipython3
import pprint

from ase.build import bulk
from ase.optimize import LBFGS
from quacc.recipes.mlp.core import relax_job

# Make an Atoms object of a bulk Cu structure
atoms = bulk("Cu")

# Run a structure relaxation
result = relax_job(
    atoms,
    method="fairchem",
    model_name="EquiformerV2-31M-OMAT24-MP-sAlex",
    local_cache="./fairchem_checkpoint_cache/",
    opt_params={"fmax": 1e-3, "optimizer": LBFGS},
)
```

```{code-cell} ipython3
pprint.pprint(result)
```

Congratulations; you ran your first relaxation using an OMat24-trained checkpoint and `quacc`!

## Example of an elastic property calculation

Let's do something more interesting that normally takes quite a bit of work in DFT: calculating an elastic constant! Elastic properties are important to understand how strong or easy to deform a material is, or how a material might change if compressed or expanded in specific directions (i.e. the Poisson ratio!).

We don't have to change much code from above, we just use a built-in recipe to calculate the elastic tensor from `quacc`. This recipe
1. (optionally) Relaxes the unit cell using the MLIP
2. Generates a number of deformed unit cells by applying strains
3. For each deformation, a relaxation using the MLIP and (optionally) a single point calculation is run
4. Finally, all of the above calculations are used to calculate the elastic properties of the material

For more documentation, see the quacc docs for [quacc.recipes.mlp.elastic_tensor_flow](https://quantum-accelerators.github.io/quacc/reference/quacc/recipes/mlp/elastic.html#quacc.recipes.mlp.elastic.elastic_tensor_flow)

```{code-cell} ipython3
from ase.build import bulk
from quacc.recipes.mlp.elastic import elastic_tensor_flow

# Make an Atoms object of a bulk Cu structure
atoms = bulk("Cu")

# Run an elastic property calculation with our favorite MLP potential
result = elastic_tensor_flow(
    atoms,
    job_params={
        "all": dict(
            method="fairchem",
            model_name="EquiformerV2-31M-OMAT24-MP-sAlex",
            local_cache="./fairchem_checkpoint_cache/",
        ),
    },
)
```

```{code-cell} ipython3
result["elasticity_doc"].bulk_modulus
```

Congratulations, you ran your first elastic tensor calculation!

## Example of a phonon calculation

Phonon calculations are very important for inorganic materials science to
* Calculate thermal conductivity
* Understand the vibrational modes, and thus entropy and free energy, of a material
* Predict the stability of a material at finite temperature (e.g. 300 K)
among many others! 
We can run a similarly straightforward calculation that
1. Runs a relaxation on the unit cell and atoms
2. Repeats the unit cell a number of times to make it sufficiently large to capture many interesting vibrational models
3. Generatives a number of finite displacement structures by moving each atom of the unit cell a little bit in each direction
4. Running single point calculations on each of (3)
5. Gathering all of the calculations and calculating second derivatives (the hessian matrix!)
6. Calculating the eigenvalues/eigenvectors of the hessian matrix to find the vibrational modes of the material
7. Analyzing the thermodynamic properties of the vibrational modes.

Note that this analysis assumes that all vibrational modes are harmonic, which is a pretty reasonable approximately for low/moderate temperature materials, but becomes less realistic at high temperatures.

```{code-cell} ipython3
from ase.build import bulk
from quacc.recipes.mlp.phonons import phonon_flow

# Make an Atoms object of a bulk Cu structure
atoms = bulk("Cu")

# Run a phonon (hessian) calculation with our favorite MLP potential
result = phonon_flow(
    atoms,
    method="fairchem",
    job_params={
        "all": dict(
            model_name="EquiformerV2-31M-OMAT24-MP-sAlex",
            local_cache="./fairchem_checkpoint_cache/",
        ),
    },
    min_lengths=10.0, # set the minimum unit cell size smaller to be compatible with limited github runner ram
)
```

```{code-cell} ipython3
print(
    f'The entropy at { result["results"]["thermal_properties"]["temperatures"][-1]:.0f} K is { result["results"]["thermal_properties"]["entropy"][-1]:.2f} kJ/mol'
)
```

Congratulations, you ran your first phonon calculation!

## Parallelizing these calculations

These calculations are super straightforward to parallelize because of how `quacc` is written. Simply choose a workflow manager like `parssl` or `prefect` in `quacc`, and run the same code! There are many ways to run these calculations in parallel. The FAIR chemistry team regularly runs hundreds of thousands of calculations in some of these packages at scale.
