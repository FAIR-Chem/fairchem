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

Phonons
------------------

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
