## `fairchem` by FAIR Chemistry
<p align="center">
  <img width="559" height="200" src="https://github.com/FAIR-Chem/fairchem/assets/45150244/d3abd756-cb96-40eb-b3a8-976939a2f0d8"?
</p>

`fairchem` is the [FAIR](https://ai.meta.com/research/) Chemistry's centralized repository of all its data, models, demos, and application efforts for materials science and quantum chemistry.

### Documentation
If you are looking for `Open-Catalyst-Project/ocp`, it can now be found at [`fairchem.core`](src/fairchem/core). Visit its corresponding documentation [here](https://fair-chem.github.io/).

### Contents
The repository is organized into several directories to help you find what you are looking for:

- [`fairchem.core`](src/fairchem/core): State of the art machine learning models for materials science and chemistry
- [`fairchem.data`](src/fairchem/data): Dataset downloads and input generation codes
- [`fairchem.demo`](src/fairchem/demo): Python API for the [Open Catalyst Demo](https://open-catalyst.metademolab.com/)
- [`fairchem.applications`](src/fairchem/applications): Follow up applications and works (AdsorbML, CatTSunami, etc.)
  
### Installation
Packages can be installed in your environment by the following:
```
pip install -e packages/fairchem-{package-to-download}
```

`fairchem.core` requires you to first create your environment
- [Installation Guide](https://fair-chem.github.io/core/install.html)

### Why a single repository?
Since many of our repositories rely heavily on our other repositories, a single repository makes it really easy to test and ensure consistency across repositories. This should also help simplify the installation process for users who are interested in integrating many of the efforts into one place.

### LICENSE
`fairchem` is available under a [MIT License](LICENSE.md).
