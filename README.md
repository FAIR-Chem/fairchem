<h1 align="center"> <code>fairchem</code> by FAIR Chemistry </h1>

<p align="center">
  <img width="559" height="200" src="https://github.com/FAIR-Chem/fairchem/assets/45150244/5872c21c-8f39-41af-b703-af9817f0affe"?
</p>


<h4 align="center">

![tests](https://github.com/FAIR-Chem/fairchem/actions/workflows/test.yml/badge.svg?branch=main)
[![documentation](https://github.com/FAIR-Chem/fairchem/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/FAIR-Chem/fairchem/actions/workflows/docs.yml)
[![Static Badge](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)

</h4>

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
pip install -e packages/fairchem-{fairchem-package-name}
```

`fairchem.core` requires you to first create your environment
- [Installation Guide](https://fair-chem.github.io/core/install.html)

### Why a single repository?
Since many of our repositories rely heavily on our other repositories, a single repository makes it really easy to test and ensure consistency across repositories. This should also help simplify the installation process for users who are interested in integrating many of the efforts into one place.

### LICENSE
`fairchem` is available under a [MIT License](LICENSE.md).
