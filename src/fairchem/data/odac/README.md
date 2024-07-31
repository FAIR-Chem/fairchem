# ODAC23 Dataset

To download the ODAC23 dataset, please see the links [here](https://fair-chem.github.io/core/datasets/odac.html).

Pre-trained ML models and configs are available [here](https://fair-chem.github.io/core/model_checkpoints.html#open-direct-air-capture-2023-odac23).

Large ODAC files can be downloaded by running the command `python src/fairchem/core/scripts/download_large_files.py odac` from the root of the fairchem repo.

This repository contains the list of [promising MOFs](https://github.com/FAIR-Chem/fairchem/tree/main/src/fairchem/data/odac/promising_mof) discovered in the ODAC23 paper, as well as details of the [classifical force field calculations](https://github.com/FAIR-Chem/fairchem/tree/main/src/fairchem/data/odac/force_field). 

Information about supercells can be found in [supercell_info.csv](https://dl.fbaipublicfiles.com/opencatalystproject/data/large_files/supercell_info.csv) for each example (this file is downloaded to the local repo only when the above script is run).

## Citing

Please consider citing the following paper in any research manuscript using the ODAC23 dataset:

```bibtex
@article{odac23_dataset,
    author = {Anuroop Sriram and Sihoon Choi and Xiaohan Yu and Logan M. Brabson and Abhishek Das and Zachary Ulissi and Matt Uyttendaele and Andrew J. Medford and David S. Sholl},
    title = {The Open DAC 2023 Dataset and Challenges for Sorbent Discovery in Direct Air Capture},
    year = {2023},
    journal={arXiv preprint arXiv:2311.00341},
}
