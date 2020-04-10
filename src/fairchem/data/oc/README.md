# Open-Catalyst-Dataset
Workflow for creating and analyzing the Open Catalyst Dataset

# Dependencies
See `setup.py`

# Usage
```
from ocdata.adsorptions import sample_structures
from ocdata.vasp import run_vasp, write_vasp_input_files

adslab, slab = sample_structures()  

#run_vasp(adslab)                # To run VASP
write_vasp_input_files(adslab)  # To write input files only
```
