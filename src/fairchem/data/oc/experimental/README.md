Place for a variety of scripts. More like a scratch directory.
---------------------------------------------------------------


`merge_traj`.py: Concatenates intermediate checkpoints into a single, full trajectory file.
Arguments: path, directory to loop through and merge runs.

Assumes directories are structured as follows:

- `merge_traj.py`
  - adsorbate directories (int)	
    - random_id directories
    - dated random id directories
      - checkpoints
      - relaxation_outputs.zip

---------------------------------------------------------------

`utils.py`: Plotting and small trajectory error checking.

- Plots energy profile of trajectory
- Checks whether trajectory is part of V0 dataset
- Checks whether trajectory observes cyclical energy profile, possible checkpoint bug.

---------------------------------------------------------------

`perturb_systems.py`: Rattle atoms objects at varying variances, stores them in independent directories along
with input files necessary to run VASP calculations.

Assumes the same directory structure as needed by `merge_traj.py`. Assumes that the required KPOINTS and POTCAR files are found in `./input_files/{adsorbate_id}/{random_id}`. Similarly, INCAR and SUB_vasp.sh scripts are expected to be found in `./base_inputs`.

---------------------------------------------------------------

`rattle_test.py`: Tests whether ASE's `.rattle()` method rattles constrained atoms.

Tests passed --> rattle does not modify fixed atom positions.
