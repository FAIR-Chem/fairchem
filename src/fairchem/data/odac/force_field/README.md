# Classical force field calculations

This folder contains data and scripts related to the classical FF analysis performed in this work.

- The `data_w_oms.json` file contains all successful FF interaction energy calculations with both system information and DFT-computed interaction energies. Calculations were performed across the in-domain training, validation, and test sets.
- The `data_w_ml.json` file contains the same information for systems with successful ML interaction energy predictions. Only systems in the in-domain test set are included here.
- The `FF_analysis.py` script performs the error calculations discussed in the paper and generates the four panels of Figure 5. All of the data used in this analysis is contained in 'data_w_oms.json" for reproducibility.
- The `FF_calcs` folder contains example calculations for classical FF interaction energy predictions.
- The `in.lammps` file is a template LAMMPS input file used for all calculations.

Two example calculations are provided: one for CO2 and one for H2O. System information is provided below. Both systems consist of promising MOFs as described in Tables S1 and S2. All topology (data.*) files were generated using LAMMPS Interface with the UFF4MOF force field and a cutoff of 0 Angstroms.

CO2 example: Training set ID `172_393`, system name: `LEWZET_w_CO2_4`

H2O example: Training set ID `189_325`, system name: `PEGCAH_0.06_0_w_H2O_random_1`
