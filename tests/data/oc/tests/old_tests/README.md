# Open Catalyst Dataset Tests Content
The structure of this folder focuses on tests done at different stages of data: tests on inputs generated, tests on relaxation trajectories, and tests on post DFT relaxation outputs. The latter two types of tests can also be used for ML relaxations trajectories and outputs.

## Tests on Data Generated
The data here refers to the slab and adslab configurations generated.

**check_inputs**
This script checks for duplicated adslabs and if bulks and adsorbates in training data leaked to val/test out of distribution data. It should be used before running DFT calculations, but can also be used after DFT calculations to double-check. To use, first convert any metadata that describes the slab/adslab into a dataframe. Using the dataframe, check if there are any duplicated adslabs, and make sure bulks and adsorbates in the train split do not show up in the val/test ood splits.
```
# provide a list of metadata, can be parallelzing if needed
metadata = [obtain_metadata(input_dir, split_tag) for (input_dir, split_tag) in inputs]
df = create_df(metadata, df_name=None)

# check if all adslabs generated is unique
print(adslabs_are_unique(df, unique_by=["mpid", "miller", "shift", "top",
                                      "adsorbate", "adsorption_site"]))

# make sure adsorbates in val/test ood_ads, ood_both does not show up in training data
print(check_commonelems(df, split1, split2, check='adsorbate'))

# make sure bulks in val/test ood_ads, ood_both does not show up in training data
print(check_commonelems(df, split1, split2, check='bulks'))
```

You can also double check your adsorbate placement is correct (i.e. there are no isolated adsorbate atoms post placement).
```
is_adsorbate_placed_correct(atoms_input, atoms_tag)
```

## Tests on relaxation trajectories
**compare_inputs_and_trajectory**
This script compares the input atoms object with the first frame of a trajectory given a system id, and check if they are identical. To use:
```
python compare_input_and_trajectory --sysid_file (name).txt --traj_path_by_sysid (name).pkl --input_dir_by_sysid (name).pkl --num_workers X
```

**check_energy_and_forces**
This script checks for three things given a system id.  1) Forces on the final frame of the trajectory are converged (below a threshold defined by users). 2) The final potential energy is lower than the initial potential energy, and potential energies are decreasing in a trajectory (a small spike is acceptable). 3) The adsorption energy is equal to the final energy minus the reference energy. To run this script:
```
python check_energy_and_forces --sysid_file (name).txt --traj_path_by_sysid (name).pkl --ref_energies (name).pkl --adsorption_energies (name).pkl --num_workers X
```
or use the snippet of the script for a particular test.

## Tests on post DFT relaxation outputs

**flag_anomaly**
This script flags any anomalies that happened during relaxations (i.e. adsorbate dissociation, adsorbate desorption, and possible surface reconstruction). This script can be used for both DFT and ML relaxations and should be used after relaxation is done. You will need the initial atoms object and final atoms object from the relaxation trajectory, as well as the list of atoms tags (where 0=bulk atoms, 1=surface atoms, 2=adsorbate atoms).

Here is an example:
```
detector = DetectTrajAnomaly(init_atoms, final_atoms, atoms_tags)
print(detector.is_adsorbate_dissociated())
print(detector.is_adsorbate_desorbed(neighbor_thres=3))
print(detector.is_surface_reconstructed(slab_movement_thres=1))
```
Here we define adsorbate desorption if the adsorbate is not connected to any surface atoms. Connection is defined as neighbor atoms within 3 angstroms (which is a reasonable number). But you can modify that value based on the systems tested. We also consider possible surface reconstruction if any slab surface atoms move more than 1 Angstrom, but this value can also be updated based on the systems tested.
