# Open-Catalyst-Dataset
Workflow for creating and analyzing the Open Catalyst Dataset.

We choose a bulk, and from that bulk we can choose from different surfaces. Separately, we choose an adsorbate. Then, the adsorbate is placed on the surface with multiple placement options. VASP input files are generated for the adsorbate placed on the surface, as well as the surface alone.


# Dependencies
See `setup.py`

# Usage

These are the supported use cases and how to run them:

1. Generate VASP input files for one random adsorbate/bulk/surface/config combination, based on a specified random seed. For the following example, files will be in two directories, `outputs/random0_surface/` and `outputs/random0_adslab/`.
```
python sample_structure.py --bulk_db bulk_database.pkl --adsorbate_db adsorbate_database.pkl --output_dir outputs/ --seed 0
```

2. Generate one specified adsorbate, one more more specified bulks, and all possible surfaces and configs. The following example generates files for all adslab placements of adsorbate 10 on all possible surfaces from bulk 20. Files will be stored in `outputs/10_20_0/surface/`, `outputs/10_20_0/adslab0/`, `outputs/10_20_0/adslab1/`, ..., `outputs/10_20_1/surface/`, `outputs/10_20_1/adslab0/`, and so on for all combinations of surfaces and adslab configs. You may also choose multiple bulks (and all of their surfaces and adslab configs) by giving a comma separated list of bulk indices.
```
python sample_structure.py --bulk_db bulk_database.pkl --adsorbate_db adsorbate_database.pkl --output_dir outputs/ --enumerate_all_structures --adsorbate_index 10 --bulk_indices 20
```

3. Generate one specified adsorbate, one or more specified bulks, one specified surface, and all possible configs. This is the same as #2 except only one surface is selected. The following example generates files for all adslab placements of adsorbate 10 on surface 0 from bulk 20, resulting in files in `outputs/10_20_0/surface/`, `outputs/10_20_0/adslab0/`, `outputs/10_20_0/adslab1/`, and so on for all the adslab configs.
```
python sample_structure.py --bulk_db bulk_database.pkl --adsorbate_db adsorbate_database.pkl --output_dir outputs/ --enumerate_all_structures --adsorbate_index 10 --bulk_indices 20 --surface_index 0
```

For any of the above, you can add `--precomputed_structures dir/` to use the precomputed structures rather than calculating all possible surfaces of a given bulk from scratch. `--verbose` will print out additional info.


