
# Making and using ASE datasets

There are multiple ways to train and evaluate FAIRChem models on data other than OC20 and OC22. Writing an LMDB is the most performant option. However, ASE-based dataset formats are also included as a convenience for people with existing data who simply want to try fairchem tools without needing to learn about LMDBs.


## Using an ASE Database

If your data is already in an [ASE Database](https://databases.fysik.dtu.dk/ase/ase/db/db.html), no additional preprocessing is necessary before running training/prediction! Although the ASE DB backends may not be sufficiently high throughput for all use cases, they are generally considered "fast enough" to train on a reasonably-sized dataset with 1-2 GPUs or predict with a single GPU. If you want to effictively utilize more resources than this, please be aware of the potential for this bottleneck and consider writing your data to an LMDB. If your dataset is small enough to fit in CPU memory, use the `keep_in_memory: True` option to avoid this bottleneck.

To use this dataset, we will just have to change our config files to use the ASE DB Dataset rather than the LMDB Dataset:

```yaml
dataset:
  format: ase_db
  train:
    src: # The path/address to your ASE DB
    connect_args:
      # Keyword arguments for ase.db.connect()
    select_args:
      # Keyword arguments for ase.db.select()
      # These can be used to query/filter the ASE DB
    a2g_args:
      r_energy: True
      r_forces: True
      # Set these if you want to train on energy/forces
      # Energy/force information must be in the ASE DB!
    keep_in_memory: False # Keeping the dataset in memory reduces random reads and is extremely fast, but this is only feasible for relatively small datasets!
    include_relaxed_energy: False # Read the last structure's energy and save as "y_relaxed" for IS2RE-Direct training
  val:
    src:
    a2g_args:
      r_energy: True
      r_forces: True
  test:
    src:
    a2g_args:
      r_energy: False
      r_forces: False
      # It is not necessary to have energy or forces if you are just making predictions.
```
## Using ASE-Readable Files

It is possible to train/predict directly on ASE-readable files. This is only recommended for smaller datasets, as directories of many small files do not scale efficiently on all computing infrastructures. There are two options for loading data with the ASE reader:

### Single-Structure Files
This dataset assumes a single structure will be obtained from each file:

```yaml
dataset:
  format: ase_read
  train:
    src: # The folder that contains ASE-readable files
    pattern: # Pattern matching each file you want to read (e.g. "*/POSCAR"). Search recursively with two wildcards: "**/*.cif".
    include_relaxed_energy: False # Read the last structure's energy and save as "y_relaxed" for IS2RE-Direct training

    ase_read_args:
      # Keyword arguments for ase.io.read()
    a2g_args:
      # Include energy and forces for training purposes
      # If True, the energy/forces must be readable from the file (ex. OUTCAR)
      r_energy: True
      r_forces: True
    keep_in_memory: False
```

### Multi-structure Files
This dataset supports reading files that each contain multiple structure (for example, an ASE .traj file). Using an index file, which tells the dataset how many structures each file contains, is recommended. Otherwise, the dataset is forced to load every file at startup and count the number of structures!

```yaml
dataset:
  format: ase_read_multi
  train:
    index_file: Filepath to an index file which contains each filename and the number of structures in each file. e.g.:
            /path/to/relaxation1.traj 200
            /path/to/relaxation2.traj 150
            ...

    # If using an index file, the src and pattern are not necessary
    src: # The folder that contains ASE-readable files
    pattern: # Pattern matching each file you want to read (e.g. "*.traj"). Search recursively with two wildcards: "**/*.xyz".

    ase_read_args:
      # Keyword arguments for ase.io.read()
    a2g_args:
      # Include energy and forces for training purposes
      r_energy: True
      r_forces: True
    keep_in_memory: False
```
