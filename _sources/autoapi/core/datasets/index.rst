core.datasets
=============

.. py:module:: core.datasets


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/core/datasets/_utils/index
   /autoapi/core/datasets/ase_datasets/index
   /autoapi/core/datasets/base_dataset/index
   /autoapi/core/datasets/embeddings/index
   /autoapi/core/datasets/lmdb_database/index
   /autoapi/core/datasets/lmdb_dataset/index
   /autoapi/core/datasets/oc22_lmdb_dataset/index
   /autoapi/core/datasets/target_metadata_guesser/index


Classes
-------

.. autoapisummary::

   core.datasets.AseDBDataset
   core.datasets.AseReadDataset
   core.datasets.AseReadMultiStructureDataset
   core.datasets.LMDBDatabase
   core.datasets.LmdbDataset


Functions
---------

.. autoapisummary::

   core.datasets.create_dataset
   core.datasets.data_list_collater


Package Contents
----------------

.. py:class:: AseDBDataset(config: dict, atoms_transform: Callable[[ase.Atoms, Any, Ellipsis], ase.Atoms] = apply_one_tags)

   Bases: :py:obj:`AseAtomsDataset`


   This Dataset connects to an ASE Database, allowing the storage of atoms objects
   with a variety of backends including JSON, SQLite, and database server options.

   For more information, see:
   https://databases.fysik.dtu.dk/ase/ase/db/db.html

   :param config:
                  src (str): Either
                          - the path an ASE DB,
                          - the connection address of an ASE DB,
                          - a folder with multiple ASE DBs,
                          - a list of folders with ASE DBs
                          - a glob string to use to find ASE DBs, or
                          - a list of ASE db paths/addresses.
                          If a folder, every file will be attempted as an ASE DB, and warnings
                          are raised for any files that can't connect cleanly

                          Note that for large datasets, ID loading can be slow and there can be many
                          ids, so it's advised to make loading the id list as easy as possible. There is not
                          an obvious way to get a full list of ids from most ASE dbs besides simply looping
                          through the entire dataset. See the AseLMDBDataset which was written with this usecase
                          in mind.

                  connect_args (dict): Keyword arguments for ase.db.connect()

                  select_args (dict): Keyword arguments for ase.db.select()
                          You can use this to query/filter your database

                  a2g_args (dict): Keyword arguments for fairchem.core.preprocessing.AtomsToGraphs()
                          default options will work for most users

                          If you are using this for a training dataset, set
                          "r_energy":True, "r_forces":True, and/or "r_stress":True as appropriate
                          In that case, energy/forces must be in the database

                  keep_in_memory (bool): Store data in memory. This helps avoid random reads if you need
                          to iterate over a dataset many times (e.g. training for many epochs).
                          Not recommended for large datasets.

                  atoms_transform_args (dict): Additional keyword arguments for the atoms_transform callable

                  transforms (dict[str, dict]): Dictionary specifying data transforms as {transform_function: config}
                          where config is a dictionary specifying arguments to the transform_function

                  key_mapping (dict[str, str]): Dictionary specifying a mapping between the name of a property used
                      in the model with the corresponding property as it was named in the dataset. Only need to use if
                      the name is different.
   :type config: dict
   :param atoms_transform: Additional preprocessing function applied to the Atoms
                           object. Useful for applying tags, for example.
   :type atoms_transform: callable, optional
   :param transform: deprecated?
   :type transform: callable, optional


   .. py:method:: _load_dataset_get_ids(config: dict) -> list[int]


   .. py:method:: get_atoms(idx: int) -> ase.Atoms

      Get atoms object corresponding to datapoint idx. Useful to read other properties not in data object.
      :param idx: index in dataset
      :type idx: int

      :returns: ASE atoms corresponding to datapoint idx
      :rtype: atoms



   .. py:method:: connect_db(address: str | pathlib.Path, connect_args: dict | None = None) -> ase.db.core.Database
      :staticmethod:



   .. py:method:: __del__()


   .. py:method:: sample_property_metadata(num_samples: int = 100) -> dict


   .. py:method:: get_relaxed_energy(identifier)
      :abstractmethod:



.. py:class:: AseReadDataset(config: dict, atoms_transform: Callable[[ase.Atoms, Any, Ellipsis], ase.Atoms] = apply_one_tags)

   Bases: :py:obj:`AseAtomsDataset`


   This Dataset uses ase.io.read to load data from a directory on disk.
   This is intended for small-scale testing and demonstrations of OCP.
   Larger datasets are better served by the efficiency of other dataset types
   such as LMDB.

   For a full list of ASE-readable filetypes, see
   https://wiki.fysik.dtu.dk/ase/ase/io/io.html

   :param config: src (str): The source folder that contains your ASE-readable files

                  pattern (str): Filepath matching each file you want to read
                          ex. "*/POSCAR", "*.cif", "*.xyz"
                          search recursively with two wildcards: "**/POSCAR" or "**/*.cif"

                  a2g_args (dict): Keyword arguments for fairchem.core.preprocessing.AtomsToGraphs()
                          default options will work for most users

                          If you are using this for a training dataset, set
                          "r_energy":True, "r_forces":True, and/or "r_stress":True as appropriate
                          In that case, energy/forces must be in the files you read (ex. OUTCAR)

                  ase_read_args (dict): Keyword arguments for ase.io.read()

                  keep_in_memory (bool): Store data in memory. This helps avoid random reads if you need
                          to iterate over a dataset many times (e.g. training for many epochs).
                          Not recommended for large datasets.

                  include_relaxed_energy (bool): Include the relaxed energy in the resulting data object.
                          The relaxed structure is assumed to be the final structure in the file
                          (e.g. the last frame of a .traj).

                  atoms_transform_args (dict): Additional keyword arguments for the atoms_transform callable

                  transform_args (dict): Additional keyword arguments for the transform callable

                  key_mapping (dict[str, str]): Dictionary specifying a mapping between the name of a property used
                      in the model with the corresponding property as it was named in the dataset. Only need to use if
                      the name is different.
   :type config: dict
   :param atoms_transform: Additional preprocessing function applied to the Atoms
                           object. Useful for applying tags, for example.
   :type atoms_transform: callable, optional


   .. py:method:: _load_dataset_get_ids(config) -> list[pathlib.Path]


   .. py:method:: get_atoms(idx: str | int) -> ase.Atoms


   .. py:method:: get_relaxed_energy(identifier) -> float


.. py:class:: AseReadMultiStructureDataset(config: dict, atoms_transform: Callable[[ase.Atoms, Any, Ellipsis], ase.Atoms] = apply_one_tags)

   Bases: :py:obj:`AseAtomsDataset`


   This Dataset can read multiple structures from each file using ase.io.read.
   The disadvantage is that all files must be read at startup.
   This is a significant cost for large datasets.

   This is intended for small-scale testing and demonstrations of OCP.
   Larger datasets are better served by the efficiency of other dataset types
   such as LMDB.

   For a full list of ASE-readable filetypes, see
   https://wiki.fysik.dtu.dk/ase/ase/io/io.html

   :param config: src (str): The source folder that contains your ASE-readable files

                  pattern (str): Filepath matching each file you want to read
                          ex. "*.traj", "*.xyz"
                          search recursively with two wildcards: "**/POSCAR" or "**/*.cif"

                  index_file (str): Filepath to an indexing file, which contains each filename
                          and the number of structures contained in each file. For instance:

                          /path/to/relaxation1.traj 200
                          /path/to/relaxation2.traj 150

                          This will overrule the src and pattern that you specify!

                  a2g_args (dict): Keyword arguments for fairchem.core.preprocessing.AtomsToGraphs()
                          default options will work for most users

                          If you are using this for a training dataset, set
                          "r_energy":True, "r_forces":True, and/or "r_stress":True as appropriate
                          In that case, energy/forces must be in the files you read (ex. OUTCAR)

                  ase_read_args (dict): Keyword arguments for ase.io.read()

                  keep_in_memory (bool): Store data in memory. This helps avoid random reads if you need
                          to iterate over a dataset many times (e.g. training for many epochs).
                          Not recommended for large datasets.

                  include_relaxed_energy (bool): Include the relaxed energy in the resulting data object.
                          The relaxed structure is assumed to be the final structure in the file
                          (e.g. the last frame of a .traj).

                  use_tqdm (bool): Use TQDM progress bar when initializing dataset

                  atoms_transform_args (dict): Additional keyword arguments for the atoms_transform callable

                  transform_args (dict): Additional keyword arguments for the transform callable

                  key_mapping (dict[str, str]): Dictionary specifying a mapping between the name of a property used
                      in the model with the corresponding property as it was named in the dataset. Only need to use if
                      the name is different.
   :type config: dict
   :param atoms_transform: Additional preprocessing function applied to the Atoms
                           object. Useful for applying tags, for example.
   :type atoms_transform: callable, optional
   :param transform: Additional preprocessing function for the Data object
   :type transform: callable, optional


   .. py:method:: _load_dataset_get_ids(config) -> list[str]


   .. py:method:: get_atoms(idx: str) -> ase.Atoms


   .. py:method:: sample_property_metadata(num_samples: int = 100) -> dict


   .. py:method:: get_relaxed_energy(identifier) -> float


.. py:function:: create_dataset(config: dict[str, Any], split: str) -> Subset

   Create a dataset from a config dictionary

   :param config: dataset config dictionary
   :type config: dict
   :param split: name of split
   :type split: str

   :returns: dataset subset class
   :rtype: Subset


.. py:class:: LMDBDatabase(filename: str | pathlib.Path | None = None, create_indices: bool = True, use_lock_file: bool = False, serial: bool = False, readonly: bool = False, *args, **kwargs)

   Bases: :py:obj:`ase.db.core.Database`


   Base class for all databases.


   .. py:attribute:: readonly


   .. py:attribute:: ids
      :value: []



   .. py:attribute:: deleted_ids
      :value: []



   .. py:method:: __enter__() -> typing_extensions.Self


   .. py:method:: __exit__(exc_type, exc_value, tb) -> None


   .. py:method:: close() -> None


   .. py:method:: _write(atoms: ase.Atoms | ase.db.row.AtomsRow, key_value_pairs: dict, data: dict | None, idx: int | None = None) -> None


   .. py:method:: _update(idx: int, key_value_pairs: dict | None = None, data: dict | None = None)


   .. py:method:: _write_deleted_ids()


   .. py:method:: delete(ids: list[int]) -> None

      Delete rows.



   .. py:method:: _get_row(idx: int, include_data: bool = True)


   .. py:method:: _get_row_by_index(index: int, include_data: bool = True)

      Auxiliary function to get the ith entry, rather than a specific id



   .. py:method:: _select(keys, cmps: list[tuple[str, str, str]], explain: bool = False, verbosity: int = 0, limit: int | None = None, offset: int = 0, sort: str | None = None, include_data: bool = True, columns: str = 'all')


   .. py:property:: metadata

      Load the metadata from the DB if present


   .. py:property:: _nextid

      Get the id of the next row to be written


   .. py:method:: count(selection=None, **kwargs) -> int

      Count rows.

      See the select() method for the selection syntax.  Use db.count() or
      len(db) to count all rows.



   .. py:method:: _load_ids() -> None

      Load ids from the DB

      Since ASE db ids are mostly 1-N integers, but can be missing entries
      if ids have been deleted. To save space and operating under the assumption
      that there will probably not be many deletions in most OCP datasets,
      we just store the deleted ids.



.. py:class:: LmdbDataset(config)

   Bases: :py:obj:`fairchem.core.datasets.base_dataset.BaseDataset`


   Base Dataset class for all OCP datasets.


   .. py:attribute:: sharded
      :type:  bool

      Dataset class to load from LMDB files containing relaxation
      trajectories or single point computations.
      Useful for Structure to Energy & Force (S2EF), Initial State to
      Relaxed State (IS2RS), and Initial State to Relaxed Energy (IS2RE) tasks.
      The keys in the LMDB must be integers (stored as ascii objects) starting
      from 0 through the length of the LMDB. For historical reasons any key named
      "length" is ignored since that was used to infer length of many lmdbs in the same
      folder, but lmdb lengths are now calculated directly from the number of keys.
      :param config: Dataset configuration
      :type config: dict


   .. py:attribute:: path


   .. py:attribute:: key_mapping


   .. py:attribute:: transforms


   .. py:method:: __getitem__(idx: int) -> T_co


   .. py:method:: connect_db(lmdb_path: pathlib.Path | None = None) -> lmdb.Environment


   .. py:method:: __del__()


   .. py:method:: sample_property_metadata(num_samples: int = 100)


.. py:function:: data_list_collater(data_list: list[torch_geometric.data.data.BaseData], otf_graph: bool = False, to_dict: bool = False) -> torch_geometric.data.data.BaseData | dict[str, torch.Tensor]

