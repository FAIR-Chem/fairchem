core.datasets.ase_datasets
==========================

.. py:module:: core.datasets.ase_datasets

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.datasets.ase_datasets.AseAtomsDataset
   core.datasets.ase_datasets.AseReadDataset
   core.datasets.ase_datasets.AseReadMultiStructureDataset
   core.datasets.ase_datasets.AseDBDataset


Functions
---------

.. autoapisummary::

   core.datasets.ase_datasets.apply_one_tags


Module Contents
---------------

.. py:function:: apply_one_tags(atoms: ase.Atoms, skip_if_nonzero: bool = True, skip_always: bool = False)

   This function will apply tags of 1 to an ASE atoms object.
   It is used as an atoms_transform in the datasets contained in this file.

   Certain models will treat atoms differently depending on their tags.
   For example, GemNet-OC by default will only compute triplet and quadruplet interactions
   for atoms with non-zero tags. This model throws an error if there are no tagged atoms.
   For this reason, the default behavior is to tag atoms in structures with no tags.

   :param skip_if_nonzero: If at least one atom has a nonzero tag, do not tag any atoms
   :type skip_if_nonzero: bool
   :param skip_always: Do not apply any tags. This arg exists so that this function can be disabled
                       without needing to pass a callable (which is currently difficult to do with main.py)
   :type skip_always: bool


.. py:class:: AseAtomsDataset(config: dict, atoms_transform: Callable[[ase.Atoms, Any, Ellipsis], ase.Atoms] = apply_one_tags)

   Bases: :py:obj:`fairchem.core.datasets.base_dataset.BaseDataset`, :py:obj:`abc.ABC`


   This is an abstract Dataset that includes helpful utilities for turning
   ASE atoms objects into OCP-usable data objects. This should not be instantiated directly
   as get_atoms_object and load_dataset_get_ids are not implemented in this base class.

   Derived classes must add at least two things:
       self.get_atoms_object(id): a function that takes an identifier and returns a corresponding atoms object

       self.load_dataset_get_ids(config: dict): This function is responsible for any initialization/loads
           of the dataset and importantly must return a list of all possible identifiers that can be passed into
           self.get_atoms_object(id)

   Identifiers need not be any particular type.


   .. py:attribute:: a2g


   .. py:attribute:: key_mapping


   .. py:attribute:: transforms


   .. py:attribute:: atoms_transform


   .. py:attribute:: ids


   .. py:attribute:: num_samples


   .. py:method:: __getitem__(idx)


   .. py:method:: get_atoms(idx: str | int) -> ase.Atoms
      :abstractmethod:



   .. py:method:: _load_dataset_get_ids(config)
      :abstractmethod:



   .. py:method:: get_relaxed_energy(identifier)
      :abstractmethod:



   .. py:method:: sample_property_metadata(num_samples: int = 100) -> dict


   .. py:method:: get_metadata(attr, idx)


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



