core.preprocessing.atoms_to_graphs
==================================

.. py:module:: core.preprocessing.atoms_to_graphs

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Attributes
----------

.. autoapisummary::

   core.preprocessing.atoms_to_graphs.AseAtomsAdaptor
   core.preprocessing.atoms_to_graphs.shell


Classes
-------

.. autoapisummary::

   core.preprocessing.atoms_to_graphs.AtomsToGraphs


Module Contents
---------------

.. py:data:: AseAtomsAdaptor
   :value: None


.. py:data:: shell

.. py:class:: AtomsToGraphs(max_neigh: int = 200, radius: int = 6, r_energy: bool = False, r_forces: bool = False, r_distances: bool = False, r_edges: bool = True, r_fixed: bool = True, r_pbc: bool = False, r_stress: bool = False, r_data_keys: collections.abc.Sequence[str] | None = None)

   A class to help convert periodic atomic structures to graphs.

   The AtomsToGraphs class takes in periodic atomic structures in form of ASE atoms objects and converts
   them into graph representations for use in PyTorch. The primary purpose of this class is to determine the
   nearest neighbors within some radius around each individual atom, taking into account PBC, and set the
   pair index and distance between atom pairs appropriately. Lastly, atomic properties and the graph information
   are put into a PyTorch geometric data object for use with PyTorch.

   :param max_neigh: Maximum number of neighbors to consider.
   :type max_neigh: int
   :param radius: Cutoff radius in Angstroms to search for neighbors.
   :type radius: int or float
   :param r_energy: Return the energy with other properties. Default is False, so the energy will not be returned.
   :type r_energy: bool
   :param r_forces: Return the forces with other properties. Default is False, so the forces will not be returned.
   :type r_forces: bool
   :param r_stress: Return the stress with other properties. Default is False, so the stress will not be returned.
   :type r_stress: bool
   :param r_distances: Return the distances with other properties.
   :type r_distances: bool
   :param Default is False:
   :param so the distances will not be returned.:
   :param r_edges: Return interatomic edges with other properties. Default is True, so edges will be returned.
   :type r_edges: bool
   :param r_fixed: Return a binary vector with flags for fixed (1) vs free (0) atoms.
   :type r_fixed: bool
   :param Default is True:
   :param so the fixed indices will be returned.:
   :param r_pbc: Return the periodic boundary conditions with other properties.
   :type r_pbc: bool
   :param Default is False:
   :param so the periodic boundary conditions will not be returned.:
   :param r_data_keys: Return values corresponding to given keys in atoms.info data with other
   :type r_data_keys: sequence of str, optional
   :param properties. Default is None:
   :param so no data will be returned as properties.:

   .. attribute:: max_neigh

      Maximum number of neighbors to consider.

      :type: int

   .. attribute:: radius

      Cutoff radius in Angstoms to search for neighbors.

      :type: int or float

   .. attribute:: r_energy

      Return the energy with other properties. Default is False, so the energy will not be returned.

      :type: bool

   .. attribute:: r_forces

      Return the forces with other properties. Default is False, so the forces will not be returned.

      :type: bool

   .. attribute:: r_stress

      Return the stress with other properties. Default is False, so the stress will not be returned.

      :type: bool

   .. attribute:: r_distances

      Return the distances with other properties.

      :type: bool

   .. attribute:: Default is False, so the distances will not be returned.

      

   .. attribute:: r_edges

      Return interatomic edges with other properties. Default is True, so edges will be returned.

      :type: bool

   .. attribute:: r_fixed

      Return a binary vector with flags for fixed (1) vs free (0) atoms.

      :type: bool

   .. attribute:: Default is True, so the fixed indices will be returned.

      

   .. attribute:: r_pbc

      Return the periodic boundary conditions with other properties.

      :type: bool

   .. attribute:: Default is False, so the periodic boundary conditions will not be returned.

      

   .. attribute:: r_data_keys

      Return values corresponding to given keys in atoms.info data with other

      :type: sequence of str, optional

   .. attribute:: properties. Default is None, so no data will be returned as properties.

      


   .. py:method:: _get_neighbors_pymatgen(atoms: ase.Atoms)

      Preforms nearest neighbor search and returns edge index, distances,
      and cell offsets



   .. py:method:: _reshape_features(c_index, n_index, n_distance, offsets)

      Stack center and neighbor index and reshapes distances,
      takes in np.arrays and returns torch tensors



   .. py:method:: convert(atoms: ase.Atoms, sid=None)

      Convert a single atomic structure to a graph.

      :param atoms: An ASE atoms object.
      :type atoms: ase.atoms.Atoms
      :param sid: An identifier that can be used to track the structure in downstream
      :type sid: uniquely identifying object
      :param tasks. Common sids used in OCP datasets include unique strings or integers.:

      :returns: A torch geometic data object with positions, atomic_numbers, tags,
                and optionally, energy, forces, distances, edges, and periodic boundary conditions.
                Optional properties can included by setting r_property=True when constructing the class.
      :rtype: data (torch_geometric.data.Data)



   .. py:method:: convert_all(atoms_collection, processed_file_path: str | None = None, collate_and_save=False, disable_tqdm=False)

      Convert all atoms objects in a list or in an ase.db to graphs.

      :param atoms_collection:
      :type atoms_collection: list of ase.atoms.Atoms or ase.db.sqlite.SQLite3Database
      :param Either a list of ASE atoms objects or an ASE database.:
      :param processed_file_path:
      :type processed_file_path: str
      :param A string of the path to where the processed file will be written. Default is None.:
      :param collate_and_save: A boolean to collate and save or not. Default is False, so will not write a file.
      :type collate_and_save: bool

      :returns: A list of torch geometric data objects containing molecular graph info and properties.
      :rtype: data_list (list of torch_geometric.data.Data)



