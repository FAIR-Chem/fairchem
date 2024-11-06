data.oc.core
============

.. py:module:: data.oc.core


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/data/oc/core/adsorbate/index
   /autoapi/data/oc/core/adsorbate_slab_config/index
   /autoapi/data/oc/core/bulk/index
   /autoapi/data/oc/core/interface_config/index
   /autoapi/data/oc/core/ion/index
   /autoapi/data/oc/core/multi_adsorbate_slab_config/index
   /autoapi/data/oc/core/slab/index
   /autoapi/data/oc/core/solvent/index


Classes
-------

.. autoapisummary::

   data.oc.core.Adsorbate
   data.oc.core.AdsorbateSlabConfig
   data.oc.core.Bulk
   data.oc.core.InterfaceConfig
   data.oc.core.Ion
   data.oc.core.MultipleAdsorbateSlabConfig
   data.oc.core.Slab
   data.oc.core.Solvent


Package Contents
----------------

.. py:class:: Adsorbate(adsorbate_atoms: ase.Atoms = None, adsorbate_id_from_db: int | None = None, adsorbate_smiles_from_db: str | None = None, adsorbate_db_path: str = ADSORBATE_PKL_PATH, adsorbate_db: dict[int, tuple[Any, Ellipsis]] | None = None, adsorbate_binding_indices: list | None = None)

   Initializes an adsorbate object in one of 4 ways:
   - Directly pass in an ase.Atoms object.
       For this, you should also provide the index of the binding atom.
   - Pass in index of adsorbate to select from adsorbate database.
   - Pass in the SMILES string of the adsorbate to select from the database.
   - Randomly sample an adsorbate from the adsorbate database.

   :param adsorbate_atoms: Adsorbate structure.
   :type adsorbate_atoms: ase.Atoms
   :param adsorbate_id_from_db: Index of adsorbate to select.
   :type adsorbate_id_from_db: int
   :param adsorbate_smiles_from_db: A SMILES string of the desired adsorbate.
   :type adsorbate_smiles_from_db: str
   :param adsorbate_db_path: Path to adsorbate database.
   :type adsorbate_db_path: str
   :param adsorbate_binding_indices: The index/indices of the adsorbate atoms which are expected to bind.
   :type adsorbate_binding_indices: list


   .. py:attribute:: adsorbate_id_from_db


   .. py:attribute:: adsorbate_db_path


   .. py:method:: __len__()


   .. py:method:: __str__()


   .. py:method:: __repr__()


   .. py:method:: _get_adsorbate_from_random(adsorbate_db)


   .. py:method:: _load_adsorbate(adsorbate: tuple[Any, Ellipsis]) -> None

      Saves the fields from an adsorbate stored in a database. Fields added
      after the first revision are conditionally added for backwards
      compatibility with older database files.



.. py:class:: AdsorbateSlabConfig(slab: fairchem.data.oc.core.slab.Slab, adsorbate: fairchem.data.oc.core.slab.Adsorbate, num_sites: int = 100, num_augmentations_per_site: int = 1, interstitial_gap: float = 0.1, mode: str = 'random')

   Initializes a list of adsorbate-catalyst systems for a given Adsorbate and Slab.

   :param slab: Slab object.
   :type slab: Slab
   :param adsorbate: Adsorbate object.
   :type adsorbate: Adsorbate
   :param num_sites: Number of sites to sample.
   :type num_sites: int
   :param num_augmentations_per_site: Number of augmentations of the adsorbate per site. Total number of
                                      generated structures will be `num_sites` * `num_augmentations_per_site`.
   :type num_augmentations_per_site: int
   :param interstitial_gap: Minimum distance in Angstroms between adsorbate and slab atoms.
   :type interstitial_gap: float
   :param mode: "random", "heuristic", or "random_site_heuristic_placement".
                This affects surface site sampling and adsorbate placement on each site.

                In "random", we do a Delaunay triangulation of the surface atoms, then
                sample sites uniformly at random within each triangle. When placing the
                adsorbate, we randomly rotate it along xyz, and place it such that the
                center of mass is at the site.

                In "heuristic", we use Pymatgen's AdsorbateSiteFinder to find the most
                energetically favorable sites, i.e., ontop, bridge, or hollow sites.
                When placing the adsorbate, we randomly rotate it along z with only
                slight rotation along x and y, and place it such that the binding atom
                is at the site.

                In "random_site_heuristic_placement", we do a Delaunay triangulation of
                the surface atoms, then sample sites uniformly at random within each
                triangle. When placing the adsorbate, we randomly rotate it along z with
                only slight rotation along x and y, and place it such that the binding
                atom is at the site.

                In all cases, the adsorbate is placed at the closest position of no
                overlap with the slab plus `interstitial_gap` along the surface normal.
   :type mode: str


   .. py:attribute:: slab


   .. py:attribute:: adsorbate


   .. py:attribute:: num_sites


   .. py:attribute:: num_augmentations_per_site


   .. py:attribute:: interstitial_gap


   .. py:attribute:: mode


   .. py:attribute:: sites


   .. py:method:: get_binding_sites(num_sites: int)

      Returns up to `num_sites` sites given the surface atoms' positions.



   .. py:method:: place_adsorbate_on_site(adsorbate: fairchem.data.oc.core.slab.Adsorbate, site: numpy.ndarray, interstitial_gap: float = 0.1)

      Place the adsorbate at the given binding site.



   .. py:method:: place_adsorbate_on_sites(sites: list, num_augmentations_per_site: int = 1, interstitial_gap: float = 0.1)

      Place the adsorbate at the given binding sites.



   .. py:method:: _get_scaled_normal(adsorbate_c: ase.Atoms, slab_c: ase.Atoms, site: numpy.ndarray, unit_normal: numpy.ndarray, interstitial_gap: float = 0.1)

      Get the scaled normal that gives a proximate configuration without atomic
      overlap by:
          1. Projecting the adsorbate and surface atoms onto the surface plane.
          2. Identify all adsorbate atom - surface atom combinations for which
              an itersection when translating along the normal would occur.
              This is where the distance between the projected points is less than
              r_surface_atom + r_adsorbate_atom
          3. Explicitly solve for the scaled normal at which the distance between
              surface atom and adsorbate atom = r_surface_atom + r_adsorbate_atom +
              interstitial_gap. This exploits the superposition of vectors and the
              distance formula, so it requires root finding.

      Assumes that the adsorbate's binding atom or center-of-mass (depending
      on mode) is already placed at the site.

      :param adsorbate_c: A copy of the adsorbate with coordinates at the site
      :type adsorbate_c: ase.Atoms
      :param slab_c: A copy of the slab
      :type slab_c: ase.Atoms
      :param site: the coordinate of the site
      :type site: np.ndarray
      :param adsorbate_atoms: the translated adsorbate
      :type adsorbate_atoms: ase.Atoms
      :param unit_normal: the unit vector normal to the surface
      :type unit_normal: np.ndarray
      :param interstitial_gap: the desired distance between the covalent radii of the
                               closest surface and adsorbate atom
      :type interstitial_gap: float

      :returns: the magnitude of the normal vector for placement
      :rtype: (float)



   .. py:method:: _find_combos_to_check(adsorbate_c2: ase.Atoms, slab_c2: ase.Atoms, unit_normal: numpy.ndarray, interstitial_gap: float)

      Find the pairs of surface and adsorbate atoms that would have an intersection event
      while traversing the normal vector. For each pair, return pertanent information for
      finding the point of intersection.
      :param adsorbate_c2: A copy of the adsorbate with coordinates at the centered site
      :type adsorbate_c2: ase.Atoms
      :param slab_c2: A copy of the slab with atoms wrapped s.t. things are centered
                      about the site
      :type slab_c2: ase.Atoms
      :param unit_normal: the unit vector normal to the surface
      :type unit_normal: np.ndarray
      :param interstitial_gap: the desired distance between the covalent radii of the
                               closest surface and adsorbate atom
      :type interstitial_gap: float

      :returns:

                each entry in the list corresponds to one pair to check. With the
                    following information:
                        [(adsorbate_idx, slab_idx), r_adsorbate_atom + r_slab_atom, slab_atom_position]
      :rtype: (list[lists])



   .. py:method:: _get_projected_points(adsorbate_c2: ase.Atoms, slab_c2: ase.Atoms, unit_normal: numpy.ndarray)

      Find the x and y coordinates of each atom projected onto the surface plane.
      :param adsorbate_c2: A copy of the adsorbate with coordinates at the centered site
      :type adsorbate_c2: ase.Atoms
      :param slab_c2: A copy of the slab with atoms wrapped s.t. things are centered
                      about the site
      :type slab_c2: ase.Atoms
      :param unit_normal: the unit vector normal to the surface
      :type unit_normal: np.ndarray

      :returns: {"ads": [[x1, y1], [x2, y2], ...], "slab": [[x1, y1], [x2, y2], ...],}
      :rtype: (dict)



   .. py:method:: get_metadata_dict(ind)

      Returns a dict containing the atoms object and metadata for
      one specified config, used for writing to files.



.. py:class:: Bulk(bulk_atoms: ase.Atoms = None, bulk_id_from_db: int | None = None, bulk_src_id_from_db: str | None = None, bulk_db_path: str = BULK_PKL_PATH, bulk_db: list[dict[str, Any]] | None = None)

   Initializes a bulk object in one of 4 ways:
   - Directly pass in an ase.Atoms object.
   - Pass in index of bulk to select from bulk database.
   - Pass in the src_id of the bulk to select from the bulk database.
   - Randomly sample a bulk from bulk database if no other option is passed.

   :param bulk_atoms: Bulk structure.
   :type bulk_atoms: ase.Atoms
   :param bulk_id_from_db: Index of bulk in database pkl to select.
   :type bulk_id_from_db: int
   :param bulk_src_id_from_db: Src id of bulk to select (e.g. "mp-30").
   :type bulk_src_id_from_db: int
   :param bulk_db_path: Path to bulk database.
   :type bulk_db_path: str
   :param bulk_db: Already-loaded database.
   :type bulk_db: List[Dict[str, Any]]


   .. py:attribute:: bulk_id_from_db


   .. py:attribute:: bulk_db_path


   .. py:method:: _get_bulk_from_random(bulk_db)


   .. py:method:: set_source_dataset_id(src_id: str)


   .. py:method:: set_bulk_id_from_db(bulk_id_from_db: int)


   .. py:method:: get_slabs(max_miller=2, precomputed_slabs_dir=None)

      Returns a list of possible slabs for this bulk instance.



   .. py:method:: __len__()


   .. py:method:: __str__()


   .. py:method:: __repr__()


   .. py:method:: __eq__(other) -> bool


.. py:class:: InterfaceConfig(slab: fairchem.data.oc.core.slab.Slab, adsorbates: list[fairchem.data.oc.core.adsorbate.Adsorbate], solvent: fairchem.data.oc.core.solvent.Solvent, ions: list[fairchem.data.oc.core.ion.Ion] | None = None, num_sites: int = 100, num_configurations: int = 1, interstitial_gap: float = 0.1, vacuum_size: int = 15, solvent_interstitial_gap: float = 2, solvent_depth: float = 8, pbc_shift: float = 0.0, packmol_tolerance: float = 2, mode: str = 'random_site_heuristic_placement')

   Bases: :py:obj:`fairchem.data.oc.core.multi_adsorbate_slab_config.MultipleAdsorbateSlabConfig`


   Class to represent a solvent, adsorbate, slab, ion config. This class only
   returns a fixed combination of adsorbates placed on the surface. Solvent
   placement is performed by packmol
   (https://m3g.github.io/packmol/userguide.shtml), with the number of solvent
   molecules controlled by its corresponding density. Ion placement is random
   within the desired volume.

   :param slab: Slab object.
   :type slab: Slab
   :param adsorbates: List of adsorbate objects to place on the slab.
   :type adsorbates: List[Adsorbate]
   :param solvent: Solvent object
   :type solvent: Solvent
   :param ions: List of ion objects to place
   :type ions: List[Ion] = []
   :param num_sites: Number of sites to sample.
   :type num_sites: int
   :param num_configurations: Number of configurations to generate per slab+adsorbate(s) combination.
                              This corresponds to selecting different site combinations to place
                              the adsorbates on.
   :type num_configurations: int
   :param interstitial_gap: Minimum distance, in Angstroms, between adsorbate and slab atoms as
                            well as the inter-adsorbate distance.
   :type interstitial_gap: float
   :param vacuum_size: Size of vacuum layer to add to both ends of the resulting atoms object.
   :type vacuum_size: int
   :param solvent_interstitial_gap: Minimum distance, in Angstroms, between the solvent environment and the
                                    adsorbate-slab environment.
   :type solvent_interstitial_gap: float
   :param solvent_depth: Volume depth to be used to pack solvents inside.
   :type solvent_depth: float
   :param pbc_shift: Cushion to add to the packmol volume to avoid overlapping atoms over pbc.
   :type pbc_shift: float
   :param packmol_tolerance: Packmol minimum distance to impose between molecules.
   :type packmol_tolerance: float
   :param mode: "random", "heuristic", or "random_site_heuristic_placement".
                This affects surface site sampling and adsorbate placement on each site.

                In "random", we do a Delaunay triangulation of the surface atoms, then
                sample sites uniformly at random within each triangle. When placing the
                adsorbate, we randomly rotate it along xyz, and place it such that the
                center of mass is at the site.

                In "heuristic", we use Pymatgen's AdsorbateSiteFinder to find the most
                energetically favorable sites, i.e., ontop, bridge, or hollow sites.
                When placing the adsorbate, we randomly rotate it along z with only
                slight rotation along x and y, and place it such that the binding atom
                is at the site.

                In "random_site_heuristic_placement", we do a Delaunay triangulation of
                the surface atoms, then sample sites uniformly at random within each
                triangle. When placing the adsorbate, we randomly rotate it along z with
                only slight rotation along x and y, and place it such that the binding
                atom is at the site.

                In all cases, the adsorbate is placed at the closest position of no
                overlap with the slab plus `interstitial_gap` along the surface normal.
   :type mode: str


   .. py:attribute:: solvent


   .. py:attribute:: ions


   .. py:attribute:: vacuum_size


   .. py:attribute:: solvent_depth


   .. py:attribute:: solvent_interstitial_gap


   .. py:attribute:: pbc_shift


   .. py:attribute:: packmol_tolerance


   .. py:attribute:: n_mol_per_volume


   .. py:method:: create_interface_on_sites(atoms_list: list[ase.Atoms], metadata_list: list[dict])

      Given adsorbate+slab configurations generated from
      (Multi)AdsorbateSlabConfig and its corresponding metadata, create the
      solvent/ion interface on top of the provided atoms objects.



   .. py:method:: create_packmol_atoms(geometry: fairchem.data.oc.utils.geometry.Geometry, n_solvent_mols: int)

      Pack solvent molecules in a provided unit cell volume. Packmol is used
      to randomly pack solvent molecules in the desired volume.

      :param geometry: Geometry object corresponding to the desired cell.
      :type geometry: Geometry
      :param n_solvent_mols: Number of solvent molecules to pack in the volume.
      :type n_solvent_mols: int



   .. py:method:: run_packmol(packmol_input: str)

      Run packmol.



   .. py:method:: randomize_coords(atoms: ase.Atoms)

      Randomly place the atoms in its unit cell.



.. py:class:: Ion(ion_atoms: ase.Atoms = None, ion_id_from_db: int | None = None, ion_db_path: str = ION_PKL_PATH)

   Initializes an ion object in one of 2 ways:
   - Directly pass in an ase.Atoms object.
   - Pass in index of ion to select from ion database.

   :param ion_atoms: ion structure.
   :type ion_atoms: ase.Atoms
   :param ion_id_from_db: Index of ion to select.
   :type ion_id_from_db: int
   :param ion_db_path: Path to ion database.
   :type ion_db_path: str


   .. py:attribute:: ion_id_from_db


   .. py:attribute:: ion_db_path


   .. py:method:: __len__()


   .. py:method:: __str__()


   .. py:method:: _load_ion(ion: dict) -> None

      Saves the fields from an ion stored in a database. Fields added
      after the first revision are conditionally added for backwards
      compatibility with older database files.



   .. py:method:: get_ion_concentration(volume)

      Compute the ion concentration units of M, given a volume in units of
      Angstrom^3.



.. py:class:: MultipleAdsorbateSlabConfig(slab: fairchem.data.oc.core.slab.Slab, adsorbates: list[fairchem.data.oc.core.adsorbate.Adsorbate], num_sites: int = 100, num_configurations: int = 1, interstitial_gap: float = 0.1, mode: str = 'random_site_heuristic_placement')

   Bases: :py:obj:`fairchem.data.oc.core.adsorbate_slab_config.AdsorbateSlabConfig`


   Class to represent a slab with multiple adsorbates on it. This class only
   returns a fixed combination of adsorbates placed on the surface. Unlike
   AdsorbateSlabConfig which enumerates all possible adsorbate placements, this
   problem gets combinatorially large.

   :param slab: Slab object.
   :type slab: Slab
   :param adsorbates: List of adsorbate objects to place on the slab.
   :type adsorbates: List[Adsorbate]
   :param num_sites: Number of sites to sample.
   :type num_sites: int
   :param num_configurations: Number of configurations to generate per slab+adsorbate(s) combination.
                              This corresponds to selecting different site combinations to place
                              the adsorbates on.
   :type num_configurations: int
   :param interstitial_gap: Minimum distance, in Angstroms, between adsorbate and slab atoms as
                            well as the inter-adsorbate distance.
   :type interstitial_gap: float
   :param mode: "random", "heuristic", or "random_site_heuristic_placement".
                This affects surface site sampling and adsorbate placement on each site.

                In "random", we do a Delaunay triangulation of the surface atoms, then
                sample sites uniformly at random within each triangle. When placing the
                adsorbate, we randomly rotate it along xyz, and place it such that the
                center of mass is at the site.

                In "heuristic", we use Pymatgen's AdsorbateSiteFinder to find the most
                energetically favorable sites, i.e., ontop, bridge, or hollow sites.
                When placing the adsorbate, we randomly rotate it along z with only
                slight rotation along x and y, and place it such that the binding atom
                is at the site.

                In "random_site_heuristic_placement", we do a Delaunay triangulation of
                the surface atoms, then sample sites uniformly at random within each
                triangle. When placing the adsorbate, we randomly rotate it along z with
                only slight rotation along x and y, and place it such that the binding
                atom is at the site.

                In all cases, the adsorbate is placed at the closest position of no
                overlap with the slab plus `interstitial_gap` along the surface normal.
   :type mode: str


   .. py:attribute:: slab


   .. py:attribute:: adsorbates


   .. py:attribute:: num_sites


   .. py:attribute:: interstitial_gap


   .. py:attribute:: mode


   .. py:attribute:: sites


   .. py:method:: place_adsorbates_on_sites(sites: list, num_configurations: int = 1, interstitial_gap: float = 0.1)

      Place the adsorbate at the given binding sites.

      This method generates a fixed number of configurations where sites are
      selected to ensure that adsorbate binding indices are at least a fair
      distance away from each other (covalent radii + interstitial gap).
      While this helps prevent adsorbate overlap it does not gaurantee it
      since non-binding adsorbate atoms can overlap if the right combination
      of angles is sampled.



   .. py:method:: get_metadata_dict(ind)

      Returns a dict containing the atoms object and metadata for
      one specified config, used for writing to files.



.. py:class:: Slab(bulk=None, slab_atoms: ase.Atoms = None, millers: tuple | None = None, shift: float | None = None, top: bool | None = None, oriented_bulk: pymatgen.core.structure.Structure = None, min_ab: float = 8.0)

   Initializes a slab object, i.e. a particular slab tiled along xyz, in
   one of 2 ways:
   - Pass in a Bulk object and a slab 5-tuple containing
   (atoms, miller, shift, top, oriented bulk).
   - Pass in a Bulk object and randomly sample a slab.

   :param bulk: Corresponding Bulk object.
   :type bulk: Bulk
   :param slab_atoms: Slab atoms, tiled and tagged
   :type slab_atoms: ase.Atoms
   :param millers: Miller indices of slab.
   :type millers: tuple
   :param shift: Shift of slab.
   :type shift: float
   :param top: Whether slab is top or bottom.
   :type top: bool
   :param min_ab: To confirm that the tiled structure spans this distance
   :type min_ab: float


   .. py:attribute:: bulk


   .. py:attribute:: atoms


   .. py:attribute:: millers


   .. py:attribute:: shift


   .. py:attribute:: top


   .. py:attribute:: oriented_bulk


   .. py:method:: from_bulk_get_random_slab(bulk=None, max_miller=2, min_ab=8.0, save_path=None)
      :classmethod:



   .. py:method:: from_bulk_get_specific_millers(specific_millers, bulk=None, min_ab=8.0, save_path=None)
      :classmethod:



   .. py:method:: from_bulk_get_all_slabs(bulk=None, max_miller=2, min_ab=8.0, save_path=None)
      :classmethod:



   .. py:method:: from_precomputed_slabs_pkl(bulk=None, precomputed_slabs_pkl=None, max_miller=2, min_ab=8.0)
      :classmethod:



   .. py:method:: from_atoms(atoms: ase.Atoms = None, bulk=None, **kwargs)
      :classmethod:



   .. py:method:: has_surface_tagged()


   .. py:method:: get_metadata_dict()


   .. py:method:: __len__()


   .. py:method:: __str__()


   .. py:method:: __repr__()


   .. py:method:: __eq__(other)


.. py:class:: Solvent(solvent_atoms: ase.Atoms = None, solvent_id_from_db: int | None = None, solvent_db_path: str | None = SOLVENT_PKL_PATH, solvent_density: float | None = None)

   Initializes a solvent object in one of 2 ways:
   - Directly pass in an ase.Atoms object.
   - Pass in index of solvent to select from solvent database.

   :param solvent_atoms: Solvent molecule
   :type solvent_atoms: ase.Atoms
   :param solvent_id_from_db: Index of solvent to select.
   :type solvent_id_from_db: int
   :param solvent_db_path: Path to solvent database.
   :type solvent_db_path: str
   :param solvent_density: Desired solvent density to use. If not specified, the default is used
                           from the solvent databases.
   :type solvent_density: float


   .. py:attribute:: solvent_id_from_db


   .. py:attribute:: solvent_db_path


   .. py:attribute:: solvent_density


   .. py:attribute:: molar_mass


   .. py:method:: __len__()


   .. py:method:: __str__()


   .. py:method:: _load_solvent(solvent: dict) -> None

      Saves the fields from an adsorbate stored in a database. Fields added
      after the first revision are conditionally added for backwards
      compatibility with older database files.



   .. py:property:: molecules_per_volume

      Convert the solvent density in g/cm3 to the number of molecules per
      angstrom cubed of volume.


