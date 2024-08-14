data.oc.core.interface_config
=============================

.. py:module:: data.oc.core.interface_config


Classes
-------

.. autoapisummary::

   data.oc.core.interface_config.InterfaceConfig


Module Contents
---------------

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



