data.oc.core.adsorbate_slab_config
==================================

.. py:module:: data.oc.core.adsorbate_slab_config


Classes
-------

.. autoapisummary::

   data.oc.core.adsorbate_slab_config.AdsorbateSlabConfig


Functions
---------

.. autoapisummary::

   data.oc.core.adsorbate_slab_config.get_random_sites_on_triangle
   data.oc.core.adsorbate_slab_config.custom_tile_atoms
   data.oc.core.adsorbate_slab_config.get_interstitial_distances
   data.oc.core.adsorbate_slab_config.there_is_overlap


Module Contents
---------------

.. py:class:: AdsorbateSlabConfig(slab: fairchem.data.oc.core.Slab, adsorbate: fairchem.data.oc.core.Adsorbate, num_sites: int = 100, num_augmentations_per_site: int = 1, interstitial_gap: float = 0.1, mode: str = 'random')

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


   .. py:method:: get_binding_sites(num_sites: int)

      Returns up to `num_sites` sites given the surface atoms' positions.



   .. py:method:: place_adsorbate_on_site(adsorbate: fairchem.data.oc.core.Adsorbate, site: numpy.ndarray, interstitial_gap: float = 0.1)

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



.. py:function:: get_random_sites_on_triangle(vertices: numpy.ndarray, num_sites: int = 10)

   Sample `num_sites` random sites uniformly on a given 3D triangle.
   Following Sec. 4.2 from https://www.cs.princeton.edu/~funk/tog02.pdf.


.. py:function:: custom_tile_atoms(atoms: ase.Atoms)

   Tile the atoms so that the center tile has the indices and positions of the
   untiled structure.

   :param atoms: the atoms object to be tiled
   :type atoms: ase.Atoms

   :returns:

             the tiled atoms which has been repeated 3 times in
                 the x and y directions but maintains the original indices on the central
                 unit cell.
   :rtype: (ase.Atoms)


.. py:function:: get_interstitial_distances(adsorbate_slab_config: ase.Atoms)

   Check to see if there is any atomic overlap between surface atoms
   and adsorbate atoms.

   :param adsorbate_slab_configuration: an slab atoms object with an
                                        adsorbate placed
   :type adsorbate_slab_configuration: ase.Atoms

   :returns: True if there is atomic overlap, otherwise False
   :rtype: (bool)


.. py:function:: there_is_overlap(adsorbate_slab_config: ase.Atoms)

   Check to see if there is any atomic overlap between surface atoms
   and adsorbate atoms.

   :param adsorbate_slab_configuration: an slab atoms object with an
                                        adsorbate placed
   :type adsorbate_slab_configuration: ase.Atoms

   :returns: True if there is atomic overlap, otherwise False
   :rtype: (bool)


