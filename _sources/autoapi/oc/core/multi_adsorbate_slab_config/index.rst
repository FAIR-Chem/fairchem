:py:mod:`oc.core.multi_adsorbate_slab_config`
=============================================

.. py:module:: oc.core.multi_adsorbate_slab_config


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   oc.core.multi_adsorbate_slab_config.MultipleAdsorbateSlabConfig



Functions
~~~~~~~~~

.. autoapisummary::

   oc.core.multi_adsorbate_slab_config.update_distance_map



.. py:class:: MultipleAdsorbateSlabConfig(slab: fairchem.data.oc.core.Slab, adsorbates: List[fairchem.data.oc.core.Adsorbate], num_sites: int = 100, num_configurations: int = 1, interstitial_gap: float = 0.1, mode: str = 'random_site_heuristic_placement')


   Bases: :py:obj:`fairchem.data.oc.core.AdsorbateSlabConfig`

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



.. py:function:: update_distance_map(prev_distance_map, site_idx, adsorbate, pseudo_atoms)

   Given a new site and the adsorbate we plan on placing there,
   update the distance mapping to reflect the new distances from sites to nearest adsorbates.
   We incorporate the covalent radii of the placed adsorbate binding atom in our distance
   calculation to prevent atom overlap.


