:py:mod:`applications.CatTSunami.ocpneb.core.autoframe`
=======================================================

.. py:module:: applications.CatTSunami.ocpneb.core.autoframe

.. autoapi-nested-parse::

   Home of the AutoFrame classes which facillitate the generation of initial
   and final frames for NEB calculations.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   applications.CatTSunami.ocpneb.core.autoframe.AutoFrame
   applications.CatTSunami.ocpneb.core.autoframe.AutoFrameDissociation
   applications.CatTSunami.ocpneb.core.autoframe.AutoFrameTransfer
   applications.CatTSunami.ocpneb.core.autoframe.AutoFrameDesorption



Functions
~~~~~~~~~

.. autoapisummary::

   applications.CatTSunami.ocpneb.core.autoframe.interpolate_and_correct_frames
   applications.CatTSunami.ocpneb.core.autoframe.get_shortest_path
   applications.CatTSunami.ocpneb.core.autoframe.traverse_adsorbate_transfer
   applications.CatTSunami.ocpneb.core.autoframe.traverse_adsorbate_dissociation
   applications.CatTSunami.ocpneb.core.autoframe.traverse_adsorbate_desorption
   applications.CatTSunami.ocpneb.core.autoframe.get_product2_idx
   applications.CatTSunami.ocpneb.core.autoframe.traverse_adsorbate_general
   applications.CatTSunami.ocpneb.core.autoframe.unwrap_atoms
   applications.CatTSunami.ocpneb.core.autoframe.interpolate
   applications.CatTSunami.ocpneb.core.autoframe.is_edge_list_respected
   applications.CatTSunami.ocpneb.core.autoframe.reorder_edge_list
   applications.CatTSunami.ocpneb.core.autoframe.is_adsorbate_adsorbed



.. py:class:: AutoFrame


   Base class to hold functions that are shared across the reaction types.

   .. py:method:: reorder_adsorbate(frame: ase.Atoms, idx_mapping: dict)

      Given the adsorbate mapping, reorder the adsorbate atoms in the final frame so that
      they match the initial frame to facillitate proper interpolation.

      :param frame: the atoms object for which the adsorbate will be reordered
      :type frame: ase.Atoms
      :param idx_mapping: the index mapping to reorder things
      :type idx_mapping: dict

      :returns: the reordered adsorbate-slab configuration
      :rtype: ase.Atoms


   .. py:method:: only_keep_unique_systems(systems, energies)

      Remove duplicate systems from `systems` and `energies`.

      :param systems: the systems to remove duplicates from
      :type systems: list[ase.Atoms]
      :param energies: the energies to remove duplicates from
      :type energies: list[float]

      :returns: the systems with duplicates removed
                list[float]: the energies with duplicates removed
      :rtype: list[ase.Atoms]


   .. py:method:: get_most_proximate_symmetric_group(initial: ase.Atoms, frame: ase.Atoms)

      For cases where the adsorbate has symmetry and the leaving group could be different
      atoms / sets of atoms, determine which one make the most sense given the geometry of
      the initial and final frames. This is done by minimizing the total distance traveled
      by all atoms from initial to final frame.

      :param initial: the initial adsorbate-surface configuration
      :type initial: ase.Atoms
      :param frame: the final adsorbate-surface configuration being considered.
      :type frame: ase.Atoms

      :returns: the mapping to be used which specifies the most apt leaving group
                int: the index of the mapping to be used
      :rtype: dict


   .. py:method:: are_all_adsorbate_atoms_overlapping(adsorbate1: ase.Atoms, adsorbate2: ase.Atoms)

      Test to see if all the adsorbate atoms are intersecting to find unique structures.
      Systems where they are overlapping are considered the same.

      :param adsorbate1: just the adsorbate atoms of a structure that is being
                         compared
      :type adsorbate1: ase.Atoms
      :param adsorbate2: just the adsorbate atoms of the other structure that
                         is being compared
      :type adsorbate2: ase.Atoms

      :returns:

                True if all adsorbate atoms are overlapping (structure is a match)
                    False if one or more of the adsorbate atoms do not overlap
      :rtype: (bool)



.. py:class:: AutoFrameDissociation(reaction: ocpneb.core.Reaction, reactant_system: ase.Atoms, product1_systems: list, product1_energies: list, product2_systems: list, product2_energies: list, r_product1_max: float = None, r_product2_max: float = None, r_product2_min: float = None)


   Bases: :py:obj:`AutoFrame`

   Base class to hold functions that are shared across the reaction types.

   .. py:method:: get_neb_frames(calculator, n_frames: int = 5, n_pdt1_sites: int = 5, n_pdt2_sites: int = 5, fmax: float = 0.05, steps: int = 200)

      Propose final frames for NEB calculations. Perform a relaxation on the final
      frame using the calculator provided. Interpolate between the initial
      and final frames for a proposed reaction trajectory. Correct the trajectory if
      there is any atomic overlap.

      :param calculator: an ase compatible calculator to be used to relax the final frame.
      :param n_frames: the number of frames per reaction trajectory
      :type n_frames: int
      :param n_pdt1_sites: The number of product 1 sites to consider
      :type n_pdt1_sites: int
      :param n_pdt2_sites: The number of product 2 sites to consider. Note this is
                           multiplicative with `n_pdt1_sites` (i.e. if `n_pdt1_sites` = 2 and
                           `n_pdt2_sites` = 3 then a total of 6 final frames will be proposed)
      :type n_pdt2_sites: int
      :param fmax: force convergence criterion for final frame optimization
      :type fmax: float
      :param steps: step number termination criterion for final frame optimization
      :type steps: int

      :returns: the initial reaction coordinates
      :rtype: list[lists]


   .. py:method:: get_best_sites_for_product1(n_sites: int = 5)

      Wrapper to find product 1 placements to be considered for the final frame
      of the NEB.

      :param n_sites: The number of sites for product 1 to consider. Notice this is
                      multiplicative with product 2 sites (i.e. if 2 is specified here and 3 there)
                      then a total of 6 initial and final frames will be considered.
      :type n_sites: int

      :returns:

                the lowest energy, proximate placements of product
                    1 to be used in the final NEB frames
      :rtype: (list[ase.Atoms])


   .. py:method:: get_best_unique_sites_for_product2(product1: ase.Atoms, n_sites: int = 5)

      Wrapper to find product 2 placements to be considered for the final frame
      of the NEB.

      :param product1: The atoms object of the product 1 placement that will be
                       considered in this function to search for product 1 + product 2 combinations
                       for the final frame.
      :type product1: ase.Atoms
      :param n_sites: The number of sites for product 1 to consider. Notice this is
                      multiplicative with product 2 sites (i.e. if 2 is specified here and 3 there)
                      then a total of 6 initial and final frames will be considered.
      :type n_sites: int

      :returns:

                the lowest energy, proximate placements of product
                    2 to be used in the final NEB frames
      :rtype: (list[ase.Atoms])


   .. py:method:: get_sites_within_r(center_coordinate: numpy.ndarray, all_systems: list, all_system_energies: list, all_systems_binding_idx: int, allowed_radius_max: float, allowed_radius_min: float, n_sites: int = 5)

      Get the n lowest energy, sites of the systems within r. For now n is
      5 or < 5 if there are fewer than 5 unique sites within r.

      :param center_coordinate: the coordinate about which r should be
                                centered.
      :type center_coordinate: np.ndarray
      :param all_systems: the list of all systems to be assessed for their
                          uniqueness and proximity to the center coordinate.
      :type all_systems: list
      :param all_systems_binding_idx: the idx of the adsorbate atom that is
                                      bound in `all_systems`
      :type all_systems_binding_idx: int
      :param allowed_radius_max: the outer radius about `center_coordinate`
                                 in which the adsorbate must lie to be considered.
      :type allowed_radius_max: float
      :param allowed_radius_min: the inner radius about `center_coordinate`
                                 which the adsorbate must lie outside of to be considered.
      :type allowed_radius_min: float
      :param n_sites: the number of unique sites in r that will be chosen.
      :type n_sites: int

      :returns: list of systems identified as candidates.
      :rtype: (list[ase.Atoms])



.. py:class:: AutoFrameTransfer(reaction: ocpneb.core.Reaction, reactant1_systems: list, reactant2_systems: list, reactant1_energies: list, reactant2_energies: list, product1_systems: list, product1_energies: list, product2_systems: list, product2_energies: list, r_traverse_max: float, r_react_max: float, r_react_min: float)


   Bases: :py:obj:`AutoFrame`

   Base class to hold functions that are shared across the reaction types.

   .. py:method:: get_neb_frames(calculator, n_frames: int = 10, n_initial_frames: int = 5, n_final_frames_per_initial: int = 5, fmax: float = 0.05, steps: int = 200)

      Propose final frames for NEB calculations. Perform a relaxation on the final
      frame using the calculator provided. Linearly interpolate between the initial
      and final frames for a proposed reaction trajectory. Correct the trajectory if
      there is any atomic overlap.

      :param calculator: an ase compatible calculator to be used to relax the initial and
                         final frames.
      :param n_frames: the number of frames per reaction trajectory
      :type n_frames: int
      :param n_initial_frames: The number of initial frames to consider
      :type n_initial_frames: int
      :param n_final_frames_per_initial: The number of final frames per inital frame to consider
      :type n_final_frames_per_initial: int
      :param fmax: force convergence criterion for final frame optimization
      :type fmax: float
      :param steps: step number termination criterion for final frame optimization
      :type steps: int

      :returns: the initial reaction coordinates
      :rtype: list[lists]


   .. py:method:: get_system_pairs_initial()

      Get the initial frames for the NEB. This is done by finding the closest
      pair of systems from `systems1` and `systems2` for which the interstitial distance
      between all adsorbate atoms is less than `rmax` and greater than `rmin`.

      :returns: the initial frames for the NEB
                list[float]: the pseudo energies of the initial frames (i.e just the sum of the
                    individual adsorption energies)
      :rtype: list[ase.Atoms]


   .. py:method:: get_system_pairs_final(system1_coord, system2_coord)

      Get the final frames for the NEB. This is done by finding the closest
      pair of systems from `systems1` and `systems2` for which the distance
      traversed by the adsorbate from the initial frame to the final frame is
      less than `rmax` and the minimum interstitial distance between the two
      products in greater than `rmin`.

      :returns: the initial frames for the NEB
                list[float]: the pseudo energies of the initial frames
      :rtype: list[ase.Atoms]



.. py:class:: AutoFrameDesorption(reaction: ocpneb.core.Reaction, reactant_systems: list, reactant_energies: list, z_desorption: float)


   Bases: :py:obj:`AutoFrame`

   Base class to hold functions that are shared across the reaction types.

   .. py:method:: get_neb_frames(calculator, n_frames: int = 5, n_systems: int = 5, fmax: float = 0.05, steps: int = 200)

      Propose final frames for NEB calculations. Perform a relaxation on the final
      frame using the calculator provided. Linearly interpolate between the initial
      and final frames for a proposed reaction trajectory. Correct the trajectory if
      there is any atomic overlap.

      :param calculator: an ase compatible calculator to be used to relax the final frame.
      :param n_frames: the number of frames per reaction trajectory
      :type n_frames: int
      :param n_pdt1_sites: The number of product 1 sites to consider
      :type n_pdt1_sites: int
      :param n_pdt2_sites: The number of product 2 sites to consider. Note this is
                           multiplicative with `n_pdt1_sites` (i.e. if `n_pdt1_sites` = 2 and
                           `n_pdt2_sites` = 3 then a total of 6 final frames will be proposed)
      :type n_pdt2_sites: int
      :param fmax: force convergence criterion for final frame optimization
      :type fmax: float
      :param steps: step number termination criterion for final frame optimization
      :type steps: int

      :returns: the initial reaction coordinates
      :rtype: list[lists]



.. py:function:: interpolate_and_correct_frames(initial: ase.Atoms, final: ase.Atoms, n_frames: int, reaction: ocpneb.core.Reaction, map_idx: int)

   Given the initial and final frames, perform the following:
   (1) Unwrap the final frame if it is wrapped around the cell
   (2) Interpolate between the initial and final frames

   :param initial: the initial frame of the NEB
   :type initial: ase.Atoms
   :param final: the proposed final frame of the NEB
   :type final: ase.Atoms
   :param n_frames: The desired number of frames for the NEB (not including initial and final)
   :type n_frames: int
   :param reaction: the reaction object which provides pertinent info
   :type reaction: ocpneb.core.Reaction
   :param map_idx: the index of the mapping to use for the final frame
   :type map_idx: int


.. py:function:: get_shortest_path(initial: ase.Atoms, final: ase.Atoms)

   Find the shortest path for all atoms about pbc and reorient the final frame so the
   atoms align with this shortest path. This allows us to perform a linear interpolation
   that does not interpolate jumps across pbc.

   :param initial: the initial frame of the NEB
   :type initial: ase.Atoms
   :param final: the proposed final frame of the NEB to be corrected
   :type final: ase.Atoms

   :returns: the corrected final frame
             (ase.Atoms): the initial frame tiled (3,3,1), which is used it later steps
             (ase.Atoms): the final frame tiled (3,3,1), which is used it later steps
   :rtype: (ase.Atoms)


.. py:function:: traverse_adsorbate_transfer(reaction: ocpneb.core.Reaction, initial: ase.Atoms, final: ase.Atoms, initial_tiled: ase.Atoms, final_tiled: ase.Atoms, edge_list_final: list)

   Traverse reactant 1, reactant 2, product 1 and product 2 in a depth first search of
   the bond graph. Unwrap the atoms to minimize the distance over the bonds. This ensures
   that when we perform the linear interpolation, the adsorbate moves as a single moity
   and avoids accidental bond breaking events over pbc.

   :param reaction: the reaction object which provides pertinent info
   :type reaction: ocpneb.core.Reaction
   :param initial: the initial frame of the NEB
   :type initial: ase.Atoms
   :param final: the proposed final frame of the NEB to be corrected
   :type final: ase.Atoms
   :param initial_tiled: the initial frame tiled (3,3,1)
   :type initial_tiled: ase.Atoms
   :param final_tiled: the final frame tiled (3,3,1)
   :type final_tiled: ase.Atoms
   :param edge_list_final: the edge list of the final frame corrected with mapping
                           idx changes
   :type edge_list_final: list

   :returns: the corrected initial frame
             (ase.Atoms): the corrected final frame
   :rtype: (ase.Atoms)


.. py:function:: traverse_adsorbate_dissociation(reaction: ocpneb.core.Reaction, initial: ase.Atoms, final: ase.Atoms, initial_tiled: ase.Atoms, final_tiled: ase.Atoms, edge_list_final: int)

   Traverse reactant 1, product 1 and product 2 in a depth first search of
   the bond graph. Unwrap the atoms to minimize the distance over the bonds. This ensures
   that when we perform the linear interpolation, the adsorbate moves as a single moity
   and avoids accidental bond breaking events over pbc.

   :param reaction: the reaction object which provides pertinent info
   :type reaction: ocpneb.core.Reaction
   :param initial: the initial frame of the NEB
   :type initial: ase.Atoms
   :param final: the proposed final frame of the NEB to be corrected
   :type final: ase.Atoms
   :param initial_tiled: the initial frame tiled (3,3,1)
   :type initial_tiled: ase.Atoms
   :param final_tiled: the final frame tiled (3,3,1)
   :type final_tiled: ase.Atoms
   :param edge_list_final: the edge list of the final frame corrected with mapping
                           idx changes
   :type edge_list_final: list

   :returns: the corrected initial frame
             (ase.Atoms): the corrected final frame
   :rtype: (ase.Atoms)


.. py:function:: traverse_adsorbate_desorption(reaction: ocpneb.core.Reaction, initial: ase.Atoms, final: ase.Atoms, initial_tiled: ase.Atoms, final_tiled: ase.Atoms)

   Traverse reactant 1 and  product 1 in a depth first search of
   the bond graph. Unwrap the atoms to minimize the distance over the bonds. This ensures
   that when we perform the linear interpolation, the adsorbate moves as a single moity
   and avoids accidental bond breaking events over pbc.

   :param reaction: the reaction object which provides pertinent info
   :type reaction: ocpneb.core.Reaction
   :param initial: the initial frame of the NEB
   :type initial: ase.Atoms
   :param final: the proposed final frame of the NEB to be corrected
   :type final: ase.Atoms
   :param initial_tiled: the initial frame tiled (3,3,1)
   :type initial_tiled: ase.Atoms
   :param final_tiled: the final frame tiled (3,3,1)
   :type final_tiled: ase.Atoms
   :param edge_list_final: the edge list of the final frame corrected with mapping
                           idx changes
   :type edge_list_final: list

   :returns: the corrected initial frame
             (ase.Atoms): the corrected final frame
   :rtype: (ase.Atoms)


.. py:function:: get_product2_idx(reaction: ocpneb.core.Reaction, edge_list_final: list, traversal_rxt1_final: list)

   For dissociation only. Use the information about the initial edge list and final edge
   list to determine which atom in product 2 lost a bond in the reaction and use this
   as the binding index for traversal in `traverse_adsorbate_dissociation`.

   :param reaction: the reaction object which provides pertinent info
   :type reaction: ocpneb.core.Reaction
   :param edge_list_final: the edge list of the final frame corrected with mapping
                           idx changes
   :type edge_list_final: list
   :param traversal_rxt1_final: the traversal of reactant 1 for the final frame
   :type traversal_rxt1_final: list

   :returns: the binding index of product 2
   :rtype: (int)


.. py:function:: traverse_adsorbate_general(traversal_rxt, slab_len: int, starting_node_idx: int, equivalent_idx_factors: numpy.ndarray, frame: ase.Atoms, frame_tiled: ase.Atoms)

   Perform the traversal to reposition atoms so that the distance along bonds is
   minimized.

   :param traversal_rxt: the traversal of the adsorbate to be traversed. It is
                         the list of edges ordered by depth first search.
   :type traversal_rxt: list
   :param slab_len: the number of atoms in the slab
   :type slab_len: int
   :param starting_node_idx: the index of the atom to start the traversal from
   :type starting_node_idx: int
   :param equivalent_idx_factors: the values to add to the untiled index
                                  which gives equivalent indices (i.e. copies of that atom in the tiled system)
   :type equivalent_idx_factors: np.ndarray
   :param frame: the frame to be corrected
   :type frame: ase.Atoms
   :param frame_tiled: the tiled (3,3,1) version of the frame which will be
                       corrected
   :type frame_tiled: ase.Atoms

   :returns: the corrected frame
   :rtype: (ase.Atoms)


.. py:function:: unwrap_atoms(initial: ase.Atoms, final: ase.Atoms, reaction: ocpneb.core.Reaction, map_idx: int)

   Make corrections to the final frame so it is no longer wrapped around the cell,
   if it has jumpped over the pbc. Ensure that for each adsorbate moity, absolute bond distances
   for all edges that exist in the initial and final frames are minimize regardles of cell location.
   This enforces the traversal of the adsorbates happens along the same path, which is not
   necessarily the minimum distance path for each atom. Changes are made in place.

   :param initial: the initial atoms object to which the final atoms should
                   be proximate
   :type initial: ase.Atoms
   :param final: the final atoms object to be corrected
   :type final: ase.Atoms
   :param reaction: the reaction object which provides pertinent info
   :type reaction: ocpneb.core.Reaction
   :param map_idx: the index of the mapping to use for the final frame
   :type map_idx: int


.. py:function:: interpolate(initial_frame: ase.Atoms, final_frame: ase.Atoms, num_frames: int)

   Interpolate between the initial and final frames starting with a linear interpolation
   along the atom-wise vectors from initial to final. Then iteratively correct the
   positions so atomic overlap is avoided/ reduced. When iteratively updating, the
   positions of adjacent frames are considered to avoid large jumps in the trajectory.

   :param initial_frame: the initial frame which will be interpolated from
   :type initial_frame: ase.Atoms
   :param final_frame: the final frame which will be interpolated to
   :type final_frame: ase.Atoms
   :param num_frames: the number of frames to be interpolated between the initial
   :type num_frames: int

   :returns: the interpolated frames
   :rtype: (list[ase.Atoms])


.. py:function:: is_edge_list_respected(frame: ase.Atoms, edge_list: list)

   Check to see that the expected adsorbate-adsorbate edges are found and no additional
   edges exist between the adsorbate atoms.

   :param frame: the atoms object for which edges will be checked.
                 This must comply with ocp tagging conventions.
   :type frame: ase.Atoms
   :param edge_list: The expected edges
   :type edge_list: list[tuples]


.. py:function:: reorder_edge_list(edge_list: list, mapping: dict)

   For the final edge list, apply the mapping so the edges correspond to the correctly
   concatenated object.

   :param edge_list: the final edgelist
   :type edge_list: list[tuples]
   :param mapping: the mapping so the final atoms concatenated have indices that correctly map
                   to the initial atoms.


.. py:function:: is_adsorbate_adsorbed(adsorbate_slab_config: ase.Atoms)

   Check to see if the adsorbate is adsorbed on the surface.

   :param adsorbate_slab_config: the combined adsorbate and slab configuration
                                 with adsorbate atoms tagged as 2s and surface atoms tagged as 1s.
   :type adsorbate_slab_config: ase.Atoms

   :returns: True if the adsorbate is adsorbed, False otherwise.
   :rtype: (bool)


