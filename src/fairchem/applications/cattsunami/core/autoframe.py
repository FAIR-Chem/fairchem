"""
Home of the AutoFrame classes which facillitate the generation of initial
and final frames for NEB calculations.
"""

from __future__ import annotations

import copy
from copy import deepcopy
from itertools import combinations, product
from typing import TYPE_CHECKING

import ase
import networkx as nx
import numpy as np
import torch
from ase.data import atomic_numbers, covalent_radii
from ase.optimize import BFGS
from fairchem.data.oc.utils import DetectTrajAnomaly
from scipy.spatial.distance import euclidean

if TYPE_CHECKING:
    from fairchem.applications.cattsunami.core import Reaction


class AutoFrame:
    """
    Base class to hold functions that are shared across the reaction types.
    """

    def reorder_adsorbate(self, frame: ase.Atoms, idx_mapping: dict):
        """
        Given the adsorbate mapping, reorder the adsorbate atoms in the final frame so that
        they match the initial frame to facillitate proper interpolation.

        Args:
            frame (ase.Atoms): the atoms object for which the adsorbate will be reordered
            idx_mapping (dict): the index mapping to reorder things
        Returns:
            ase.Atoms: the reordered adsorbate-slab configuration
        """
        adsorbate = frame[
            [idx for idx, tag in enumerate(frame.get_tags()) if tag == 2]
        ].copy()
        slab = frame[
            [idx for idx, tag in enumerate(frame.get_tags()) if tag != 2]
        ].copy()
        fresh_indices = [idx_mapping[i] for i in range(len(adsorbate))]
        adsorbate_reordered = adsorbate[fresh_indices]
        return slab + adsorbate_reordered

    def only_keep_unique_systems(self, systems, energies):
        """
        Remove duplicate systems from `systems` and `energies`.

        Args:
            systems (list[ase.Atoms]): the systems to remove duplicates from
            energies (list[float]): the energies to remove duplicates from

        Returns:
            list[ase.Atoms]: the systems with duplicates removed
            list[float]: the energies with duplicates removed

        """
        # Order systems by energy so we keep the lowest energy match
        e_sort = np.argsort(energies)
        systems = [systems[idx] for idx in e_sort]
        energies = np.sort(energies)

        # Remove duplicate systems
        unique_systems = []
        unique_energies = []
        unique_system_atoms = []
        adsorbates_stripped_out = []

        # Just grab the adsorbate because it is all we care about
        for system in systems:
            adsorbates_stripped_out.append(
                system[[idx for idx, tag in enumerate(system.get_tags()) if tag == 2]]
            )

        # Iterate over the systems and see where there are matches (systems where every adsorbate atom is overlapping)
        for idx in range(len(systems)):
            if not any(
                self.are_all_adsorbate_atoms_overlapping(
                    adsorbates_stripped_out[unique_system],
                    adsorbates_stripped_out[idx],
                )
                for unique_system in unique_systems
            ):
                unique_systems.append(idx)
                unique_energies.append(energies[idx])
                unique_system_atoms.append(systems[idx])
        return unique_system_atoms, unique_energies

    def get_most_proximate_symmetric_group(self, initial: ase.Atoms, frame: ase.Atoms):
        """
        For cases where the adsorbate has symmetry and the leaving group could be different
        atoms / sets of atoms, determine which one make the most sense given the geometry of
        the initial and final frames. This is done by minimizing the total distance traveled
        by all atoms from initial to final frame.

        Args:
            initial (ase.Atoms): the initial adsorbate-surface configuration
            frame (ase.Atoms): the final adsorbate-surface configuration being considered.

        Returns:
            dict: the mapping to be used which specifies the most apt leaving group
            int: the index of the mapping to be used
        """
        distances = []
        adsorbate_idxs = [idx for idx, tag in enumerate(initial.get_tags()) if tag == 2]
        dummy_frame = frame.copy()
        for mapping in self.reaction.idx_mapping:
            distance_now = 0
            for idx, adsorbate_idx in enumerate(adsorbate_idxs):
                false_atoms = ase.Atoms(
                    "HH",
                    positions=[
                        initial.positions[adsorbate_idx],
                        frame.positions[adsorbate_idx + mapping[idx] - idx],
                    ],
                )
                dummy_frame += false_atoms
                distance_now += dummy_frame.get_distance(-1, -2, mic=True)
            distances.append(distance_now)
        best_map = np.argmin(distances)
        return self.reaction.idx_mapping[best_map], best_map

    def are_all_adsorbate_atoms_overlapping(
        self,
        adsorbate1: ase.Atoms,
        adsorbate2: ase.Atoms,
    ):
        """
        Test to see if all the adsorbate atoms are intersecting to find unique structures.
        Systems where they are overlapping are considered the same.

        Args:
            adsorbate1 (ase.Atoms): just the adsorbate atoms of a structure that is being
                compared
            adsorbate2 (ase.Atoms): just the adsorbate atoms of the other structure that
                is being compared

        Returns:
            (bool): True if all adsorbate atoms are overlapping (structure is a match)
                False if one or more of the adsorbate atoms do not overlap
        """
        bools = []
        assert np.allclose(adsorbate1.cell, adsorbate2.cell)
        elements = adsorbate1.get_chemical_symbols()
        dummy_atoms = adsorbate1.copy()
        dummy_atoms += adsorbate2.copy()
        for idx in range(len(adsorbate1)):
            distance = dummy_atoms.get_distance(idx, idx + len(adsorbate1), mic=True)
            bools.append(distance < covalent_radii[atomic_numbers[elements[idx]]])
        return all(bools)


class AutoFrameDissociation(AutoFrame):
    def __init__(
        self,
        reaction: Reaction,
        reactant_system: ase.Atoms,
        product1_systems: list,
        product1_energies: list,
        product2_systems: list,
        product2_energies: list,
        r_product1_max: float | None = None,
        r_product2_max: float | None = None,
        r_product2_min: float | None = None,
    ):
        """
        Initialize class to handle the automatic generation of NEB frames for dissociation.

        Args:
            reaction (ocp.core.Reaction): the reaction object which provides pertinent info
                about the reaction
            reactant_system (ase.Atoms): a relaxed atoms object of the reactant
                sitting on a slab in the configuration to be considered for NEBs.
            product1_systems (list[ase.Atoms]): a list of relaxed atoms objects of
                the product that contains the binding atom. A list of multiple relaxed
                adsorbate placements should be provided so that multiple possible NEBs
                can be created.
            product1_energies (list[float]): the energies of the systems included in
                `product1_systems`
            product2_systems (list[ase.Atoms]): a list of relaxed atoms objects of
                the product that does not contain the binding atom. A list of multiple relaxed
                adsorbate placements should be provided so that multiple possible NEBs
                can be created.
            product2_energies (list[float]): the energies of the systems included in
                `product2_systems`
            r_product1_max (float): the radius about the binding atom of `reactant_system`
                for which configurations of product 1 will be considered for the final
                frame of the NEB if inside. If None, the radius will be set to 2x the covalent
                radius of the largest surface atom.
            r_product2_max (float): the radius about the binding atom of `product1_systems`
                for which configurations of product 2 will be considered for the final
                frame of the NEB if inside. If None, the radius will be set to 3x the covalent
                radius of the largest surface atom.
            r_product2_min (float): the radius about the binding atom of `product1_systems`
                for which configurations of product 2 will be considered for the final
                frame of the NEB if outside. If None, the radius will be set to 3x the covalent
                radius of the binding atom.
        """

        # Store info to the class
        self.reaction = reaction
        self.product1_systems, self.product1_energies = self.only_keep_unique_systems(
            product1_systems, product1_energies
        )
        self.product2_systems, self.product2_energies = self.only_keep_unique_systems(
            product2_systems, product2_energies
        )
        self.reactant1_systems = [reactant_system]

        # Calculate r_product1/2 if not supplied and store to class
        if r_product1_max is not None and r_product2_max is not None:
            self.r_product1_max = r_product1_max
            self.r_product2_max = r_product2_max
        else:
            slab_idxs = self.reactant1_systems[0].get_tags() != 2
            slab = self.reactant1_systems[0].copy()
            slab = slab[slab_idxs]
            slab_els = np.unique(slab.get_chemical_symbols())
            slab_radii = [covalent_radii[atomic_numbers[el]] for el in slab_els]
            r_slab_max = max(slab_radii)

        if r_product1_max is None:
            self.r_product1_max = 2 * r_slab_max
        if r_product2_max is None:
            self.r_product2_max = 3 * r_slab_max

        if r_product2_min is not None:
            self.r_product2_min = r_product2_min
        else:
            product1_binding_element = product1_systems[0].get_chemical_symbols()[
                self.reaction.binding_atom_idx_product1 - len(self.reaction.product1)
            ]
            self.r_product2_min = covalent_radii[
                atomic_numbers[product1_binding_element]
            ]

        # Check that the reactant has the bond we are interested in breaking
        if not is_edge_list_respected(
            self.reactant1_systems[0], self.reaction.edge_list_initial
        ):
            raise ValueError(
                "The reactant is not fully connected. This is necessary for decent NEB placement."
            )

    def get_neb_frames(
        self,
        calculator,
        n_frames: int = 5,
        n_pdt1_sites: int = 5,
        n_pdt2_sites: int = 5,
        fmax: float = 0.05,
        steps: int = 200,
    ):
        """
        Propose final frames for NEB calculations. Perform a relaxation on the final
        frame using the calculator provided. Interpolate between the initial
        and final frames for a proposed reaction trajectory. Correct the trajectory if
        there is any atomic overlap.

        Args:
            calculator: an ase compatible calculator to be used to relax the final frame.
            n_frames (int): the number of frames per reaction trajectory
            n_pdt1_sites (int): The number of product 1 sites to consider
            n_pdt2_sites (int): The number of product 2 sites to consider. Note this is
                multiplicative with `n_pdt1_sites` (i.e. if `n_pdt1_sites` = 2 and
                `n_pdt2_sites` = 3 then a total of 6 final frames will be proposed)
            fmax (float): force convergence criterion for final frame optimization
            steps (int): step number termination criterion for final frame optimization

        Returns:
            list[lists]: the initial reaction coordinates

        """
        # Get proposed final frames
        final_frames = []
        product1_sites = self.get_best_sites_for_product1(n_pdt1_sites)
        for site in product1_sites:
            site_frames = self.get_best_unique_sites_for_product2(site, n_pdt2_sites)
            final_frames.extend(site_frames)

        # Relax final frames. Discard those that don't converge or are anomalies
        converged_final_frames = []
        slab_of_rxt = self.reactant1_systems[0][
            [
                idx
                for idx, tag in enumerate(self.reactant1_systems[0].get_tags())
                if tag != 2
            ]
        ]
        for frame in final_frames:
            initial = frame.copy()
            frame.calc = calculator
            opt = BFGS(frame)
            converged = opt.run(fmax=fmax, steps=steps)
            if converged:
                dt = DetectTrajAnomaly(initial, frame, initial.get_tags(), slab_of_rxt)
                edge_check = is_edge_list_respected(
                    frame, self.reaction.edge_list_final
                )
                if all(
                    [
                        not dt.is_adsorbate_intercalated(),
                        not dt.is_adsorbate_desorbed(),
                        not dt.has_surface_changed(),
                        edge_check,
                    ]
                ):
                    converged_final_frames.append(frame)

        # Correct the adsorbate index order of the final frame to match the initial.
        # Once corrected, interpolate between the intial and final state, to be returned
        neb_frame_sets = []
        map_idx_list = []

        reactant_system_c = self.reactant1_systems[0].copy()
        cell_center = deepcopy(
            np.dot(np.array([0.5, 0.5, 0.5]), reactant_system_c.cell)
        )
        slab_len = len(reactant_system_c) - list(reactant_system_c.get_tags()).count(2)
        site = deepcopy(
            reactant_system_c.positions[
                self.reaction.binding_atom_idx_reactant1 + slab_len
            ]
        )
        reactant_system_c.translate(cell_center - site)
        reactant_system_c.wrap()

        for frame in converged_final_frames:
            if len(self.reaction.idx_mapping) > 1:
                mapping_to_use, map_idx = self.get_most_proximate_symmetric_group(
                    initial, frame
                )
                reordered_converged_final_frame = self.reorder_adsorbate(
                    frame, mapping_to_use
                )
            else:
                map_idx = 0
                reordered_converged_final_frame = self.reorder_adsorbate(
                    frame, self.reaction.idx_mapping[map_idx]
                )
            reordered_converged_final_frame_c = reordered_converged_final_frame.copy()
            reordered_converged_final_frame_c.translate(cell_center - site)
            reordered_converged_final_frame_c.wrap()
            neb_frames = interpolate_and_correct_frames(
                reactant_system_c,
                reordered_converged_final_frame_c,
                n_frames,
                self.reaction,
                map_idx,
            )
            if len(neb_frames) > 0:
                neb_frame_sets.append(neb_frames)
                map_idx_list.append(map_idx)

        return neb_frame_sets, map_idx_list

    def get_best_sites_for_product1(self, n_sites: int = 5):
        """
        Wrapper to find product 1 placements to be considered for the final frame
        of the NEB.

        Args:
            n_sites (int): The number of sites for product 1 to consider. Notice this is
                multiplicative with product 2 sites (i.e. if 2 is specified here and 3 there)
                then a total of 6 initial and final frames will be considered.

        Returns:
            (list[ase.Atoms]): the lowest energy, proximate placements of product
                1 to be used in the final NEB frames
        """
        center_coordinate = self.reactant1_systems[0].positions[
            self.reaction.binding_atom_idx_reactant1 - len(self.reaction.reactant1)
        ]
        return self.get_sites_within_r(
            center_coordinate,
            self.product1_systems,
            self.product1_energies,
            self.reaction.binding_atom_idx_product1 - len(self.reaction.product1),
            self.r_product1_max,
            0,
            n_sites,
        )

    def get_best_unique_sites_for_product2(self, product1: ase.Atoms, n_sites: int = 5):
        """
        Wrapper to find product 2 placements to be considered for the final frame
        of the NEB.

        Args:
            product1 (ase.Atoms): The atoms object of the product 1 placement that will be
                considered in this function to search for product 1 + product 2 combinations
                for the final frame.
            n_sites (int): The number of sites for product 1 to consider. Notice this is
                multiplicative with product 2 sites (i.e. if 2 is specified here and 3 there)
                then a total of 6 initial and final frames will be considered.

        Returns:
            (list[ase.Atoms]): the lowest energy, proximate placements of product
                2 to be used in the final NEB frames
        """
        center_coordinate = product1.positions[
            self.reaction.binding_atom_idx_product1 - len(self.reaction.product1)
        ]
        sites = self.get_sites_within_r(
            center_coordinate,
            self.product2_systems,
            self.product2_energies,
            self.reaction.binding_atom_idx_product2 - len(self.reaction.product2),
            self.r_product2_max,
            self.r_product2_min,
            n_sites,
        )
        final_frames = []
        for site in sites:
            adsorbate2 = site[
                [idx for idx, tag in enumerate(site.get_tags()) if tag == 2]
            ]
            product1_c = product1.copy()
            product1_c += adsorbate2
            final_frames.append(product1_c)
        return final_frames

    def get_sites_within_r(
        self,
        center_coordinate: np.ndarray,
        all_systems: list,
        all_system_energies: list,
        all_systems_binding_idx: int,
        allowed_radius_max: float,
        allowed_radius_min: float,
        n_sites: int = 5,
    ):
        """
        Get the n lowest energy, sites of the systems within r. For now n is
        5 or < 5 if there are fewer than 5 unique sites within r.

        Args:
            center_coordinate (np.ndarray): the coordinate about which r should be
                centered.
            all_systems (list): the list of all systems to be assessed for their
                uniqueness and proximity to the center coordinate.
            all_systems_binding_idx (int): the idx of the adsorbate atom that is
                bound in `all_systems`
            allowed_radius_max (float): the outer radius about `center_coordinate`
                in which the adsorbate must lie to be considered.
            allowed_radius_min (float): the inner radius about `center_coordinate`
                which the adsorbate must lie outside of to be considered.
            n_sites (int): the number of unique sites in r that will be chosen.

        Returns:
            (list[ase.Atoms]): list of systems identified as candidates.

        """
        # # Sort the systems to make it easy to choose the lowest energy structures
        energies_arg_sort = np.argsort(all_system_energies)
        all_systems_sorted = [all_systems[idx] for idx in energies_arg_sort]

        # Find the systems within the cutoff radius
        system_idxs_in_range = []
        for idx, system in enumerate(all_systems_sorted):
            dummy_system = system.copy()
            dummy_system += ase.Atoms("H", positions=[center_coordinate])
            distance = dummy_system.get_distance(
                all_systems_binding_idx - 1, -1, mic=True
            )
            if distance <= allowed_radius_max and distance >= allowed_radius_min:
                system_idxs_in_range.append(idx)

        # return the 5 lowest energy configurations (where available)
        systems_in_r = [all_systems_sorted[idx] for idx in system_idxs_in_range]
        if len(system_idxs_in_range) > n_sites:
            return systems_in_r[0:n_sites]
        else:
            return systems_in_r


class AutoFrameTransfer(AutoFrame):
    def __init__(
        self,
        reaction: Reaction,
        reactant1_systems: list,
        reactant2_systems: list,
        reactant1_energies: list,
        reactant2_energies: list,
        product1_systems: list,
        product1_energies: list,
        product2_systems: list,
        product2_energies: list,
        r_traverse_max: float,
        r_react_max: float,
        r_react_min: float,
    ):
        """
        Initialize class to handle the automatic generation of NEB frames for transfer reactions.

        Args:
            reaction (ocp.core.Reaction): the reaction object which provides pertinent info
            reactant1_systems (ase.Atoms): the relaxed atoms objects of reactant 1. A list of
                multiple relaxed adsorbate placements should be provided so that multiple
                possible NEBs can be created.
            reactant2_systems (ase.Atoms): the relaxed atoms objects of reactant 2. A list of
                multiple relaxed adsorbate placements should be provided so that multiple
                possible NEBs can be created.
            reactant1_energies (list[float]): the energies of the systems included in
                `product1_systems`
            reactant2_energies (list[float]): the energies of the systems included in
                `product1_systems`
            product1_systems (list[ase.Atoms]): a list of relaxed atoms objects of
                the product that contains the binding atom. A list of multiple relaxed
                adsorbate placements should be provided so that multiple possible NEBs
                can be created.
            product1_energies (list[float]): the energies of the systems included in
                `product1_systems`
            product2_systems (list[ase.Atoms]): a list of relaxed atoms objects of
                the product that does not contain the binding atom. A list of
                multiple relaxed adsorbate placements should be provided so that multiple
                possible NEBs can be created.
            product2_energies (list[float]): the energies of the systems included in
                `product2_systems`
            r_traverse_max (float): the radius about the binding atom of `reactant1_system`
                for which configurations of product 1 will be considered for the final
                frame of the NEB if inside.
            r_react_max (float): we look at all of the pairwise atom interstitial distances between the
                atoms in reactant 1 and reactant 2. If the minimum such distance is greater than this
                value, the combination of reactant 1 and reactant 2 is not considered
            r_react_min (float): we look at all of the pairwise atom interstitial distances between the
                atoms in reactant 1 and reactant 2. If the minimum such distance is less than this
                value, the combination of reactant 1 and reactant 2 is not considered
        """

        # Store info to the class
        self.reaction = reaction
        self.r_traverse_max = r_traverse_max
        self.r_react_max = r_react_max
        self.r_react_min = r_react_min

        # Get unique systems and store to class
        self.reactant1_systems, self.reactant1_energies = self.only_keep_unique_systems(
            reactant1_systems, reactant1_energies
        )
        self.reactant2_systems, self.reactant2_energies = self.only_keep_unique_systems(
            reactant2_systems, reactant2_energies
        )
        self.product1_systems, self.product1_energies = self.only_keep_unique_systems(
            product1_systems, product1_energies
        )
        self.product2_systems, self.product2_energies = self.only_keep_unique_systems(
            product2_systems, product2_energies
        )

    def get_neb_frames(
        self,
        calculator,
        n_frames: int = 10,
        n_initial_frames: int = 5,
        n_final_frames_per_initial: int = 5,
        fmax: float = 0.05,
        steps: int = 200,
    ):
        """
        Propose final frames for NEB calculations. Perform a relaxation on the final
        frame using the calculator provided. Linearly interpolate between the initial
        and final frames for a proposed reaction trajectory. Correct the trajectory if
        there is any atomic overlap.

        Args:
            calculator: an ase compatible calculator to be used to relax the initial and
                final frames.
            n_frames (int): the number of frames per reaction trajectory
            n_initial_frames (int): The number of initial frames to consider
            n_final_frames_per_initial (int): The number of final frames per inital frame to consider
            fmax (float): force convergence criterion for final frame optimization
            steps (int): step number termination criterion for final frame optimization

        Returns:
            list[lists]: the initial reaction coordinates

        """
        # Get proposed initial frames
        initial_final_frame_pairs = []
        initial_frames, pseudoenergies = self.get_system_pairs_initial()

        len_rxt2 = list(self.reactant2_systems[0].get_tags()).count(2)
        len_rxt1 = list(self.reactant1_systems[0].get_tags()).count(2)
        pseudoenergies_sort = np.argsort(pseudoenergies)

        # Iterate over the potential initial frames in order of low pseudoenergy to high pseudoenergy
        # Until we have found `n_initial_frame` candidates
        idx = 0
        relaxed_initial_frames = []
        while len(relaxed_initial_frames) < n_initial_frames and idx + 1 < len(
            pseudoenergies_sort
        ):
            # Relax the combined initial frame (this does not yet exist because we concatenated the rxt1 + rxt 2 obj)
            frame_now = initial_frames[pseudoenergies_sort[idx]].copy()
            frame_now.calc = calculator
            opt = BFGS(frame_now)
            converged = opt.run(fmax=fmax, steps=steps)

            # Check for anomolous behavior
            dt = DetectTrajAnomaly(
                initial_frames[pseudoenergies_sort[idx]],
                frame_now,
                frame_now.get_tags(),
            )
            detector_bool = all(
                [
                    not dt.is_adsorbate_intercalated(),
                    not dt.is_adsorbate_desorbed(),
                    not dt.has_surface_changed(),
                ]
            )

            # If there are no anomalies and the adsorbates are bonded in the way we expect, we will procede with the system
            if (
                converged
                and is_edge_list_respected(frame_now, self.reaction.edge_list_initial)
                and detector_bool
            ):
                relaxed_final_frames = []

                reactant_system_c = frame_now.copy()
                cell_center = deepcopy(
                    np.dot(np.array([0.5, 0.5, 0.5]), reactant_system_c.cell)
                )
                slab_len = len(reactant_system_c) - list(
                    reactant_system_c.get_tags()
                ).count(2)
                site = deepcopy(
                    reactant_system_c.positions[
                        self.reaction.binding_atom_idx_reactant1 + slab_len
                    ]
                )
                reactant_system_c.translate(cell_center - site)
                reactant_system_c.wrap()
                relaxed_initial_frames.append(reactant_system_c)

                # Iterate over the potential final frames in order of low pseudoenergy to high pseudoenergy
                # Until we have found `n_final_frames_per_initial` candidates
                final_frames, pseudo_energies_final = self.get_system_pairs_final(
                    frame_now.positions[
                        self.reaction.binding_atom_idx_reactant1 - len_rxt2 - len_rxt1
                    ],
                    frame_now.positions[
                        self.reaction.binding_atom_idx_reactant2 - len_rxt2
                    ],
                )
                idx2 = 0
                frame_now_slab = frame_now.copy()[
                    [idx for idx, tag in enumerate(frame_now.get_tags()) if tag != 2]
                ]

                # While there are still final frames to consider and we do not yet have n_final frames per initial,
                # consider additional final frames.
                while len(
                    relaxed_final_frames
                ) < n_final_frames_per_initial and idx2 + 1 < len(final_frames):
                    # Relax the final frame
                    frame_now_fin = final_frames[pseudo_energies_final[idx2]].copy()
                    frame_now_fin.calc = calculator
                    opt = BFGS(frame_now_fin)
                    converged = opt.run(fmax=fmax, steps=steps)

                    # Check for anomalies
                    dt2 = DetectTrajAnomaly(
                        final_frames[pseudo_energies_final[idx2]],
                        frame_now_fin,
                        frame_now_fin.get_tags(),
                        frame_now_slab,
                    )
                    detector_bool2 = all(
                        [
                            not dt2.is_adsorbate_intercalated(),
                            not dt2.is_adsorbate_desorbed(),
                            not dt2.has_surface_changed(),
                        ]
                    )

                    # If there are no anomalies and the bonds we are expecting to have are respected, we will consider
                    # the initial - final set for interpolation
                    if (
                        converged
                        and is_edge_list_respected(
                            frame_now_fin, self.reaction.edge_list_final
                        )
                        and detector_bool2
                    ):
                        product_system_c = frame_now_fin.copy()
                        product_system_c.translate(cell_center - site)
                        product_system_c.wrap()
                        relaxed_final_frames.append(product_system_c)
                        initial_final_frame_pairs.append(
                            [reactant_system_c, product_system_c]
                        )
                    idx2 += 1
            idx += 1

        # Correct the adsorbate index order of the final frame to match the initial
        # Once corrected, interpolate between the intial and final state and save the
        # NEB frame set.
        neb_frame_sets = []
        map_idx_list = []
        for init_fin_frame_set in initial_final_frame_pairs:
            final = init_fin_frame_set[1]
            initial = init_fin_frame_set[0]
            if len(self.reaction.idx_mapping) > 1:
                mapping_to_use, map_idx = self.get_most_proximate_symmetric_group(
                    initial, final
                )
                reordered_converged_final_frame = self.reorder_adsorbate(
                    final, mapping_to_use
                )

            else:
                map_idx = 0
                reordered_converged_final_frame = self.reorder_adsorbate(
                    final, self.reaction.idx_mapping[map_idx]
                )

            neb_frames = interpolate_and_correct_frames(
                initial,
                reordered_converged_final_frame,
                n_frames,
                self.reaction,
                map_idx,
            )

            if len(neb_frames) > 0:
                neb_frame_sets.append(neb_frames)
                map_idx_list.append(map_idx)

        return neb_frame_sets, map_idx_list

    def get_system_pairs_initial(self):
        """
        Get the initial frames for the NEB. This is done by finding the closest
        pair of systems from `systems1` and `systems2` for which the interstitial distance
        between all adsorbate atoms is less than `rmax` and greater than `rmin`.

        Returns:
            list[ase.Atoms]: the initial frames for the NEB
            list[float]: the pseudo energies of the initial frames (i.e just the sum of the
                individual adsorption energies)
        """
        rmax = self.r_react_max
        rmin = self.r_react_min

        # Get the initial frames
        initial_frames = []
        pseudo_energies = []
        for idx_sys1, system1 in enumerate(self.reactant1_systems):
            adsorbate1 = system1[
                [idx for idx, tag in enumerate(system1.get_tags()) if tag == 2]
            ].copy()
            for idx_sys2, system2 in enumerate(self.reactant2_systems):
                # Get the adsorbate
                adsorbate2 = system2[
                    [idx for idx, tag in enumerate(system2.get_tags()) if tag == 2]
                ]

                # Get the distance between the binding atom and the adsorbate
                dummy_system = adsorbate2.copy()
                dummy_system.cell = system2.cell
                dummy_system.pbc = system2.pbc
                dummy_system += adsorbate1

                all_interstitial_distances = []
                rxt1_radii = [
                    covalent_radii[atomic_numbers[el]]
                    for el in adsorbate1.get_chemical_symbols()
                ]
                rxt1_idxs = np.array(range(len(adsorbate1))) + len(adsorbate2)
                rxt2_chem_syms = adsorbate2.get_chemical_symbols()
                for rxt2_idx in range(len(adsorbate2)):
                    r_now = covalent_radii[atomic_numbers[rxt2_chem_syms[rxt2_idx]]]
                    distances_now = list(
                        dummy_system.get_distances(rxt2_idx, rxt1_idxs, mic=True)
                    )
                    interstitial_distances_now = [
                        distance_now - r_now - rxt1_radii[idx]
                        for idx, distance_now in enumerate(distances_now)
                    ]
                    all_interstitial_distances.extend(interstitial_distances_now)

                # Check if the distance is within the bounds
                if (rmin < min(all_interstitial_distances)) and (
                    rmax > min(all_interstitial_distances)
                ):
                    initial_frame = system1.copy()
                    initial_frame += adsorbate2
                    initial_frames.append(initial_frame)
                    pseudo_energies.append(
                        self.reactant1_energies[idx_sys1]
                        + self.reactant2_energies[idx_sys2]
                    )

        return initial_frames, pseudo_energies

    def get_system_pairs_final(self, system1_coord, system2_coord):
        """
        Get the final frames for the NEB. This is done by finding the closest
        pair of systems from `systems1` and `systems2` for which the distance
        traversed by the adsorbate from the initial frame to the final frame is
        less than `rmax` and the minimum interstitial distance between the two
        products in greater than `rmin`.

        Returns:
            list[ase.Atoms]: the initial frames for the NEB
            list[float]: the pseudo energies of the initial frames

        """
        rmax = self.r_traverse_max
        rmin = self.r_react_min
        # Get the initial frames
        final_frames = []
        pseudo_energies = []
        # Iterate over the product 1 systems and see how far rxt 1 would have
        #  to traverse to be located at product 1s location
        for idx_sys1, system1 in enumerate(self.product1_systems):
            adsorbate1 = system1[
                [idx for idx, tag in enumerate(system1.get_tags()) if tag == 2]
            ].copy()
            dummy_system = system1.copy()
            dummy_system += ase.Atoms("H", positions=[system1_coord])
            distance_now = dummy_system.get_distance(
                self.reaction.binding_atom_idx_product1
                - 1
                - len(self.reaction.product1),
                -1,
                mic=True,
            )

            # If this distance is less than the max traversal distance, then keep pdt1s placement
            # and look for pdt2 placements
            if distance_now < rmax:
                for idx_sys2, system2 in enumerate(self.product2_systems):
                    # Get the adsorbate from system 2
                    adsorbate2 = system2[
                        [idx for idx, tag in enumerate(system2.get_tags()) if tag == 2]
                    ]

                    # Repeat the process to see how far rxt2 would have to traverse to assume the position of pdt2
                    dummy_system = system2.copy()
                    dummy_system += ase.Atoms("H", positions=[system2_coord])
                    distance_now = dummy_system.get_distance(
                        self.reaction.binding_atom_idx_product2
                        - 1
                        - len(self.reaction.product2),
                        -1,
                        mic=True,
                    )

                    # If this distance is less than the max traversal distance then there is just one more check
                    if distance_now < rmax:
                        # Get the interstitial distances between product 1 and product 2 and ensure the minimum
                        # interstitial distance is greater than the reaction min.
                        dummy_system = adsorbate2.copy()
                        dummy_system.cell = system2.cell
                        dummy_system.pbc = system2.pbc
                        dummy_system += adsorbate1

                        all_interstitial_distances = []
                        pdt1_radii = [
                            covalent_radii[atomic_numbers[el]]
                            for el in adsorbate1.get_chemical_symbols()
                        ]
                        pdt1_idxs = np.array(range(len(adsorbate1))) + len(adsorbate2)
                        pdt2_chem_syms = adsorbate2.get_chemical_symbols()

                        for pdt2_idx in range(len(adsorbate2)):
                            r_now = covalent_radii[
                                atomic_numbers[pdt2_chem_syms[pdt2_idx]]
                            ]
                            distances_now = list(
                                dummy_system.get_distances(
                                    pdt2_idx, pdt1_idxs, mic=True
                                )
                            )
                            interstitial_distances_now = [
                                distance_now - r_now - pdt1_radii[idx]
                                for idx, distance_now in enumerate(distances_now)
                            ]
                            all_interstitial_distances.extend(
                                interstitial_distances_now
                            )
                        # Check if the distance is within the bounds
                        if rmin < min(all_interstitial_distances):
                            final_frame = system1.copy()
                            final_frame += adsorbate2
                            final_frames.append(final_frame)
                            pseudo_energies.append(
                                self.product1_energies[idx_sys1]
                                + self.product2_energies[idx_sys2]
                            )
        pseudo_energies_sort = np.argsort(pseudo_energies)
        return final_frames, pseudo_energies_sort


class AutoFrameDesorption(AutoFrame):
    def __init__(
        self,
        reaction: Reaction,
        reactant_systems: list,
        reactant_energies: list,
        z_desorption: float,
    ):
        """
        Initialize class to handle the automatic generation of NEB frames for desorption reactions.

        Args:
            reaction (Reaction): the reaction object which provides pertinent info
            reactant_systems (list[ase.Atoms]): the relaxed atoms objects of the adsorbed system.
                A list of multiple relaxed adsorbate placements should be provided so that multiple
                possible NEBs can be created.
            reactant_energies (list[float]): the energies of the systems included in `reactant_systems`
            z_desorption (float): the distance along the vector normal to the surface that the
                adsorbate will be translated to instantiate the final frame of the NEB.
        """

        # Store the binding indices and product indices / mapping to the class
        self.reactant1_systems, self.reactant1_energies = self.only_keep_unique_systems(
            reactant_systems, reactant_energies
        )
        self.z_desorption = z_desorption
        self.reaction = reaction

    def get_neb_frames(
        self,
        calculator,
        n_frames: int = 5,
        n_systems: int = 5,
        fmax: float = 0.05,
        steps: int = 200,
    ):
        """
        Propose final frames for NEB calculations. Perform a relaxation on the final
        frame using the calculator provided. Linearly interpolate between the initial
        and final frames for a proposed reaction trajectory. Correct the trajectory if
        there is any atomic overlap.

        Args:
            calculator: an ase compatible calculator to be used to relax the final frame.
            n_frames (int): the number of frames per reaction trajectory
            n_pdt1_sites (int): The number of product 1 sites to consider
            n_pdt2_sites (int): The number of product 2 sites to consider. Note this is
                multiplicative with `n_pdt1_sites` (i.e. if `n_pdt1_sites` = 2 and
                `n_pdt2_sites` = 3 then a total of 6 final frames will be proposed)
            fmax (float): force convergence criterion for final frame optimization
            steps (int): step number termination criterion for final frame optimization

        Returns:
            list[lists]: the initial reaction coordinates

        """
        # Get proposed intitial frames
        ## Check that the reactant has the bonds we expect
        initial_frames = []
        initial_adsorbates = []
        idx_it = 0
        energies_sort = np.argsort(self.reactant1_energies)
        while (len(initial_frames) < n_systems) and (
            (idx_it + 1) < len(self.reactant1_systems)
        ):
            reactant_system = self.reactant1_systems[energies_sort[idx_it]]
            if is_edge_list_respected(
                reactant_system, self.reaction.edge_list_initial
            ) and is_adsorbate_adsorbed(reactant_system):
                adsorbate = reactant_system[
                    [
                        idx
                        for idx, tag in enumerate(reactant_system.get_tags())
                        if tag == 2
                    ]
                ]

                initial_frames.append(reactant_system)
                initial_adsorbates.append(adsorbate)
            else:
                print("not fully connected or desorbed")
            idx_it += 1

        # Get proposed final frames
        final_frames = []
        normal = np.cross(initial_frames[0].cell[0], initial_frames[0].cell[1])
        unit_normal = normal / np.linalg.norm(normal)
        for frame in initial_frames:
            final_slab = frame[
                [idx for idx, tag in enumerate(frame.get_tags()) if tag != 2]
            ].copy()
            final_adsorbate = frame[
                [idx for idx, tag in enumerate(frame.get_tags()) if tag == 2]
            ].copy()
            final_adsorbate.translate(unit_normal * self.z_desorption)
            final_frames.append(final_slab + final_adsorbate)

        # Relax final frames
        converged_final_frames = []
        for idx, frame in enumerate(final_frames):
            frame.calc = calculator
            opt = BFGS(frame)
            converged = opt.run(fmax=fmax, steps=steps)
            if (
                converged
                and (not is_adsorbate_adsorbed(frame))
                and (is_edge_list_respected(frame, self.reaction.edge_list_final))
            ):
                slab_of_initial = initial_frames[idx][
                    [
                        idx
                        for idx, tag in enumerate(initial_frames[idx].get_tags())
                        if tag != 2
                    ]
                ].copy()
                dt = DetectTrajAnomaly(frame, frame, frame.get_tags(), slab_of_initial)
                if not dt.has_surface_changed():
                    converged_final_frames.append(frame)

        # Interpolate between the initial and final frames
        neb_frame_sets = []
        for idx, frame in enumerate(converged_final_frames):
            neb_frames = interpolate_and_correct_frames(
                initial_frames[idx],
                frame,
                n_frames,
                self.reaction,
                0,
            )
            if len(neb_frames) > 0:
                neb_frame_sets.append(neb_frames)

        return neb_frame_sets


def interpolate_and_correct_frames(
    initial: ase.Atoms,
    final: ase.Atoms,
    n_frames: int,
    reaction: Reaction,
    map_idx: int,
):
    """
    Given the initial and final frames, perform the following:
    (1) Unwrap the final frame if it is wrapped around the cell
    (2) Interpolate between the initial and final frames

    Args:
        initial (ase.Atoms): the initial frame of the NEB
        final (ase.Atoms): the proposed final frame of the NEB
        n_frames (int): The desired number of frames for the NEB (not including initial and final)
        reaction (Reaction): the reaction object which provides pertinent info
        map_idx (int): the index of the mapping to use for the final frame
    """
    # Perform checks
    edge_list_final = reorder_edge_list(
        reaction.edge_list_final, reaction.idx_mapping[map_idx]
    )
    assert is_edge_list_respected(initial, reaction.edge_list_initial)
    assert is_edge_list_respected(final, edge_list_final)

    if reaction.reaction_type == "desorption":
        assert is_adsorbate_adsorbed(initial)
        assert not is_adsorbate_adsorbed(final)

    # Perform the interpolation
    initial, final = unwrap_atoms(
        initial,
        final,
        reaction,
        map_idx,
    )
    return interpolate(initial, final, n_frames)


def get_shortest_path(
    initial: ase.Atoms,
    final: ase.Atoms,
):
    """
    Find the shortest path for all atoms about pbc and reorient the final frame so the
    atoms align with this shortest path. This allows us to perform a linear interpolation
    that does not interpolate jumps across pbc.

    Args:
        initial (ase.Atoms): the initial frame of the NEB
        final (ase.Atoms): the proposed final frame of the NEB to be corrected

    Returns:
        (ase.Atoms): the corrected final frame
        (ase.Atoms): the initial frame tiled (3,3,1), which is used it later steps
        (ase.Atoms): the final frame tiled (3,3,1), which is used it later steps
    """
    # Tile the atoms so all placements about the center cell are considered.
    ## For the final frame
    vectors = [
        v for v in final.cell if ((round(v[0], 3) != 0) or (round(v[1], 3 != 0)))
    ]
    repeats = list(product([-1, 0, 1], repeat=2))
    repeats.remove((0, 0))
    new_atoms_final = copy.deepcopy(final)
    for repeat in repeats:
        atoms_shifted = copy.deepcopy(final)
        atoms_shifted.set_positions(
            final.get_positions() + vectors[0] * repeat[0] + vectors[1] * repeat[1]
        )
        new_atoms_final += atoms_shifted

    # For the initial frame
    new_atoms_initial = copy.deepcopy(initial)
    for repeat in repeats:
        atoms_shifted = copy.deepcopy(initial)
        atoms_shifted.set_positions(
            initial.get_positions() + vectors[0] * repeat[0] + vectors[1] * repeat[1]
        )
        new_atoms_initial += atoms_shifted

    # Find the shortest path between the initial and final atoms and unwrap atoms in the final
    # frame about these positions. These corrected positions will be kept for (1) all slab
    # atoms (2) the bound atom of product 1 (3) the atom of product 2 which formed a new bond
    shortest_path_final_positions = []
    equivalent_idx_factors = len(initial) * np.array(list(range(9)))
    for idx, _atom in enumerate(initial):
        equivalent_indices = equivalent_idx_factors + idx
        final_distances = [
            euclidean(initial.positions[idx], new_atoms_final.positions[i])
            for i in equivalent_indices
        ]
        min_idx = np.argmin(final_distances)
        shortest_path_final_positions.append(
            new_atoms_final.positions[equivalent_indices[min_idx]]
        )

    # In place, assign the atoms to the shortest path positions
    final.set_positions(np.array(shortest_path_final_positions))
    return final, new_atoms_initial, new_atoms_final


def traverse_adsorbate_transfer(
    reaction: Reaction,
    initial: ase.Atoms,
    final: ase.Atoms,
    initial_tiled: ase.Atoms,
    final_tiled: ase.Atoms,
    edge_list_final: list,
):
    """
    Traverse reactant 1, reactant 2, product 1 and product 2 in a depth first search of
    the bond graph. Unwrap the atoms to minimize the distance over the bonds. This ensures
    that when we perform the linear interpolation, the adsorbate moves as a single moity
    and avoids accidental bond breaking events over pbc.

    Args:
        reaction (Reaction): the reaction object which provides pertinent info
        initial (ase.Atoms): the initial frame of the NEB
        final (ase.Atoms): the proposed final frame of the NEB to be corrected
        initial_tiled (ase.Atoms): the initial frame tiled (3,3,1)
        final_tiled (ase.Atoms): the final frame tiled (3,3,1)
        edge_list_final (list): the edge list of the final frame corrected with mapping
            idx changes

    Returns:
        (ase.Atoms): the corrected initial frame
        (ase.Atoms): the corrected final frame
    """

    # Unpack the reactant binding indices
    reactant1_binding_idx = reaction.binding_atom_idx_reactant1
    reactant2_binding_idx = reaction.binding_atom_idx_reactant2 + len(
        reaction.reactant1
    )
    equivalent_idx_factors = len(initial) * np.array(list(range(9)))
    slab_len = list(initial.get_tags()).count(1) + list(initial.get_tags()).count(0)

    # Make a networkx graph of the adsorbates
    ads_graph_initial = nx.from_edgelist(reaction.edge_list_initial)
    ads_graph_final = nx.from_edgelist(edge_list_final)

    # Perform a depth first search to order the traversal
    if len(reaction.reactant1) >= 2:
        traversal_rxt1_initial = list(
            nx.dfs_edges(ads_graph_initial, source=reactant1_binding_idx)
        )
    else:
        traversal_rxt1_initial = []
    if len(reaction.product1) >= 2:
        traversal_rxt1_final = list(
            nx.dfs_edges(ads_graph_final, source=reactant1_binding_idx)
        )
    else:
        traversal_rxt1_final = []

    if len(reaction.reactant2) >= 2:
        traversal_rxt2_initial = list(
            nx.dfs_edges(ads_graph_initial, source=reactant2_binding_idx)
        )
    else:
        traversal_rxt2_initial = []

    if len(reaction.product2) >= 2:
        traversal_rxt2_final = list(
            nx.dfs_edges(ads_graph_final, source=reactant2_binding_idx)
        )
    else:
        traversal_rxt2_final = []

    # Traverse the adsorbate(s) and choose positions that minimize the distance
    # over the bonds
    initial = traverse_adsorbate_general(
        traversal_rxt1_initial,
        slab_len,
        reactant1_binding_idx,
        equivalent_idx_factors,
        initial,
        initial_tiled,
    )

    initial = traverse_adsorbate_general(
        traversal_rxt2_initial,
        slab_len,
        reactant2_binding_idx,
        equivalent_idx_factors,
        initial,
        initial_tiled,
    )

    final = traverse_adsorbate_general(
        traversal_rxt1_final,
        slab_len,
        reactant1_binding_idx,
        equivalent_idx_factors,
        final,
        final_tiled,
    )

    final = traverse_adsorbate_general(
        traversal_rxt2_final,
        slab_len,
        reactant2_binding_idx,
        equivalent_idx_factors,
        final,
        final_tiled,
    )

    return initial, final


def traverse_adsorbate_dissociation(
    reaction: Reaction,
    initial: ase.Atoms,
    final: ase.Atoms,
    initial_tiled: ase.Atoms,
    final_tiled: ase.Atoms,
    edge_list_final: int,
):
    """
    Traverse reactant 1, product 1 and product 2 in a depth first search of
    the bond graph. Unwrap the atoms to minimize the distance over the bonds. This ensures
    that when we perform the linear interpolation, the adsorbate moves as a single moity
    and avoids accidental bond breaking events over pbc.

    Args:
        reaction (Reaction): the reaction object which provides pertinent info
        initial (ase.Atoms): the initial frame of the NEB
        final (ase.Atoms): the proposed final frame of the NEB to be corrected
        initial_tiled (ase.Atoms): the initial frame tiled (3,3,1)
        final_tiled (ase.Atoms): the final frame tiled (3,3,1)
        edge_list_final (list): the edge list of the final frame corrected with mapping
            idx changes

    Returns:
        (ase.Atoms): the corrected initial frame
        (ase.Atoms): the corrected final frame
    """

    # Unpack the reactant binding index
    reactant1_binding_idx = reaction.binding_atom_idx_reactant1
    slab_len = list(initial.get_tags()).count(1) + list(initial.get_tags()).count(0)

    # Find equivaled indices for the tiled systems
    equivalent_idx_factors = len(initial) * np.array(list(range(9)))

    # Make a networkx graph of the adsorbates
    ads_graph_initial = nx.from_edgelist(reaction.edge_list_initial)
    ads_graph_final = nx.from_edgelist(edge_list_final)

    # Perform a depth first search to order the traversal
    traversal_rxt1_initial = list(
        nx.dfs_edges(ads_graph_initial, source=reactant1_binding_idx)
    )
    if len(reaction.product1) >= 2:
        traversal_rxt1_final = list(
            nx.dfs_edges(ads_graph_final, source=reactant1_binding_idx)
        )
    else:
        traversal_rxt1_final = []

    # Get product 2 binding index
    if len(reaction.product1) == 1:
        product2_idx = get_product2_idx(
            reaction, edge_list_final, [[reactant1_binding_idx]]
        )
    else:
        product2_idx = get_product2_idx(reaction, edge_list_final, traversal_rxt1_final)

    if len(reaction.product2) >= 2:
        traversal_rxt2 = list(nx.dfs_edges(ads_graph_final, source=product2_idx))
    else:
        traversal_rxt2 = []

    # Traverse the adsorbate(s) and choose positions that minimize the distance
    # over the bonds
    initial = traverse_adsorbate_general(
        traversal_rxt1_initial,
        slab_len,
        reactant1_binding_idx,
        equivalent_idx_factors,
        initial,
        initial_tiled,
    )

    final = traverse_adsorbate_general(
        traversal_rxt1_final,
        slab_len,
        reactant1_binding_idx,
        equivalent_idx_factors,
        final,
        final_tiled,
    )
    final = traverse_adsorbate_general(
        traversal_rxt2,
        slab_len,
        product2_idx,
        equivalent_idx_factors,
        final,
        final_tiled,
    )
    return initial, final


def traverse_adsorbate_desorption(
    reaction: Reaction,
    initial: ase.Atoms,
    final: ase.Atoms,
    initial_tiled: ase.Atoms,
    final_tiled: ase.Atoms,
):
    """
    Traverse reactant 1 and  product 1 in a depth first search of
    the bond graph. Unwrap the atoms to minimize the distance over the bonds. This ensures
    that when we perform the linear interpolation, the adsorbate moves as a single moity
    and avoids accidental bond breaking events over pbc.

    Args:
        reaction (Reaction): the reaction object which provides pertinent info
        initial (ase.Atoms): the initial frame of the NEB
        final (ase.Atoms): the proposed final frame of the NEB to be corrected
        initial_tiled (ase.Atoms): the initial frame tiled (3,3,1)
        final_tiled (ase.Atoms): the final frame tiled (3,3,1)
        edge_list_final (list): the edge list of the final frame corrected with mapping
            idx changes

    Returns:
        (ase.Atoms): the corrected initial frame
        (ase.Atoms): the corrected final frame
    """
    # Unpack the reactant binding indices
    reactant1_binding_idx = reaction.binding_atom_idx_reactant1
    equivalent_idx_factors = len(initial) * np.array(list(range(9)))
    slab_len = list(initial.get_tags()).count(1) + list(initial.get_tags()).count(0)

    # Make a graph of the adsorbate to be traversed
    ads_graph = nx.from_edgelist(reaction.edge_list_initial)

    # Perform a depth first search to order the traversal
    traversal_rxt1 = list(nx.dfs_edges(ads_graph, source=reactant1_binding_idx))

    # Traverse the adsorbate(s) and choose positions that minimize the distance across bonds
    initial = traverse_adsorbate_general(
        traversal_rxt1,
        slab_len,
        reactant1_binding_idx,
        equivalent_idx_factors,
        initial,
        initial_tiled,
    )

    final = traverse_adsorbate_general(
        traversal_rxt1,
        slab_len,
        reactant1_binding_idx,
        equivalent_idx_factors,
        final,
        final_tiled,
    )
    return initial, final


def get_product2_idx(
    reaction: Reaction,
    edge_list_final: list,
    traversal_rxt1_final: list,
):
    """
    For dissociation only. Use the information about the initial edge list and final edge
    list to determine which atom in product 2 lost a bond in the reaction and use this
    as the binding index for traversal in `traverse_adsorbate_dissociation`.

    Args:
        reaction (Reaction): the reaction object which provides pertinent info
        edge_list_final (list): the edge list of the final frame corrected with mapping
            idx changes
        traversal_rxt1_final (list): the traversal of reactant 1 for the final frame

    Returns:
        (int): the binding index of product 2
    """
    broken_edge = [
        edge for edge in reaction.edge_list_initial if edge not in edge_list_final
    ][0]
    flat_nodes = [item for sublist in traversal_rxt1_final for item in sublist]
    return [idx for idx in broken_edge if idx not in flat_nodes][0]


def traverse_adsorbate_general(
    traversal_rxt,
    slab_len: int,
    starting_node_idx: int,
    equivalent_idx_factors: np.ndarray,
    frame: ase.Atoms,
    frame_tiled: ase.Atoms,
):
    """
    Perform the traversal to reposition atoms so that the distance along bonds is
    minimized.

    Args:
        traversal_rxt (list): the traversal of the adsorbate to be traversed. It is
            the list of edges ordered by depth first search.
        slab_len (int): the number of atoms in the slab
        starting_node_idx (int): the index of the atom to start the traversal from
        equivalent_idx_factors (np.ndarray): the values to add to the untiled index
            which gives equivalent indices (i.e. copies of that atom in the tiled system)
        frame (ase.Atoms): the frame to be corrected
        frame_tiled (ase.Atoms): the tiled (3,3,1) version of the frame which will be
            corrected

    Returns:
        (ase.Atoms): the corrected frame
    """
    nodes_visited = [starting_node_idx]
    for edge in traversal_rxt:
        idx = [i for i in edge if i not in nodes_visited]
        if len(idx) == 1:
            source_node_idx = [i for i in edge if i != idx[0]][0] + slab_len
            nodes_visited.append(idx[0])
            idx = idx[0] + slab_len
            equivalent_indices = equivalent_idx_factors + idx
            initial_distances = [
                euclidean(frame.positions[source_node_idx], frame_tiled.positions[i])
                for i in equivalent_indices
            ]
            min_idx = np.argmin(initial_distances)
            positions_now = frame.get_positions()
            positions_now[idx] = frame_tiled.positions[equivalent_indices[min_idx]]
            frame.set_positions(positions_now)
    return frame


def unwrap_atoms(
    initial: ase.Atoms,
    final: ase.Atoms,
    reaction: Reaction,
    map_idx: int,
):
    """
    Make corrections to the final frame so it is no longer wrapped around the cell,
    if it has jumpped over the pbc. Ensure that for each adsorbate moity, absolute bond distances
    for all edges that exist in the initial and final frames are minimize regardles of cell location.
    This enforces the traversal of the adsorbates happens along the same path, which is not
    necessarily the minimum distance path for each atom. Changes are made in place.

    Args:
        initial (ase.Atoms): the initial atoms object to which the final atoms should
            be proximate
        final (ase.Atoms): the final atoms object to be corrected
        reaction (Reaction): the reaction object which provides pertinent info
        map_idx (int): the index of the mapping to use for the final frame
    """

    # Get the overall placement of the adsorbate(s) in the final frame that
    # supports traversing the minimum path
    final, initial_tiled, final_tiled = get_shortest_path(initial, final)

    # Get the corrected final edge list
    if reaction.reaction_type in ["transfer", "dissociation"]:
        edge_list_final = reorder_edge_list(
            reaction.edge_list_final, reaction.idx_mapping[map_idx]
        )

    if reaction.reaction_type == "transfer":
        initial, final = traverse_adsorbate_transfer(
            reaction, initial, final, initial_tiled, final_tiled, edge_list_final
        )
    elif reaction.reaction_type == "dissociation":
        initial, final = traverse_adsorbate_dissociation(
            reaction, initial, final, initial_tiled, final_tiled, edge_list_final
        )
    elif reaction.reaction_type == "desorption":
        initial, final = traverse_adsorbate_desorption(
            reaction, initial, final, initial_tiled, final_tiled
        )

    return initial, final


def interpolate(initial_frame: ase.Atoms, final_frame: ase.Atoms, num_frames: int):
    """
    Interpolate between the initial and final frames starting with a linear interpolation
    along the atom-wise vectors from initial to final. Then iteratively correct the
    positions so atomic overlap is avoided/ reduced. When iteratively updating, the
    positions of adjacent frames are considered to avoid large jumps in the trajectory.

    Args:
        initial_frame (ase.Atoms): the initial frame which will be interpolated from
        final_frame (ase.Atoms): the final frame which will be interpolated to
        num_frames (int): the number of frames to be interpolated between the initial

    Returns:
        (list[ase.Atoms]): the interpolated frames
    """
    # Linearly interpolate between the initial and final frames
    start_pos = torch.from_numpy(initial_frame.get_positions())
    end_pos = torch.from_numpy(final_frame.get_positions())
    device = start_pos.device
    num_atoms = len(start_pos)

    start_dist = torch.from_numpy(initial_frame.get_all_distances(mic=True))
    end_dist = torch.from_numpy(final_frame.get_all_distances(mic=True))

    alpha = torch.range(0, num_frames - 1, device=device) / (num_frames - 1)
    frames = start_pos.view(1, num_atoms, 3) * alpha.view(-1, 1, 1) + end_pos.view(
        1, num_atoms, 3
    ) * (1.0 - alpha.view(-1, 1, 1))

    target_dist = start_dist.view(1, -1) * alpha.view(-1, 1) + end_dist.view(1, -1) * (
        1.0 - alpha.view(-1, 1)
    )
    atoms_frames = []
    for frame in reversed(frames):
        atoms_now = initial_frame.copy()
        atoms_now.set_positions(frame)
        atoms_frames.append(atoms_now)

    # Iteratively update positions to avoid overlap
    for _i in range(100):
        rate = 0.1

        frame_dist = []
        frame_vec = []
        for frame_now in reversed(atoms_frames):
            distances_now = frame_now.get_all_distances(mic=True)
            distances_now = np.concatenate(distances_now).ravel()
            vec_now = frame_now.get_all_distances(mic=True, vector=True)
            frame_dist.append(distances_now)
            frame_vec.append(vec_now)

        frame_dist = np.array(frame_dist)
        frame_vec = np.array(frame_vec)

        frame_dist = torch.from_numpy(frame_dist)
        frame_vec = torch.from_numpy(frame_vec)

        frame_vec = frame_vec / (torch.norm(frame_vec, dim=3).unsqueeze(-1) + 0.0001)

        # Okay to have a distance greater than target, and larger distances matter less
        delta = torch.clamp(frame_dist - target_dist, max=0)
        weight = torch.exp(-(target_dist * target_dist) / (0.5 * 2.0 * 2.0))
        weight = weight + torch.exp(-(frame_dist * frame_dist) / (0.5 * 2.0 * 2.0))

        delta = delta * weight

        delta = rate * frame_vec * delta.view(num_frames, num_atoms, num_atoms, 1)
        delta = torch.sum(delta, dim=1)
        delta[0] = delta[0] * 0.0
        delta[num_frames - 1] = delta[num_frames - 1] * 0.0

        frames = frames - delta

        ## Consider adjacent frames to avoid large jumps
        mean_pos = (frames[0 : num_frames - 2] + frames[2:num_frames]) / 2.0
        frames[1 : num_frames - 1] = 0.9 * frames[1 : num_frames - 1] + 0.1 * mean_pos

        atoms_frames = []
        for frame in reversed(frames):
            atoms_now = initial_frame.copy()
            atoms_now.set_positions(frame)
            atoms_frames.append(atoms_now)
    for atoms_now in atoms_frames:
        atoms_now.wrap()
    return atoms_frames


def is_edge_list_respected(frame: ase.Atoms, edge_list: list):
    """
    Check to see that the expected adsorbate-adsorbate edges are found and no additional
    edges exist between the adsorbate atoms.

    Args:
        frame (ase.Atoms): the atoms object for which edges will be checked.
            This must comply with ocp tagging conventions.
        edge_list (list[tuples]): The expected edges
    """
    adsorbate = frame[[idx for idx, tag in enumerate(frame.get_tags()) if tag == 2]]
    elements = adsorbate.get_chemical_symbols()
    all_combos = list(combinations(range(len(adsorbate)), 2))
    for combo in all_combos:
        total_distance = adsorbate.get_distance(combo[0], combo[1], mic=True)
        r1 = covalent_radii[atomic_numbers[elements[combo[0]]]]
        r2 = covalent_radii[atomic_numbers[elements[combo[1]]]]
        distance_ratio = total_distance / (r1 + r2)
        if combo in edge_list:
            if distance_ratio > 1.25:
                return False
        elif distance_ratio < 1.25:
            return False
    return True


def reorder_edge_list(
    edge_list: list,
    mapping: dict,
):
    """
    For the final edge list, apply the mapping so the edges correspond to the correctly
    concatenated object.

    Args:
        edge_list (list[tuples]): the final edgelist
        mapping: the mapping so the final atoms concatenated have indices that correctly map
            to the initial atoms.
    """

    inverse_mapping = {v: k for k, v in mapping.items()}
    new_edge_list = []
    for edge in edge_list:
        new_edge_list.append(
            tuple(np.sort([inverse_mapping[edge[0]], inverse_mapping[edge[1]]]))
        )
    return new_edge_list


def is_adsorbate_adsorbed(adsorbate_slab_config: ase.Atoms):
    """
    Check to see if the adsorbate is adsorbed on the surface.

    Args:
        adsorbate_slab_config (ase.Atoms): the combined adsorbate and slab configuration
            with adsorbate atoms tagged as 2s and surface atoms tagged as 1s.

    Returns:
        (bool): True if the adsorbate is adsorbed, False otherwise.
    """
    adsorbate = adsorbate_slab_config[
        [idx for idx, tag in enumerate(adsorbate_slab_config.get_tags()) if tag == 2]
    ].copy()
    surface = adsorbate_slab_config[
        [idx for idx, tag in enumerate(adsorbate_slab_config.get_tags()) if tag == 1]
    ].copy()
    ads_elements = adsorbate.get_chemical_symbols()
    surface_elements = surface.get_chemical_symbols()
    ads_surface = surface + adsorbate
    all_combos = list(product(range(len(adsorbate)), range(len(surface))))
    for combo in all_combos:
        total_distance = ads_surface.get_distance(
            combo[1], combo[0] + len(surface_elements), mic=True
        )
        r1 = covalent_radii[atomic_numbers[ads_elements[combo[0]]]]
        r2 = covalent_radii[atomic_numbers[surface_elements[combo[1]]]]
        distance_ratio = total_distance / (r1 + r2)
        if distance_ratio < 1.25:
            return True
    return False
