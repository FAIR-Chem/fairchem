from typing import List

import numpy as np
from ase import Atoms
from ase.data import covalent_radii

from ocdata.core import Adsorbate, AdsorbateSlabConfig, Slab


class MultipleAdsorbateSlabConfig(AdsorbateSlabConfig):
    """
    Class to represent a slab with multiple adsorbates on it. This class only
    returns a fixed combination of adsorbates placed on the surface. Unlike
    AdsorbateSlabConfig which enumerates all possible adsorbate placements, this
    problem gets combinatorially large.

    Arguments
    ---------
    slab: Slab
        Slab object.
    adsorbates: List[Adsorbate]
        List of adsorbate objects to place on the slab.
    num_sites: int
        Number of sites to sample.
    num_configurations: int
        Number of configurations to generate per slab+adsorbate(s) combination.
        This corresponds to selecting different site combinations to place
        the adsorbates on.
    interstitial_gap: float
        Minimum distance, in Angstroms, between adsorbate and slab atoms as
        well as the inter-adsorbate distance.
    mode: str
        "random", "heuristic", or "random_site_heuristic_placement".
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
    """

    def __init__(
        self,
        slab: Slab,
        adsorbates: List[Adsorbate],
        num_sites: int = 100,
        num_configurations: int = 1,
        interstitial_gap: float = 0.1,
        mode: str = "random_site_heuristic_placement",
    ):
        assert mode in ["random", "heuristic", "random_site_heuristic_placement"]
        assert interstitial_gap < 5 and interstitial_gap >= 0

        self.slab = slab
        self.adsorbates = adsorbates
        self.num_sites = num_sites
        self.interstitial_gap = interstitial_gap
        self.mode = mode

        self.sites = self.get_binding_sites(num_sites)
        self.atoms_list, self.metadata_list = self.place_adsorbates_on_sites(
            self.sites,
            num_configurations,
            interstitial_gap,
        )

    def place_adsorbates_on_sites(
        self,
        sites: list,
        num_configurations: int = 1,
        interstitial_gap: float = 0.1,
    ):
        """
        Place the adsorbate at the given binding sites.

        This method generates a fixed number of configurations where sites are
        selected to ensure that adsorbate binding indices are at least a fair
        distance away from each other (covalent radii + interstitial gap).
        While this helps prevent adsorbate overlap it does not gaurantee it
        since non-binding adsorbate atoms can overlap if the right combination
        of angles is sampled.
        """
        # Build a fake atoms object with the positions as the sites.
        # This allows us to easily compute distances while accounting for periodicity.
        pseudo_atoms = Atoms(
            [1] * len(sites), positions=sites, cell=self.slab.atoms.get_cell(), pbc=True
        )
        num_sites = len(sites)

        atoms_list = []
        metadata_list = []
        # NOTE: We can hard enforce these configurations to be non-overlapping.
        for _ in range(num_configurations):
            metadata = []

            # Build mapping to store distance of site to nearest adsorbate.
            # Initialize to an arbitrarily large number to represent no adsorbates placed.
            distance_to_nearest_adsorbate_map = 1e10 * np.ones(num_sites)

            # Randomly select a site to place the first adsorbate
            site_idx = np.random.choice(num_sites)
            site = sites[site_idx]

            initial_adsorbate = self.adsorbates[0]

            # Place adsorbate on site
            base_atoms, sampled_angles = self.place_adsorbate_on_site(
                initial_adsorbate, site, interstitial_gap
            )
            # Keep track of adsorbate indices
            adsorbate_indices = (base_atoms.get_tags() == 2).nonzero()[0]

            metadata.append(
                {
                    "adsorbate": initial_adsorbate,
                    "site": site,
                    "xyz_angles": sampled_angles,
                    "adsorbate_indices": adsorbate_indices,
                }
            )

            # For the initial adsorbate, update the distance mapping based
            distance_to_nearest_adsorbate_map = update_distance_map(
                distance_to_nearest_adsorbate_map,
                site_idx,
                initial_adsorbate,
                pseudo_atoms,
            )

            for idx, adsorbate in enumerate(self.adsorbates[1:]):
                binding_idx = adsorbate.binding_indices[0]
                binding_atom = adsorbate.atoms.get_atomic_numbers()[binding_idx]
                covalent_radius = covalent_radii[binding_atom]

                # A site is allowed if the distance to the next closest adsorbate is
                # at least the interstitial_gap + covalent radius of the binding atom away.
                # The covalent radius of the nearest adsorbate is already considered in the
                # distance mapping.
                mask = (
                    distance_to_nearest_adsorbate_map
                    >= interstitial_gap + covalent_radius
                ).nonzero()[0]

                site_idx = np.random.choice(mask)
                site = sites[site_idx]

                atoms, sampled_angles = self.place_adsorbate_on_site(
                    adsorbate, site, interstitial_gap
                )

                # Slabs are not altered in the adsorbat placement step
                # We can add the adsorbate directly to the base atoms
                adsorbate_indices = np.arange(
                    len(base_atoms), len(base_atoms) + len(adsorbate.atoms)
                )
                base_atoms += atoms[atoms.get_tags() == 2]

                distance_to_nearest_adsorbate_map = update_distance_map(
                    distance_to_nearest_adsorbate_map,
                    site_idx,
                    adsorbate,
                    pseudo_atoms,
                )

                metadata.append(
                    {
                        "adsorbate": adsorbate,
                        "site": site,
                        "xyz_angles": sampled_angles,
                        "adsorbate_indices": adsorbate_indices,
                    }
                )

            atoms_list.append(base_atoms)
            metadata_list.append(metadata)

        return atoms_list, metadata_list

    def get_metadata_dict(self, ind):
        """
        Returns a dict containing the atoms object and metadata for
        one specified config, used for writing to files.
        """
        return {
            "adsorbed_slab_atomsobject": self.atoms_list[ind],
            "adsorbed_slab_metadata": {
                "bulk_id": self.slab.bulk.src_id,
                "millers": self.slab.millers,
                "shift": self.slab.shift,
                "top": self.slab.top,
                "adsorbates": self.metadata_list[ind],
            },
        }


def update_distance_map(prev_distance_map, site_idx, adsorbate, pseudo_atoms):
    """
    Given a new site and the adsorbate we plan on placing there,
    update the distance mapping to reflect the new distances from sites to nearest adsorbates.
    We incorporate the covalent radii of the placed adsorbate binding atom in our distance
    calculation to prevent atom overlap.
    """
    binding_idx = adsorbate.binding_indices[0]
    binding_atom = adsorbate.atoms.get_atomic_numbers()[binding_idx]
    covalent_radius = covalent_radii[binding_atom]

    new_site_distances = (
        pseudo_atoms.get_distances(site_idx, range(len(pseudo_atoms)), mic=True)
        - covalent_radius
    )

    # update previous distance mapping by taking the minimum per-element distance between
    # the new distance mapping for the placed site and the previous mapping.
    updated_distance_map = np.minimum(prev_distance_map, new_site_distances)

    return updated_distance_map
