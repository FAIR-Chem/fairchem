from __future__ import annotations

import copy
import logging
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
import scipy
from ase.data import atomic_numbers, covalent_radii
from ase.geometry import wrap_positions
from fairchem.data.oc.core.adsorbate import randomly_rotate_adsorbate
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.optimize import fsolve

if TYPE_CHECKING:
    import ase
    from fairchem.data.oc.core.slab import Adsorbate, Slab

# warnings.filterwarnings("ignore", "The iteration is not making good progress")


class AdsorbateSlabConfig:
    """
    Initializes a list of adsorbate-catalyst systems for a given Adsorbate and Slab.

    Arguments
    ---------
    slab: Slab
        Slab object.
    adsorbate: Adsorbate
        Adsorbate object.
    num_sites: int
        Number of sites to sample.
    num_augmentations_per_site: int
        Number of augmentations of the adsorbate per site. Total number of
        generated structures will be `num_sites` * `num_augmentations_per_site`.
    interstitial_gap: float
        Minimum distance in Angstroms between adsorbate and slab atoms.
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
        adsorbate: Adsorbate,
        num_sites: int = 100,
        num_augmentations_per_site: int = 1,
        interstitial_gap: float = 0.1,
        mode: str = "random",
    ):
        assert mode in ["random", "heuristic", "random_site_heuristic_placement"]
        assert interstitial_gap < 5
        assert interstitial_gap >= 0

        self.slab = slab
        self.adsorbate = adsorbate
        self.num_sites = num_sites
        self.num_augmentations_per_site = num_augmentations_per_site
        self.interstitial_gap = interstitial_gap
        self.mode = mode

        self.sites = self.get_binding_sites(num_sites)
        self.atoms_list, self.metadata_list = self.place_adsorbate_on_sites(
            self.sites,
            num_augmentations_per_site,
            interstitial_gap,
        )

    def get_binding_sites(self, num_sites: int):
        """
        Returns up to `num_sites` sites given the surface atoms' positions.
        """
        assert self.slab.has_surface_tagged()

        all_sites = []
        if self.mode in ["random", "random_site_heuristic_placement"]:
            # The Delaunay triangulation of surface atoms doesn't take PBC into
            # account, so we end up undersampling triangles near the edges of
            # the central unit cell. To avoid that, we explicitly tile the slab
            # in the x-y plane, then consider the triangles with at least one
            # vertex in the central cell.
            #
            # Pymatgen does something similar where they repeat the central cell
            # in the x-y plane, generate sites, and then remove symmetric sites.
            # https://github.com/materialsproject/pymatgen/blob/v2023.3.23/pymatgen/analysis/adsorption.py#L257
            # https://github.com/materialsproject/pymatgen/blob/v2023.3.23/pymatgen/analysis/adsorption.py#L292
            unit_surface_atoms_idx = [
                i for i, atom in enumerate(self.slab.atoms) if atom.tag == 1
            ]

            tiled_slab_atoms = custom_tile_atoms(self.slab.atoms)
            tiled_surface_atoms_idx = [
                i for i, atom in enumerate(tiled_slab_atoms) if atom.tag == 1
            ]
            tiled_surface_atoms_pos = tiled_slab_atoms[
                tiled_surface_atoms_idx
            ].get_positions()

            dt = scipy.spatial.Delaunay(tiled_surface_atoms_pos[:, :2])
            simplices = dt.simplices

            # Only keep triangles with at least one vertex in central cell.
            pruned_simplices = [
                tri
                for tri in simplices
                if np.any(
                    [
                        tiled_surface_atoms_idx[ver] in unit_surface_atoms_idx
                        for ver in tri
                    ]
                )
            ]
            simplices = np.array(pruned_simplices)

            # Uniformly sample sites on each triangle.
            #
            # We oversample by 2x to account for the fact that some sites will
            # later be removed due to being outside the central cell.
            num_sites_per_triangle = int(np.ceil(2.0 * num_sites / len(simplices)))
            for tri in simplices:
                triangle_positions = tiled_surface_atoms_pos[tri]
                sites = get_random_sites_on_triangle(
                    triangle_positions, num_sites_per_triangle
                )
                all_sites += sites

            # Some vertices will be outside the central cell, drop them.
            uw_sites = np.array(all_sites)
            w_sites = wrap_positions(
                uw_sites, self.slab.atoms.cell, pbc=(True, True, False)
            )
            keep_idx = np.isclose(uw_sites, w_sites).all(axis=1)
            all_sites = uw_sites[keep_idx]

            np.random.shuffle(all_sites)
            return all_sites[:num_sites]
        elif self.mode == "heuristic":
            # Explicitly specify "surface" / "subsurface" atoms so that
            # Pymatgen doesn't recompute tags.
            site_properties = {"surface_properties": []}
            for atom in self.slab.atoms:
                if atom.tag == 1:
                    site_properties["surface_properties"].append("surface")
                else:
                    site_properties["surface_properties"].append("subsurface")
            struct = AseAtomsAdaptor.get_structure(self.slab.atoms)
            # Copy because Pymatgen doesn't let us update site_properties.
            struct = struct.copy(site_properties=site_properties)
            asf = AdsorbateSiteFinder(struct)
            # `distance` refers to the distance along the surface normal between
            # the slab and the adsorbate. We set it to 0 here since we later
            # explicitly check for atomic overlap and set the adsorbate height.
            all_sites += asf.find_adsorption_sites(distance=0)["all"]

            if len(all_sites) > num_sites:
                logging.warning(
                    f"Found {len(all_sites)} sites in `get_binding_sites` run with mode='heuristic' and num_sites={num_sites}. Heuristic mode returns all found sites."
                )

            np.random.shuffle(all_sites)
            return all_sites
        else:
            raise NotImplementedError

    def place_adsorbate_on_site(
        self,
        adsorbate: Adsorbate,
        site: np.ndarray,
        interstitial_gap: float = 0.1,
    ):
        """
        Place the adsorbate at the given binding site.
        """
        adsorbate_c = adsorbate.atoms.copy()
        slab_c = self.slab.atoms.copy()

        binding_idx = None
        if self.mode in ["heuristic", "random_site_heuristic_placement"]:
            binding_idx = np.random.choice(adsorbate.binding_indices)

        # Rotate adsorbate along xyz, only if adsorbate has more than 1 atom.
        sampled_angles = np.array([0, 0, 0])
        if len(adsorbate.atoms) > 1:
            adsorbate_c, sampled_angles = randomly_rotate_adsorbate(
                adsorbate_c,
                mode=self.mode,
                binding_idx=binding_idx,
            )

        # Translate adsorbate to binding site.
        if self.mode == "random":
            placement_center = adsorbate_c.get_center_of_mass()
        elif self.mode in ["heuristic", "random_site_heuristic_placement"]:
            placement_center = adsorbate_c.positions[binding_idx]
        else:
            raise NotImplementedError

        translation_vector = site - placement_center
        adsorbate_c.translate(translation_vector)

        # Translate the adsorbate by the normal so has no intersections
        normal = np.cross(self.slab.atoms.cell[0], self.slab.atoms.cell[1])
        unit_normal = normal / np.linalg.norm(normal)

        scaled_normal = self._get_scaled_normal(
            adsorbate_c,
            slab_c,
            site,
            unit_normal,
            interstitial_gap,
        )
        adsorbate_c.translate(scaled_normal * unit_normal)
        adsorbate_slab_config = slab_c + adsorbate_c
        tags = [2] * len(adsorbate_c)
        final_tags = list(slab_c.get_tags()) + tags
        adsorbate_slab_config.set_tags(final_tags)

        # Set pbc and cell.
        adsorbate_slab_config.cell = (
            slab_c.cell
        )  # Comment (@brookwander): I think this is unnecessary?
        adsorbate_slab_config.pbc = [True, True, False]

        return adsorbate_slab_config, sampled_angles

    def place_adsorbate_on_sites(
        self,
        sites: list,
        num_augmentations_per_site: int = 1,
        interstitial_gap: float = 0.1,
    ):
        """
        Place the adsorbate at the given binding sites.
        """
        atoms_list = []
        metadata_list = []
        for site in sites:
            for _ in range(num_augmentations_per_site):
                atoms, sampled_angles = self.place_adsorbate_on_site(
                    self.adsorbate, site, interstitial_gap
                )
                atoms_list.append(atoms)
                metadata_list.append({"site": site, "xyz_angles": sampled_angles})
        return atoms_list, metadata_list

    def _get_scaled_normal(
        self,
        adsorbate_c: ase.Atoms,
        slab_c: ase.Atoms,
        site: np.ndarray,
        unit_normal: np.ndarray,
        interstitial_gap: float = 0.1,
    ):
        """
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

        Args:
            adsorbate_c (ase.Atoms): A copy of the adsorbate with coordinates at the site
            slab_c (ase.Atoms): A copy of the slab
            site (np.ndarray): the coordinate of the site
            adsorbate_atoms (ase.Atoms): the translated adsorbate
            unit_normal (np.ndarray): the unit vector normal to the surface
            interstitial_gap (float): the desired distance between the covalent radii of the
                closest surface and adsorbate atom
        Returns:
            (float): the magnitude of the normal vector for placement
        """
        # Center everthing about the site so we dont need to deal with pbc issues
        slab_c2 = slab_c.copy()
        cell_center = np.dot(np.array([0.5, 0.5, 0.5]), slab_c2.cell)
        slab_c2.translate(cell_center - site)
        slab_c2.wrap()

        adsorbate_positions = adsorbate_c.get_positions()

        adsorbate_c2 = adsorbate_c.copy()
        adsorbate_c2.translate(cell_center - site)

        # See which combos have a possible intersection event
        combos = self._find_combos_to_check(
            adsorbate_c2, slab_c2, unit_normal, interstitial_gap
        )

        # Solve for the intersections
        def fun(x):
            return (
                (surf_pos[0] - (cell_center[0] + x * unit_normal[0] + u_[0])) ** 2
                + (surf_pos[1] - (cell_center[1] + x * unit_normal[1] + u_[1])) ** 2
                + (surf_pos[2] - (cell_center[2] + x * unit_normal[2] + u_[2])) ** 2
                - (d_min + interstitial_gap) ** 2
            )

        if len(combos) > 0:
            scaled_norms = []
            for combo in combos:
                closest_idxs, d_min, surf_pos = combo
                u_ = adsorbate_positions[closest_idxs[0]] - site
                n_scale = fsolve(fun, d_min * 3)
                scaled_norms.append(n_scale[0])
            return max(scaled_norms)
        else:
            # Comment(@brookwander): this is a kinda scary edge case
            return (
                0  # if there are no possible surface itersections, place it at the site
            )

    def _find_combos_to_check(
        self,
        adsorbate_c2: ase.Atoms,
        slab_c2: ase.Atoms,
        unit_normal: np.ndarray,
        interstitial_gap: float,
    ):
        """
        Find the pairs of surface and adsorbate atoms that would have an intersection event
        while traversing the normal vector. For each pair, return pertanent information for
        finding the point of intersection.
        Args:
            adsorbate_c2 (ase.Atoms): A copy of the adsorbate with coordinates at the centered site
            slab_c2 (ase.Atoms): A copy of the slab with atoms wrapped s.t. things are centered
                about the site
            unit_normal (np.ndarray): the unit vector normal to the surface
            interstitial_gap (float): the desired distance between the covalent radii of the
                closest surface and adsorbate atom

        Returns:
            (list[lists]): each entry in the list corresponds to one pair to check. With the
                following information:
                    [(adsorbate_idx, slab_idx), r_adsorbate_atom + r_slab_atom, slab_atom_position]
        """
        adsorbate_elements = adsorbate_c2.get_chemical_symbols()
        slab_elements = slab_c2.get_chemical_symbols()
        projected_points = self._get_projected_points(
            adsorbate_c2, slab_c2, unit_normal
        )

        pairs = list(product(list(range(len(adsorbate_c2))), list(range(len(slab_c2)))))

        combos_to_check = []
        for combo in pairs:
            distance = np.linalg.norm(
                projected_points["ads"][combo[0]] - projected_points["slab"][combo[1]]
            )
            radial_distance = (
                covalent_radii[atomic_numbers[adsorbate_elements[combo[0]]]]
                + covalent_radii[atomic_numbers[slab_elements[combo[1]]]]
            )
            if distance <= (radial_distance + interstitial_gap):
                combos_to_check.append(
                    [combo, radial_distance, slab_c2.positions[combo[1]]]
                )
        return combos_to_check

    def _get_projected_points(
        self, adsorbate_c2: ase.Atoms, slab_c2: ase.Atoms, unit_normal: np.ndarray
    ):
        """
        Find the x and y coordinates of each atom projected onto the surface plane.
        Args:
            adsorbate_c2 (ase.Atoms): A copy of the adsorbate with coordinates at the centered site
            slab_c2 (ase.Atoms): A copy of the slab with atoms wrapped s.t. things are centered
                about the site
            unit_normal (np.ndarray): the unit vector normal to the surface

        Returns:
            (dict): {"ads": [[x1, y1], [x2, y2], ...], "slab": [[x1, y1], [x2, y2], ...],}
        """
        projected_points = {"ads": [], "slab": []}
        point_on_surface = slab_c2.cell[0]
        for atom_position in adsorbate_c2.positions:
            v_ = atom_position - point_on_surface
            projected_point = point_on_surface + (
                v_
                - (np.dot(v_, unit_normal) / np.linalg.norm(unit_normal) ** 2)
                * unit_normal
            )
            projected_points["ads"].append(projected_point)

        for atom_position in slab_c2.positions:
            v_ = atom_position - point_on_surface
            projected_point = point_on_surface + (
                v_
                - (np.dot(v_, unit_normal) / np.linalg.norm(unit_normal) ** 2)
                * unit_normal
            )
            projected_points["slab"].append(projected_point)
        return projected_points

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
                "smiles": self.adsorbate.smiles,
                "site": self.metadata_list[ind]["site"],
                "xyz_angles": self.metadata_list[ind]["xyz_angles"],
            },
        }


def get_random_sites_on_triangle(
    vertices: np.ndarray,
    num_sites: int = 10,
):
    """
    Sample `num_sites` random sites uniformly on a given 3D triangle.
    Following Sec. 4.2 from https://www.cs.princeton.edu/~funk/tog02.pdf.
    """
    assert len(vertices) == 3
    r1_sqrt = np.sqrt(np.random.uniform(0, 1, num_sites))[:, np.newaxis]
    r2 = np.random.uniform(0, 1, num_sites)[:, np.newaxis]
    sites = (
        (1 - r1_sqrt) * vertices[0]
        + r1_sqrt * (1 - r2) * vertices[1]
        + r1_sqrt * r2 * vertices[2]
    )
    return list(sites)


def custom_tile_atoms(atoms: ase.Atoms):
    """
    Tile the atoms so that the center tile has the indices and positions of the
    untiled structure.

    Args:
        atoms (ase.Atoms): the atoms object to be tiled

    Return:
        (ase.Atoms): the tiled atoms which has been repeated 3 times in
            the x and y directions but maintains the original indices on the central
            unit cell.
    """
    vectors = [
        v for v in atoms.cell if ((round(v[0], 3) != 0) or (round(v[1], 3 != 0)))
    ]
    repeats = list(product([-1, 0, 1], repeat=2))
    repeats.remove((0, 0))
    new_atoms = copy.deepcopy(atoms)
    for repeat in repeats:
        atoms_shifted = copy.deepcopy(atoms)
        atoms_shifted.set_positions(
            atoms.get_positions() + vectors[0] * repeat[0] + vectors[1] * repeat[1]
        )
        new_atoms += atoms_shifted
    return new_atoms


def get_interstitial_distances(adsorbate_slab_config: ase.Atoms, overlap_tag: int = 2):
    """
    Check to see if there is any atomic overlap between atoms with a particular
    tag and all other atoms. Used to check overlap between adsorbate and
    surface atoms.

    Args:
        adsorbate_slab_configuration (ase.Atoms): an slab atoms object with an
            adsorbate placed
        overlap_tag (int): Tag to check overlap with

    Returns:
        (bool): True if there is atomic overlap, otherwise False
    """
    ads_slab_config = adsorbate_slab_config.copy()
    mask = adsorbate_slab_config.get_tags() == overlap_tag
    adsorbate_atoms = adsorbate_slab_config[mask]
    adsorbate_com = adsorbate_atoms.get_center_of_mass()

    # wrap atoms so we dont have to worry about pbc
    cell_center = np.dot(np.array([0.5, 0.5, 0.5]), ads_slab_config.cell)
    ads_slab_config.translate(cell_center - adsorbate_com)
    ads_slab_config.wrap()

    adsorbate_atoms = ads_slab_config[mask]
    adsorbate_elements = adsorbate_atoms.get_chemical_symbols()

    mask = ads_slab_config.get_tags() != overlap_tag
    surface_atoms = ads_slab_config[mask]
    surface_elements = surface_atoms.get_chemical_symbols()

    pairs = list(product(range(len(surface_atoms)), range(len(adsorbate_atoms))))

    post_radial_distances = []
    for combo in pairs:
        total_distance = ads_slab_config.get_distance(
            combo[0], combo[1] + len(surface_elements), mic=True
        )
        total_distance = np.linalg.norm(
            adsorbate_atoms.positions[combo[1]] - surface_atoms.positions[combo[0]]
        )
        post_radial_distance = (
            total_distance
            - covalent_radii[atomic_numbers[surface_elements[combo[0]]]]
            - covalent_radii[atomic_numbers[adsorbate_elements[combo[1]]]]
        )
        post_radial_distances.append(post_radial_distance)
    return post_radial_distances


def there_is_overlap(adsorbate_slab_config: ase.Atoms, overlap_tag: int = 2):
    """
    Check to see if there is any atomic overlap between surface atoms
    and adsorbate atoms.

    Args:
        adsorbate_slab_configuration (ase.Atoms): an slab atoms object with an
            adsorbate placed
        overlap_tag (int): Tag to check overlap with

    Returns:
        (bool): True if there is atomic overlap, otherwise False
    """
    post_radial_distances = get_interstitial_distances(
        adsorbate_slab_config, overlap_tag
    )
    return not all(np.array(post_radial_distances) >= 0)
