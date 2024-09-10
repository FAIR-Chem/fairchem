from __future__ import annotations

import math
import os
import pickle
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
from ase.constraints import FixAtoms
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.core.composition import Composition
from pymatgen.core.surface import (
    SlabGenerator,
    get_symmetrically_distinct_miller_indices,
)
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

if TYPE_CHECKING:
    import ase
    from pymatgen.core.structure import Structure


class Slab:
    """
    Initializes a slab object, i.e. a particular slab tiled along xyz, in
    one of 2 ways:
    - Pass in a Bulk object and a slab 5-tuple containing
    (atoms, miller, shift, top, oriented bulk).
    - Pass in a Bulk object and randomly sample a slab.

    Arguments
    ---------
    bulk: Bulk
        Corresponding Bulk object.
    slab_atoms: ase.Atoms
        Slab atoms, tiled and tagged
    millers: tuple
        Miller indices of slab.
    shift: float
        Shift of slab.
    top: bool
        Whether slab is top or bottom.
    min_ab: float
        To confirm that the tiled structure spans this distance
    """

    def __init__(
        self,
        bulk=None,
        slab_atoms: ase.Atoms = None,
        millers: tuple | None = None,
        shift: float | None = None,
        top: bool | None = None,
        oriented_bulk: Structure = None,
        min_ab: float = 8.0,
    ):
        assert bulk is not None
        self.bulk = bulk

        self.atoms = slab_atoms
        self.millers = millers
        self.shift = shift
        self.top = top
        self.oriented_bulk = oriented_bulk

        assert (
            Composition(self.atoms.get_chemical_formula()).reduced_formula
            == Composition(bulk.atoms.get_chemical_formula()).reduced_formula
        ), "Mismatched bulk and surface"
        assert np.linalg.norm(self.atoms.cell[0]) >= min_ab, "Slab not tiled"
        assert np.linalg.norm(self.atoms.cell[1]) >= min_ab, "Slab not tiled"
        assert self.has_surface_tagged(), "Slab not tagged"
        assert len(self.atoms.constraints) > 0, "Sub-surface atoms not constrained"

    @classmethod
    def from_bulk_get_random_slab(
        cls, bulk=None, max_miller=2, min_ab=8.0, save_path=None
    ):
        if save_path is not None:
            all_slabs = Slab.from_bulk_get_all_slabs(
                bulk, max_miller, min_ab, save_path
            )
            slab_idx = np.random.randint(len(all_slabs))
            return all_slabs[slab_idx]
        else:
            # If we're not saving all slabs, just tile and tag one
            assert bulk is not None
            bulk_struct = standardize_bulk(bulk.atoms)
            avail_millers = get_symmetrically_distinct_miller_indices(
                bulk_struct, max_miller
            )
            sampled_miller_idx = np.random.randint(len(avail_millers))
            slabs = Slab.from_bulk_get_specific_millers(
                avail_millers[sampled_miller_idx], bulk, min_ab
            )

            # If multiple shifts exist, sample one randomly
            return np.random.choice(slabs)

    @classmethod
    def from_bulk_get_specific_millers(
        cls, specific_millers, bulk=None, min_ab=8.0, save_path=None
    ):
        assert isinstance(specific_millers, tuple)
        assert len(specific_millers) == 3

        if save_path is not None:
            all_slabs = Slab.from_bulk_get_all_slabs(
                bulk, max(np.abs(specific_millers)), min_ab, save_path
            )
            return [slab for slab in all_slabs if slab.millers == specific_millers]
        else:
            # If we're not saving all slabs, just tile and tag those with correct millers
            assert bulk is not None
            untiled_slabs = compute_slabs(
                bulk.atoms,
                max_miller=max(np.abs(specific_millers)),
                specific_millers=[specific_millers],
            )
            slabs = [
                (
                    tile_and_tag_atoms(s[0], bulk.atoms, min_ab=min_ab),
                    s[1],
                    s[2],
                    s[3],
                    s[4],
                )
                for s in untiled_slabs
            ]
            return [cls(bulk, s[0], s[1], s[2], s[3], s[4]) for s in slabs]

    @classmethod
    def from_bulk_get_all_slabs(
        cls, bulk=None, max_miller=2, min_ab=8.0, save_path=None
    ):
        assert bulk is not None

        untiled_slabs = compute_slabs(
            bulk.atoms,
            max_miller=max_miller,
        )
        slabs = [
            (
                tile_and_tag_atoms(s[0], bulk.atoms, min_ab=min_ab),
                s[1],
                s[2],
                s[3],
                s[4],
            )
            for s in untiled_slabs
        ]

        # if path is provided, save out the pkl
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump(slabs, f)

        return [cls(bulk, s[0], s[1], s[2], s[3], s[4]) for s in slabs]

    @classmethod
    def from_precomputed_slabs_pkl(
        cls,
        bulk=None,
        precomputed_slabs_pkl=None,
        max_miller=2,
        min_ab=8.0,
    ):
        assert bulk is not None
        assert precomputed_slabs_pkl is not None
        assert os.path.exists(precomputed_slabs_pkl)

        with open(precomputed_slabs_pkl, "rb") as fp:
            slabs = pickle.load(fp)

        is_slab_obj = np.all([isinstance(s, Slab) for s in slabs])
        if is_slab_obj:
            assert np.all(np.array([s.millers for s in slabs]) <= max_miller)
            return slabs
        else:
            assert np.all(np.array([s[1] for s in slabs]) <= max_miller)
            return [cls(bulk, s[0], s[1], s[2], s[3], s[4]) for s in slabs]

    @classmethod
    def from_atoms(cls, atoms: ase.Atoms = None, bulk=None, **kwargs):
        assert atoms is not None
        return cls(bulk, AseAtomsAdaptor.get_structure(atoms), **kwargs)

    def has_surface_tagged(self):
        return np.any(self.atoms.get_tags() == 1)

    def get_metadata_dict(self):
        # Returns a dict containing the atoms object and metadata,
        # used for writing to files.
        return {
            "slab_atomsobject": self.atoms,
            "slab_metadata": {
                "bulk_id": self.bulk.src_id,
                "millers": self.millers,
                "shift": self.shift,
                "top": self.top,
                "oriented_bulk": self.oriented_bulk,
            },
        }

    def __len__(self):
        return len(self.atoms)

    def __str__(self):
        return f"Slab: ({self.atoms.get_chemical_formula()}, {self.millers}, {self.shift}, {self.top})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return (
            self.atoms == other.atoms
            and self.millers == other.millers
            and self.shift == other.shift
            and self.top == other.top
            and self.oriented_bulk == other.oriented_bulk
        )


def tile_and_tag_atoms(
    unit_slab_struct: Structure, bulk_atoms: ase.Atoms, min_ab: float = 8
):
    """
    This function combines the next three functions that tile, tag,
    and constrain the atoms.

    Arguments
    ---------
    unit_slab_struct: Structure
        The untiled slab structure
    bulk_atoms: ase.Atoms
        Atoms of the corresponding bulk structure, used for tagging
    min_ab: float
        The minimum distance in x and y spanned by the tiled structure.

    Returns
    -------
    atoms_tiled: ase.Atoms
        A copy of the slab atoms that is tiled, tagged, and constrained
    """
    slab_atoms = AseAtomsAdaptor.get_atoms(unit_slab_struct)
    return set_fixed_atom_constraints(
        tile_atoms(tag_surface_atoms(slab_atoms, bulk_atoms), min_ab=min_ab)
    )


def set_fixed_atom_constraints(atoms):
    """
    This function fixes sub-surface atoms of a surface. Also works on systems
    that have surface + adsorbate(s), as long as the bulk atoms are tagged with
    `0`, surface atoms are tagged with `1`, and the adsorbate atoms are tagged
    with `2` or above.

    This is used for both surface atoms and the combined surface+adsorbate.

    Arguments
    ---------
    atoms: ase.Atoms
        Atoms object of the slab or slab+adsorbate system, with bulk atoms
        tagged as `0`, surface atoms tagged as `1`, and adsorbate atoms tagged
        as `2` or above.

    Returns
    -------
    atoms: ase.Atoms
        A deep copy of the `atoms` argument, but where the appropriate
        atoms are constrained.
    """
    # We'll be making a `mask` list to feed to the `FixAtoms` class. This
    # list should contain a `True` if we want an atom to be constrained, and
    # `False` otherwise.
    atoms = atoms.copy()
    mask = [atom.tag == 0 for atom in atoms]
    atoms.constraints += [FixAtoms(mask=mask)]
    return atoms


def tag_surface_atoms(
    slab_atoms: ase.Atoms = None,
    bulk_atoms: ase.Atoms = None,
):
    """
    Sets the tags of an `ase.Atoms` object. Any atom that we consider a "bulk"
    atom will have a tag of 0, and any atom that we consider a "surface" atom
    will have a tag of 1. We use a combination of Voronoi neighbor algorithms
    (adapted from `pymatgen.core.surface.Slab.get_surface_sites`; see
    https://pymatgen.org/pymatgen.core.surface.html) and a distance cutoff.

    Arguments
    ---------
    slab_atoms: ase.Atoms
        The slab where you are trying to find surface sites.
    bulk_atoms: ase.Atoms
        The bulk structure that the surface was cut from.

    Returns
    -------
    slab_atoms: ase.Atoms
        A copy of the slab atoms with the surface atoms tagged as 1.
    """
    assert slab_atoms is not None
    slab_atoms = slab_atoms.copy()

    height_tags = find_surface_atoms_by_height(slab_atoms)

    if bulk_atoms is not None:
        tags = find_surface_atoms_with_voronoi_given_height(
            bulk_atoms, slab_atoms, height_tags
        )
    else:
        tags = height_tags

    slab_atoms.set_tags(tags)

    return slab_atoms


def tile_atoms(atoms: ase.Atoms, min_ab: float = 8):
    """
    This function will repeat an atoms structure in the direction of the a and b
    lattice vectors such that they are at least as wide as the min_ab constant.

    Arguments
    ---------
    atoms: ase.Atoms
        The structure to tile.
    min_ab: float
        The minimum distance in x and y spanned by the tiled structure.

    Returns
    -------
    atoms_tiled: ase.Atoms
        The tiled structure.
    """
    a_length = np.linalg.norm(atoms.cell[0])
    b_length = np.linalg.norm(atoms.cell[1])
    na = int(math.ceil(min_ab / a_length))
    nb = int(math.ceil(min_ab / b_length))
    n_abc = (na, nb, 1)
    return atoms.repeat(n_abc)


def find_surface_atoms_by_height(surface_atoms):
    """
    As discussed in the docstring for `find_surface_atoms_with_voronoi`,
    sometimes we might accidentally tag a surface atom as a bulk atom if there
    are multiple coordination environments for that atom type within the bulk.
    One heuristic that we use to address this is to simply figure out if an
    atom is close to the surface. This function will figure that out.

    Specifically:  We consider an atom a surface atom if it is within 2
    Angstroms of the heighest atom in the z-direction (or more accurately, the
    direction of the 3rd unit cell vector).

    Arguments
    ---------
    surface_atoms: ase.Atoms

    Returns
    -------
    tags: list
        A list that contains the indices of the surface atoms.
    """
    unit_cell_height = np.linalg.norm(surface_atoms.cell[2])
    scaled_positions = surface_atoms.get_scaled_positions()
    scaled_max_height = max(scaled_position[2] for scaled_position in scaled_positions)
    scaled_threshold = scaled_max_height - 2.0 / unit_cell_height

    return [
        0 if scaled_position[2] < scaled_threshold else 1
        for scaled_position in scaled_positions
    ]


def find_surface_atoms_with_voronoi_given_height(bulk_atoms, slab_atoms, height_tags):
    """
    Labels atoms as surface or bulk atoms according to their coordination
    relative to their bulk structure. If an atom's coordination is less than it
    normally is in a bulk, then we consider it a surface atom. We calculate the
    coordination using pymatgen's Voronoi algorithms.

    Note that if a single element has different sites within a bulk and these
    sites have different coordinations, then we consider slab atoms
    "under-coordinated" only if they are less coordinated than the most under
    undercoordinated bulk atom. For example:  Say we have a bulk with two Cu
    sites. One site has a coordination of 12 and another a coordination of 9.
    If a slab atom has a coordination of 10, we will consider it a bulk atom.

    Arguments
    ---------
    bulk_atoms: ase.Atoms
        The bulk structure that the surface was cut from.
    slab_atoms: ase.Atoms
        The slab structure.
    height_tags: list
        The tags determined by the `find_surface_atoms_by_height` algo.

    Returns
    -------
    tags: list
        A list of 0s and 1s whose indices align with the atoms in
        `slab_atoms`. 0s indicate a bulk atom and 1 indicates a surface atom.
    """
    # Initializations
    surface_struct = AseAtomsAdaptor.get_structure(slab_atoms)
    center_of_mass = calculate_center_of_mass(surface_struct)
    bulk_cn_dict = calculate_coordination_of_bulk_atoms(bulk_atoms)

    voronoi_nn = VoronoiNN(tol=0.1)  # 0.1 chosen for better detection
    tags = height_tags
    for idx, site in enumerate(surface_struct):
        if height_tags[idx] == 1 or site.frac_coords[2] < center_of_mass[2]:
            continue
        try:
            # Tag as surface if atom is under-coordinated
            cn = voronoi_nn.get_cn(surface_struct, idx, use_weights=True)
            cn = round(cn, 5)
            if cn < min(bulk_cn_dict[site.species_string]):
                tags[idx] = 1

        # Tag as surface if we get a pathological error
        except RuntimeError:
            tags[idx] = 1

    return tags


def calculate_center_of_mass(struct):
    """
    Calculates the center of mass of the slab.
    """
    weights = [site.species.weight for site in struct]
    return np.average(struct.frac_coords, weights=weights, axis=0)


def calculate_coordination_of_bulk_atoms(bulk_atoms):
    """
    Finds all unique atoms in a bulk structure and then determines their
    coordination number. Then parses these coordination numbers into a
    dictionary whose keys are the elements of the atoms and whose values are
    their possible coordination numbers.
    For example: `bulk_cns = {'Pt': {3., 12.}, 'Pd': {12.}}`

    Arguments
    ---------
    bulk_atoms: ase.Atoms
        The bulk structure.

    Returns
    -------
    bulk_cn_dict: dict
        A dictionary whose keys are the elements of the atoms and whose values
        are their possible coordination numbers.
    """
    voronoi_nn = VoronoiNN(tol=0.1)  # 0.1 chosen for better detection

    # Object type conversion so we can use Voronoi
    bulk_struct = AseAtomsAdaptor.get_structure(bulk_atoms)
    sga = SpacegroupAnalyzer(bulk_struct)
    sym_struct = sga.get_symmetrized_structure()

    # We'll only loop over the symmetrically distinct sites for speed's sake
    bulk_cn_dict = defaultdict(set)
    for idx in sym_struct.equivalent_indices:
        site = sym_struct[idx[0]]
        cn = voronoi_nn.get_cn(sym_struct, idx[0], use_weights=True)
        cn = round(cn, 5)
        bulk_cn_dict[site.species_string].add(cn)
    return bulk_cn_dict


def compute_slabs(
    bulk_atoms: ase.Atoms = None,
    max_miller: int = 2,
    specific_millers: list | None = None,
):
    """
    Enumerates all the symmetrically distinct slabs of a bulk structure.
    It will not enumerate slabs with Miller indices above the
    `max_miller` argument. Note that we also look at the bottoms of slabs
    if they are distinct from the top. If they are distinct, we flip the
    surface so the bottom is pointing upwards.

    Arguments
    ---------
    bulk_atoms: ase.Atoms
        The bulk structure.
    max_miller: int
        The maximum Miller index of the slabs to enumerate. Increasing this
        argument will increase the number of slabs, and the slabs will generally
        become larger.
    specific_millers: list
        A list of Miller indices that you want to enumerate. If this argument
        is not `None`, then the `max_miller` argument is ignored.

    Returns
    -------
    all_slabs_info: list
        A list of 5-tuples containing pymatgen structure objects for enumerated
        slabs, the Miller indices, floats for the shifts, booleans for top, and
        the oriented bulk structure.
    """
    assert bulk_atoms is not None
    bulk_struct = standardize_bulk(bulk_atoms)

    if specific_millers is None:
        specific_millers = get_symmetrically_distinct_miller_indices(
            bulk_struct, max_miller
        )

    all_slabs_info = []
    for millers in specific_millers:
        slab_gen = SlabGenerator(
            initial_structure=bulk_struct,
            miller_index=millers,
            min_slab_size=7.0,
            min_vacuum_size=20.0,
            lll_reduce=False,
            center_slab=True,
            primitive=True,
            max_normal_search=1,
        )
        slabs = slab_gen.get_slabs(
            tol=0.3, bonds=None, max_broken_bonds=0, symmetrize=False
        )

        # If the bottom of the slabs are different than the tops, then we
        # want to consider them too.
        if len(slabs) != 0:
            flipped_slabs_info = [
                (flip_struct(slab), millers, slab.shift, False, slab.oriented_unit_cell)
                for slab in slabs
                if is_structure_invertible(slab) is False
            ]

            # Concatenate all the results together
            slabs_info = [
                (slab, millers, slab.shift, True, slab.oriented_unit_cell)
                for slab in slabs
            ]
            all_slabs_info.extend(slabs_info + flipped_slabs_info)

    return all_slabs_info


def flip_struct(struct: Structure):
    """
    Flips an atoms object upside down. Normally used to flip slabs.

    Arguments
    ---------
    struct: Structure
        pymatgen structure object of the surface you want to flip

    Returns
    -------
    flipped_struct: Structure
        pymatgen structure object of the flipped surface.
    """
    atoms = AseAtomsAdaptor.get_atoms(struct)

    # This is black magic wizardry to me. Good look figuring it out.
    atoms.wrap()
    atoms.rotate(180, "x", rotate_cell=True, center="COM")
    if atoms.cell[2][2] < 0.0:
        atoms.cell[2] = -atoms.cell[2]
    if np.cross(atoms.cell[0], atoms.cell[1])[2] < 0.0:
        atoms.cell[1] = -atoms.cell[1]
    atoms.center()
    atoms.wrap()

    return AseAtomsAdaptor.get_structure(atoms)


def is_structure_invertible(struct: Structure):
    """
    This function figures out whether or not an `Structure`
    object has symmetricity. In this function, the affine matrix is a rotation
    matrix that is multiplied with the XYZ positions of the crystal. If the z,z
    component of that is negative, it means symmetry operation exist, it could
    be a mirror operation, or one that involves multiple rotations/etc.
    Regardless, it means that the top becomes the bottom and vice-versa, and the
    structure is the symmetric. i.e. structure_XYZ = structure_XYZ*M.

    In short:  If this function returns `False`, then the input structure can
    be flipped in the z-direction to create a new structure.

    Arguments
    ---------
    struct: Structure
        pymatgen structure object of the slab.

    Returns
    -------
        A boolean indicating whether or not your `ase.Atoms` object is
        symmetric in z-direction (i.e. symmetric with respect to x-y plane).
    """
    # If any of the operations involve a transformation in the z-direction,
    # then the structure is invertible.
    sga = SpacegroupAnalyzer(struct, symprec=0.1)
    for operation in sga.get_symmetry_operations():
        xform_matrix = operation.affine_matrix
        z_xform = xform_matrix[2, 2]
        if z_xform == -1:
            return True
    return False


def standardize_bulk(atoms: ase.Atoms):
    """
    There are many ways to define a bulk unit cell. If you change the unit
    cell itself but also change the locations of the atoms within the unit
    cell, you can effectively get the same bulk structure. To address this,
    there is a standardization method used to reduce the degrees of freedom
    such that each unit cell only has one "true" configuration. This
    function will align a unit cell you give it to fit within this
    standardization.

    Arguments
    ---------
    atoms: ase.Atoms
        `ase.Atoms` object of the bulk you want to standardize.

    Returns
    -------
    standardized_struct: Structure
        pymatgen structure object of the standardized bulk.
    """
    struct = AseAtomsAdaptor.get_structure(atoms)
    sga = SpacegroupAnalyzer(struct, symprec=0.1)
    return sga.get_conventional_standard_structure()
