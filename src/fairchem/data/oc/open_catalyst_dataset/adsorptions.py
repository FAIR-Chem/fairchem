'''
This submodule contains the scripts that the we used to sample the adsorption
structures.

Note that some of these scripts were taken from
[GASpy](https://github.com/ulissigroup/GASpy) with permission of author.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

import warnings
import math
import random
import numpy as np
import ase
import ase.db
from ase.constraints import FixAtoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.surface import SlabGenerator, get_symmetrically_distinct_miller_indices
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.adsorption import AdsorbateSiteFinder


ELEMENTS = {1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
            9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
            16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti',
            23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29:
            'Cu', 30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br',
            36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42:
            'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd',
            49: 'In', 50: 'Sn', 51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55:
            'Cs', 56: 'Ba', 57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm',
            62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy', 67: 'Ho', 68:
            'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W',
            75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg', 81:
            'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn', 87: 'Fr',
            88: 'Ra', 89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U', 93: 'Np', 94:
            'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es', 100: 'Fm',
            101: 'Md', 102: 'No', 103: 'Lr', 104: 'Rf', 105: 'Db', 106: 'Sg',
            107: 'Bh', 108: 'Hs', 109: 'Mt', 110: 'Ds', 111: 'Rg', 112: 'Cn',
            113: 'Nh', 114: 'Fl', 115: 'Mc', 116: 'Lv', 117: 'Ts', 118: 'Og'}

# We will enumerate surfaces with Miller indices <= MAX_MILLER
MAX_MILLER = 2

# We will create surfaces that are at least MIN_XY Angstroms wide. GASpy uses
# 4.5, but our larger adsorbates here can be up to 3.6 Angstroms long. So 4.5 +
# 3.6 ~= 8 Angstroms
MIN_XY = 8.


def sample_structures(bulk_database='bulks.db', n_species_weights=None):
    '''
    '''
    # Choose which surface we want
    n_species = choose_n_species(n_species_weights)
    elements = choose_elements(bulk_database, n_species)
    bulk = choose_bulk(bulk_database, elements)
    surface = choose_surface(bulk)

    # Choose the adsorbate and place it on the surface
    adsorbate = choose_adsorbate()
    site = choose_site(surface, adsorbate)
    adsorbed_surface = add_adsorbate_onto_surface(surface, adsorbate, site)

    # Add appropriate constraints
    adsorbed_surface = constrain_surface(adsorbed_surface)
    surface = constrain_surface(surface)
    return adsorbed_surface, surface


def choose_n_species(n_species_weights):
    '''
    '''
    if n_species_weights is None:
        n_species_weights = {1: 0.05, 2: 0.65, 3: 0.3}

    n_species = list(n_species_weights.keys())
    weights = list(n_species_weights.values())
    assert math.isclose(sum(weights), 1)

    n_species = np.random.choice(n_species, p=weights)
    return n_species


def choose_elements(bulk_database, n):
    '''
    '''
    db = ase.db.connect(bulk_database)
    all_elements = {ELEMENTS[number] for row in db.select() for number in row.numbers}
    elements = random.sample(all_elements, n)

    # Make sure we choose a combination of elements that exists in our bulk
    # database
    while db.count(elements) == 0:
        warnings.warn('Sampled the elements %s, but could not find any matching '
                      'bulks in the database (%s). Trying to re-sample'
                      % (elements, bulk_database), RuntimeWarning)
        elements = random.sample(all_elements, n)

    return elements


def choose_bulk(bulk_database, elements):
    '''
    '''
    db = ase.db.connect(bulk_database)
    all_atoms = [row.toatoms() for row in db.select(elements)]
    atoms = random.choice(all_atoms)
    return atoms


def choose_surface(bulk_atoms):
    '''
    '''
    surfaces = enumerate_surfaces(bulk_atoms)
    surface_struct = random.choice(surfaces)
    surface_atoms = AseAtomsAdaptor.get_atoms(surface_struct)
    return surface_atoms


def enumerate_surfaces(bulk_atoms):
    '''
    '''
    bulk_struct = standardize_bulk(bulk_atoms)

    all_slabs = []
    for miller_indices in get_symmetrically_distinct_miller_indices(bulk_struct, MAX_MILLER):
        slab_gen = SlabGenerator(initial_structure=bulk_struct,
                                 miller_index=miller_indices,
                                 min_slab_size=7.,
                                 min_vacuum_size=20.,
                                 lll_reduce=False,
                                 center_slab=True,
                                 primitive=True,
                                 max_normal_search=1)
        slabs = slab_gen.get_slabs(tol=0.3,
                                   bonds=None,
                                   max_broken_bonds=0,
                                   symmetrize=False)

        # If the bottoms of the slabs are different than the tops, then we want
        # to consider them, too
        flipped_slabs = [flip_struct(slab) for slab in slabs
                         if is_structure_invertible(slab) is False]
        slabs.extend(flipped_slabs)

        all_slabs.extend(slabs)
    return all_slabs


def standardize_bulk(atoms):
    '''
    '''
    struct = AseAtomsAdaptor.get_structure(atoms)
    sga = SpacegroupAnalyzer(struct, symprec=0.1)
    standardized_struct = sga.get_conventional_standard_structure()
    return standardized_struct


def is_structure_invertible(structure):
    '''
    This function figures out whether or not an `pymatgen.Structure` object has
    symmetricity. In this function, the affine matrix is a rotation matrix that
    is multiplied with the XYZ positions of the crystal. If the z,z component
    of that is negative, it means symmetry operation exist, it could be a
    mirror operation, or one that involves multiple rotations/etc. Regardless,
    it means that the top becomes the bottom and vice-versa, and the structure
    is the symmetric. i.e. structure_XYZ = structure_XYZ*M.

    In short:  If this function returns `False`, then the input structure can
    be flipped in the z-direction to create a new structure.

    Arg:
        structure   A `pymatgen.Structure` object.
    Returns
        A boolean indicating whether or not your `ase.Atoms` object is
        symmetric in z-direction (i.e. symmetric with respect to x-y plane).
    '''
    # If any of the operations involve a transformation in the z-direction,
    # then the structure is invertible.
    sga = SpacegroupAnalyzer(structure, symprec=0.1)
    for operation in sga.get_symmetry_operations():
        xform_matrix = operation.affine_matrix
        z_xform = xform_matrix[2, 2]
        if z_xform == -1:
            return True
    return False


def flip_struct(struct):
    '''
    Flips an atoms object upside down. Normally used to flip slabs.

    Arg:
        atoms   `pymatgen.Structure` object
    Returns:
        flipped_struct  The same `ase.Atoms` object that was fed as an
                        argument, but flipped upside down.
    '''
    atoms = AseAtomsAdaptor.get_atoms(struct)

    # This is black magic wizardry to me. Good look figuring it out.
    atoms.wrap()
    atoms.rotate(180, 'x', rotate_cell=True, center='COM')
    if atoms.cell[2][2] < 0.:
        atoms.cell[2] = -atoms.cell[2]
    if np.cross(atoms.cell[0], atoms.cell[1])[2] < 0.0:
        atoms.cell[1] = -atoms.cell[1]
    atoms.wrap()

    flipped_struct = AseAtomsAdaptor.get_structure(atoms)
    return flipped_struct


def choose_site(surface_atoms, adsorbate):
    '''
    '''
    # TODO:  Pari to update this section and add bidentate site selection
    surface_atoms = tile_atoms(surface_atoms)
    sites = enumerate_adsorption_sites(surface_atoms)
    site = random.choice(sites)
    return site


def tile_atoms(atoms):
    '''
    This function will repeat an atoms structure in the x and y direction until
    the x and y dimensions are at least as wide as the MIN_XY constant.

    Args:
        atoms   `ase.Atoms` object of the structure that you want to tile
    Returns:
        atoms_tiled     An `ase.Atoms` object that's just a tiled version of
                        the `atoms` argument.
    '''
    x_length = np.linalg.norm(atoms.cell[0])
    y_length = np.linalg.norm(atoms.cell[1])
    nx = int(math.ceil(MIN_XY/x_length))
    ny = int(math.ceil(MIN_XY/y_length))
    n_xyz = (nx, ny, 1)
    atoms_tiled = atoms.repeat(n_xyz)
    return atoms_tiled


def enumerate_adsorption_sites(atoms):
    '''
    A wrapper for pymatgen to get all of the adsorption sites of a slab.

    Arg:
        atoms   The slab where you are trying to find adsorption sites in
                `ase.Atoms` format
    Output:
        sites   A `numpy.ndarray` object that contains the x-y-z coordinates of
                the adsorptions sites
    '''
    struct = AseAtomsAdaptor.get_structure(atoms)
    sites_dict = AdsorbateSiteFinder(struct).find_adsorption_sites(put_inside=True)
    sites = sites_dict['all']
    return sites


def choose_adsorbate():
    '''
    '''
    # TODO:  Kevin to add the real adsorbates here
    CO = ase.Atoms('CO', positions=[[0., 0., 0.], [0., 0., 1.2]])
    return CO


def add_adsorbate_onto_surface(surface, adsorbate, site):
    '''
    There are a lot of small details that need to be considered when adding an
    adsorbate onto a surface. This function will take care of those details for
    you.

    Args:
        surface     An `ase.Atoms` object of the surface
        adsorbate   An `ase.Atoms` object of the adsorbate
        site        A 3-long sequence containing floats that indicate the
                    cartesian coordinates of the site you want to add the
                    adsorbate onto.
    Returns:
        ads_surface     An `ase.Atoms` object containing the adsorbate and
                        surface. Slab atoms will be tagged with a `0` and
                        adsorbate atoms will be tagged with a `1`.
    '''
    adsorbate = adsorbate.copy()  # To make sure we don't mess with the original
    adsorbate.translate(site)
    # TODO:  Pari to incorporate the adsorption vector changes around here

    ads_surface = adsorbate + surface
    ads_surface.cell = surface.cell
    ads_surface.pbc = [True, True, True]

    # We set the tags of surface atoms to 0, and set the tags of the adsorbate to 1.
    tags = [1] * len(adsorbate)
    tags.extend([0] * len(surface))
    ads_surface.set_tags(tags)
    return ads_surface


def constrain_surface(atoms, z_cutoff=3.):
    '''
    This function fixes sub-surface atoms of a surface. Also works on systems that
    have surface + adsorbate(s), as long as the surface atoms are tagged with `0`
    and the adsorbate atoms are tagged with positive integers.

    Inputs:
        atoms       `ase.Atoms` class of the surface system. The tags of these
                    atoms must be set such that any surface atom is tagged with
                    `0`, and any adsorbate atom is tagged with a positive
                    integer.
        z_cutoff    The threshold to see if surface atoms are in the same plane
                    as the highest atom in the surface
    Returns:
        atoms   A deep copy of the `atoms` argument, but where the appropriate
                atoms are constrained
    '''
    # Work on a copy so that we don't modify the original
    atoms = atoms.copy()

    # We'll be making a `mask` list to feed to the `FixAtoms` class. This list
    # should contain a `True` if we want an atom to be constrained, and `False`
    # otherwise
    mask = []

    # If we assume that the third component of the unit cell lattice is
    # orthogonal to the slab surface, then atoms with higher values in the
    # third coordinate of their scaled positions are higher in the slab. We
    # make this assumption here, which means that we will be working with
    # scaled positions instead of Cartesian ones.
    scaled_positions = atoms.get_scaled_positions()
    unit_cell_height = np.linalg.norm(atoms.cell[2])

    # If the slab is pointing upwards, then fix atoms that are below the
    # threshold
    if atoms.cell[2, 2] > 0:
        max_height = max(position[2] for position, atom in zip(scaled_positions, atoms)
                         if atom.tag == 0)
        threshold = max_height - z_cutoff / unit_cell_height
        for position, atom in zip(scaled_positions, atoms):
            if atom.tag == 0 and position[2] < threshold:
                mask.append(True)
            else:
                mask.append(False)

    # If the slab is pointing downwards, then fix atoms that are above the
    # threshold
    elif atoms.cell[2, 2] < 0:
        min_height = min(position[2] for position, atom in zip(scaled_positions, atoms)
                         if atom.tag == 0)
        threshold = min_height + z_cutoff / unit_cell_height
        for position, atom in zip(scaled_positions, atoms):
            if atom.tag == 0 and position[2] > threshold:
                mask.append(True)
            else:
                mask.append(False)

    else:
        raise RuntimeError('Tried to constrain a slab that points in neither '
                           'the positive nor negative z directions, so we do '
                           'not know which side to fix')

    atoms.constraints += [FixAtoms(mask=mask)]
    return atoms
