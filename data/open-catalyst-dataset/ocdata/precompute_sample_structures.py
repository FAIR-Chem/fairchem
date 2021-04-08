'''
This submodule contains the scripts that the we used to sample the adsorption
structures.

Note that some of these scripts were taken from
[GASpy](https://github.com/ulissigroup/GASpy) with permission of author.
'''

__authors__ = ['Kevin Tran', 'Aini Palizhati', 'Siddharth Goyal', 'Zachary Ulissi']
__email__ = ['ktran@andrew.cmu.edu']

import math
from collections import defaultdict
import random
import pickle
import numpy as np
import catkit
import ase
import ase.db
from ase import neighborlist
from ase.constraints import FixAtoms
from ase.neighborlist import natural_cutoffs
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.surface import SlabGenerator, get_symmetrically_distinct_miller_indices
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import VoronoiNN
from .base_atoms.pkls import BULK_PKL, ADSORBATE_PKL
from .constants import MAX_MILLER
import sys
import time

def enumerate_surfaces_for_saving(bulk_atoms, max_miller=MAX_MILLER):
    '''
    Enumerate all the symmetrically distinct surfaces of a bulk structure. It
    will not enumerate surfaces with Miller indices above the `max_miller`
    argument. Note that we also look at the bottoms of surfaces if they are
    distinct from the top. If they are distinct, we flip the surface so the bottom
    is pointing upwards.

    Args:
        bulk_atoms  `ase.Atoms` object of the bulk you want to enumerate
                    surfaces from.
        max_miller  An integer indicating the maximum Miller index of the surfaces
                    you are willing to enumerate. Increasing this argument will
                    increase the number of surfaces, but the surfaces will
                    generally become larger.
    Returns:
        all_slabs_info  A list of 4-tuples containing:  `pymatgen.Structure`
                        objects for surfaces we have enumerated, the Miller
                        indices, floats for the shifts, and Booleans for "top".
    '''
    bulk_struct = standardize_bulk(bulk_atoms)

    all_slabs_info = []
    for millers in get_symmetrically_distinct_miller_indices(bulk_struct, MAX_MILLER):
        slab_gen = SlabGenerator(initial_structure=bulk_struct,
                                 miller_index=millers,
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
        flipped_slabs_info = [(flip_struct(slab), millers, slab.shift, False)
                              for slab in slabs if is_structure_invertible(slab) is False]

        # Concatenate all the results together
        slabs_info = [(slab, millers, slab.shift, True) for slab in slabs]
        all_slabs_info.extend(slabs_info + flipped_slabs_info)
    return all_slabs_info


def standardize_bulk(atoms):
    '''
    There are many ways to define a bulk unit cell. If you change the unit cell
    itself but also change the locations of the atoms within the unit cell, you
    can get effectively the same bulk structure. To address this, there is a
    standardization method used to reduce the degrees of freedom such that each
    unit cell only has one "true" configuration. This function will align a
    unit cell you give it to fit within this standardization.

    Arg:
        atoms   `ase.Atoms` object of the bulk you want to standardize
    Returns:
        standardized_struct     `pymatgen.Structure` of the standardized bulk
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
    Flips an atoms object upside down. Normally used to flip surfaces.

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

    
def precompute_enumerate_surface(bulk_database, bulk_index, opfile):

    with open(bulk_database, 'rb') as f:
        inv_index = pickle.load(f)
    flatten = inv_index[1] + inv_index[2] + inv_index[3]
    assert bulk_index < len(flatten)

    bulk, mpid = flatten[bulk_index] 

    print(bulk, mpid)
    surfaces_info = enumerate_surfaces_for_saving(bulk)

    with open(opfile, 'wb') as g:
        pickle.dump(surfaces_info, g)

s = time.time()
precompute_enumerate_surface(BULK_PKL, int(sys.argv[1]), sys.argv[2])
e = time.time()
print(sys.argv[1], "Done in", e - s )
