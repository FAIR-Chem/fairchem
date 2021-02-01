
import math
import numpy as np
import os
import pickle

from pymatgen.core.surface import SlabGenerator, get_symmetrically_distinct_miller_indices
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from .constants import MAX_MILLER


class Bulk():
    '''
    This class handles all things with the bulk.
    It also provides possible surfaces, later used to create a Surface object.
    '''
    def __init__(self, bulk_database, precomputed_structures=None, bulk_index=None, max_elems=3):
        self.precomputed_structures = precomputed_structures
        self.choose_bulk_pkl(bulk_database, bulk_index, max_elems)

    def choose_bulk_pkl(self, bulk_db, bulk_index, max_elems):
        '''
        Chooses a bulk from our pkl file at random as long as the bulk contains
        the specified number of elements in any composition.

        Args:
            bulk_db         Unpickled dict or list of bulks
            bulk_index      Index of which bulk to select. If None, randomly sample one.
            max_elems       Max elems for any bulk structure. Currently it is 3 by default.

        Sets as class variables:
            bulk_atoms                  `ase.Atoms` of the chosen bulk structure.
            mpid                        A string indicating which MPID the bulk is
            bulk_sampling_str           A string to enumerate the sampled structure
            index_of_bulk_atoms         Index of the chosen bulk in the array (should match
                                        bulk_index if provided)
        '''

        try:
            if bulk_index is not None:
                assert len(bulk_db) > max_elems, f'Bulk db only has {len(bulk_db)} entries. Did you pass in the correct bulk database?'
                assert isinstance(bulk_db[bulk_index], tuple)

                self.bulk_atoms, self.mpid, self.bulk_sampling_str, self.index_of_bulk_atoms = bulk_db[bulk_index]
                self.n_elems = len(set(self.bulk_atoms.symbols)) # 1, 2, or 3
                self.elem_sampling_str = f'{self.n_elems}/{max_elems}'

            else:
                self.sample_n_elems()
                assert isinstance(bulk_db, dict), 'Did you pass in the correct bulk database?'
                assert self.n_elems in bulk_db.keys(), f'Bulk db does not have bulks of {self.n_elems} elements'
                assert isinstance(bulk_db[self.n_elems], list), 'Did you pass in the correct bulk database?'

                total_elements_for_key = len(bulk_db[self.n_elems])
                row_bulk_index = np.random.choice(total_elements_for_key)
                self.bulk_atoms, self.mpid, self.bulk_sampling_str, self.index_of_bulk_atoms = bulk_db[self.n_elems][row_bulk_index]

        except IndexError:
            raise ValueError('Randomly chose to look for a %i-component material, '
                             'but no such materials exist. Please add one '
                             'to the database or change the weights to exclude '
                             'this number of components.'
                             % self.n_elems)

    def sample_n_elems(self, n_cat_elems_weights={1: 0.05, 2: 0.65, 3: 0.3}): # TODO make these weights an input param?
        '''
        Chooses the number of species we should look for in this sample.

        Arg:
            n_cat_elems_weights A dictionary whose keys are integers containing the
                                number of species you want to consider and whose
                                values are the probabilities of selecting this
                                number. The probabilities must sum to 1.
        Sets:
            n_elems             An integer showing how many species have been chosen.
            elem_sampling_str     Enum string of [chosen n_elems]/[total number of choices]
        '''

        possible_n_elems = list(n_cat_elems_weights.keys())
        weights = list(n_cat_elems_weights.values())
        assert math.isclose(sum(weights), 1)

        self.n_elems = np.random.choice(possible_n_elems, p=weights)
        self.elem_sampling_str = str(self.n_elems) + "/" + str(len(possible_n_elems))

    def get_possible_surfaces(self):
        # returns a list of possible surfaces for this bulk instance.
        # this can be used to iterate through all surfaces, or select one at random, to make a Surface object.
        if self.precomputed_structures:
            surfaces_info = self.read_from_precomputed_enumerations(self.index_of_bulk_atoms)
        else:
            surfaces_info = self.enumerate_surfaces()
        return surfaces_info

    def read_from_precomputed_enumerations(self, index):
        with open(os.path.join(self.precomputed_structures, str(index) + ".pkl"), "rb") as f:
            surfaces_info = pickle.load(f)
        return surfaces_info

    def enumerate_surfaces(self, max_miller=MAX_MILLER):
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
        bulk_struct = self.standardize_bulk(self.bulk_atoms)

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
            flipped_slabs_info = [(self.flip_struct(slab), millers, slab.shift, False)
                                  for slab in slabs if self.is_structure_invertible(slab) is False]

            # Concatenate all the results together
            slabs_info = [(slab, millers, slab.shift, True) for slab in slabs]
            all_slabs_info.extend(slabs_info + flipped_slabs_info)
        return all_slabs_info

    def standardize_bulk(self, atoms):
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

    def flip_struct(self, struct):
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

    def is_structure_invertible(self, structure):
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
