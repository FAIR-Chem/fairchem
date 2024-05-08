:py:mod:`data.oc.scripts.precompute_sample_structures`
======================================================

.. py:module:: data.oc.scripts.precompute_sample_structures

.. autoapi-nested-parse::

   This submodule contains the scripts that the we used to sample the adsorption
   structures.

   Note that some of these scripts were taken from
   [GASpy](https://github.com/ulissigroup/GASpy) with permission of author.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   data.oc.scripts.precompute_sample_structures.enumerate_surfaces_for_saving
   data.oc.scripts.precompute_sample_structures.standardize_bulk
   data.oc.scripts.precompute_sample_structures.is_structure_invertible
   data.oc.scripts.precompute_sample_structures.flip_struct
   data.oc.scripts.precompute_sample_structures.precompute_enumerate_surface



Attributes
~~~~~~~~~~

.. autoapisummary::

   data.oc.scripts.precompute_sample_structures.__authors__
   data.oc.scripts.precompute_sample_structures.__email__
   data.oc.scripts.precompute_sample_structures.s


.. py:data:: __authors__
   :value: ['Kevin Tran', 'Aini Palizhati', 'Siddharth Goyal', 'Zachary Ulissi']

   

.. py:data:: __email__
   :value: ['ktran@andrew.cmu.edu']

   

.. py:function:: enumerate_surfaces_for_saving(bulk_atoms, max_miller=MAX_MILLER)

   Enumerate all the symmetrically distinct surfaces of a bulk structure. It
   will not enumerate surfaces with Miller indices above the `max_miller`
   argument. Note that we also look at the bottoms of surfaces if they are
   distinct from the top. If they are distinct, we flip the surface so the bottom
   is pointing upwards.

   :param bulk_atoms  `ase.Atoms` object of the bulk you want to enumerate: surfaces from.
   :param max_miller  An integer indicating the maximum Miller index of the surfaces: you are willing to enumerate. Increasing this argument will
                                                                                      increase the number of surfaces, but the surfaces will
                                                                                      generally become larger.

   :returns:

             `pymatgen.Structure`
                             objects for surfaces we have enumerated, the Miller
                             indices, floats for the shifts, and Booleans for "top".
   :rtype: all_slabs_info  A list of 4-tuples containing


.. py:function:: standardize_bulk(atoms)

   There are many ways to define a bulk unit cell. If you change the unit cell
   itself but also change the locations of the atoms within the unit cell, you
   can get effectively the same bulk structure. To address this, there is a
   standardization method used to reduce the degrees of freedom such that each
   unit cell only has one "true" configuration. This function will align a
   unit cell you give it to fit within this standardization.

   Arg:
       atoms   `ase.Atoms` object of the bulk you want to standardize
   :returns: standardized_struct     `pymatgen.Structure` of the standardized bulk


.. py:function:: is_structure_invertible(structure)

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


.. py:function:: flip_struct(struct)

   Flips an atoms object upside down. Normally used to flip surfaces.

   Arg:
       atoms   `pymatgen.Structure` object
   :returns:

             flipped_struct  The same `ase.Atoms` object that was fed as an
                             argument, but flipped upside down.


.. py:function:: precompute_enumerate_surface(bulk_database, bulk_index, opfile)


.. py:data:: s

   

