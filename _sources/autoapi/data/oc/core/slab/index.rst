:py:mod:`data.oc.core.slab`
===========================

.. py:module:: data.oc.core.slab


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   data.oc.core.slab.Slab



Functions
~~~~~~~~~

.. autoapisummary::

   data.oc.core.slab.tile_and_tag_atoms
   data.oc.core.slab.set_fixed_atom_constraints
   data.oc.core.slab.tag_surface_atoms
   data.oc.core.slab.tile_atoms
   data.oc.core.slab.find_surface_atoms_by_height
   data.oc.core.slab.find_surface_atoms_with_voronoi_given_height
   data.oc.core.slab.calculate_center_of_mass
   data.oc.core.slab.calculate_coordination_of_bulk_atoms
   data.oc.core.slab.compute_slabs
   data.oc.core.slab.flip_struct
   data.oc.core.slab.is_structure_invertible
   data.oc.core.slab.standardize_bulk



.. py:class:: Slab(bulk=None, slab_atoms: ase.Atoms = None, millers: tuple = None, shift: float = None, top: bool = None, oriented_bulk: pymatgen.core.structure.Structure = None, min_ab: float = 0.8)


   Initializes a slab object, i.e. a particular slab tiled along xyz, in
   one of 2 ways:
   - Pass in a Bulk object and a slab 5-tuple containing
   (atoms, miller, shift, top, oriented bulk).
   - Pass in a Bulk object and randomly sample a slab.

   :param bulk: Corresponding Bulk object.
   :type bulk: Bulk
   :param slab_atoms: Slab atoms, tiled and tagged
   :type slab_atoms: ase.Atoms
   :param millers: Miller indices of slab.
   :type millers: tuple
   :param shift: Shift of slab.
   :type shift: float
   :param top: Whether slab is top or bottom.
   :type top: bool
   :param min_ab: To confirm that the tiled structure spans this distance
   :type min_ab: float

   .. py:method:: from_bulk_get_random_slab(bulk=None, max_miller=2, min_ab=8.0, save_path=None)
      :classmethod:


   .. py:method:: from_bulk_get_specific_millers(specific_millers, bulk=None, min_ab=8.0, save_path=None)
      :classmethod:


   .. py:method:: from_bulk_get_all_slabs(bulk=None, max_miller=2, min_ab=8.0, save_path=None)
      :classmethod:


   .. py:method:: from_precomputed_slabs_pkl(bulk=None, precomputed_slabs_pkl=None, max_miller=2, min_ab=8.0)
      :classmethod:


   .. py:method:: from_atoms(atoms: ase.Atoms = None, bulk=None, **kwargs)
      :classmethod:


   .. py:method:: has_surface_tagged()


   .. py:method:: get_metadata_dict()


   .. py:method:: __len__()


   .. py:method:: __str__()

      Return str(self).


   .. py:method:: __repr__()

      Return repr(self).


   .. py:method:: __eq__(other)

      Return self==value.



.. py:function:: tile_and_tag_atoms(unit_slab_struct: pymatgen.core.structure.Structure, bulk_atoms: ase.Atoms, min_ab: float = 8)

   This function combines the next three functions that tile, tag,
   and constrain the atoms.

   :param unit_slab_struct: The untiled slab structure
   :type unit_slab_struct: Structure
   :param bulk_atoms: Atoms of the corresponding bulk structure, used for tagging
   :type bulk_atoms: ase.Atoms
   :param min_ab: The minimum distance in x and y spanned by the tiled structure.
   :type min_ab: float

   :returns: **atoms_tiled** -- A copy of the slab atoms that is tiled, tagged, and constrained
   :rtype: ase.Atoms


.. py:function:: set_fixed_atom_constraints(atoms)

   This function fixes sub-surface atoms of a surface. Also works on systems
   that have surface + adsorbate(s), as long as the bulk atoms are tagged with
   `0`, surface atoms are tagged with `1`, and the adsorbate atoms are tagged
   with `2` or above.

   This is used for both surface atoms and the combined surface+adsorbate.

   :param atoms: Atoms object of the slab or slab+adsorbate system, with bulk atoms
                 tagged as `0`, surface atoms tagged as `1`, and adsorbate atoms tagged
                 as `2` or above.
   :type atoms: ase.Atoms

   :returns: **atoms** -- A deep copy of the `atoms` argument, but where the appropriate
             atoms are constrained.
   :rtype: ase.Atoms


.. py:function:: tag_surface_atoms(slab_atoms: ase.Atoms = None, bulk_atoms: ase.Atoms = None)

   Sets the tags of an `ase.Atoms` object. Any atom that we consider a "bulk"
   atom will have a tag of 0, and any atom that we consider a "surface" atom
   will have a tag of 1. We use a combination of Voronoi neighbor algorithms
   (adapted from `pymatgen.core.surface.Slab.get_surface_sites`; see
   https://pymatgen.org/pymatgen.core.surface.html) and a distance cutoff.

   :param slab_atoms: The slab where you are trying to find surface sites.
   :type slab_atoms: ase.Atoms
   :param bulk_atoms: The bulk structure that the surface was cut from.
   :type bulk_atoms: ase.Atoms

   :returns: **slab_atoms** -- A copy of the slab atoms with the surface atoms tagged as 1.
   :rtype: ase.Atoms


.. py:function:: tile_atoms(atoms: ase.Atoms, min_ab: float = 8)

   This function will repeat an atoms structure in the direction of the a and b
   lattice vectors such that they are at least as wide as the min_ab constant.

   :param atoms: The structure to tile.
   :type atoms: ase.Atoms
   :param min_ab: The minimum distance in x and y spanned by the tiled structure.
   :type min_ab: float

   :returns: **atoms_tiled** -- The tiled structure.
   :rtype: ase.Atoms


.. py:function:: find_surface_atoms_by_height(surface_atoms)

   As discussed in the docstring for `find_surface_atoms_with_voronoi`,
   sometimes we might accidentally tag a surface atom as a bulk atom if there
   are multiple coordination environments for that atom type within the bulk.
   One heuristic that we use to address this is to simply figure out if an
   atom is close to the surface. This function will figure that out.

   Specifically:  We consider an atom a surface atom if it is within 2
   Angstroms of the heighest atom in the z-direction (or more accurately, the
   direction of the 3rd unit cell vector).

   :param surface_atoms:
   :type surface_atoms: ase.Atoms

   :returns: **tags** -- A list that contains the indices of the surface atoms.
   :rtype: list


.. py:function:: find_surface_atoms_with_voronoi_given_height(bulk_atoms, slab_atoms, height_tags)

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

   :param bulk_atoms: The bulk structure that the surface was cut from.
   :type bulk_atoms: ase.Atoms
   :param slab_atoms: The slab structure.
   :type slab_atoms: ase.Atoms
   :param height_tags: The tags determined by the `find_surface_atoms_by_height` algo.
   :type height_tags: list

   :returns: **tags** -- A list of 0s and 1s whose indices align with the atoms in
             `slab_atoms`. 0s indicate a bulk atom and 1 indicates a surface atom.
   :rtype: list


.. py:function:: calculate_center_of_mass(struct)

   Calculates the center of mass of the slab.


.. py:function:: calculate_coordination_of_bulk_atoms(bulk_atoms)

   Finds all unique atoms in a bulk structure and then determines their
   coordination number. Then parses these coordination numbers into a
   dictionary whose keys are the elements of the atoms and whose values are
   their possible coordination numbers.
   For example: `bulk_cns = {'Pt': {3., 12.}, 'Pd': {12.}}`

   :param bulk_atoms: The bulk structure.
   :type bulk_atoms: ase.Atoms

   :returns: **bulk_cn_dict** -- A dictionary whose keys are the elements of the atoms and whose values
             are their possible coordination numbers.
   :rtype: dict


.. py:function:: compute_slabs(bulk_atoms: ase.Atoms = None, max_miller: int = 2, specific_millers: list = None)

   Enumerates all the symmetrically distinct slabs of a bulk structure.
   It will not enumerate slabs with Miller indices above the
   `max_miller` argument. Note that we also look at the bottoms of slabs
   if they are distinct from the top. If they are distinct, we flip the
   surface so the bottom is pointing upwards.

   :param bulk_atoms: The bulk structure.
   :type bulk_atoms: ase.Atoms
   :param max_miller: The maximum Miller index of the slabs to enumerate. Increasing this
                      argument will increase the number of slabs, and the slabs will generally
                      become larger.
   :type max_miller: int
   :param specific_millers: A list of Miller indices that you want to enumerate. If this argument
                            is not `None`, then the `max_miller` argument is ignored.
   :type specific_millers: list

   :returns: **all_slabs_info** -- A list of 5-tuples containing pymatgen structure objects for enumerated
             slabs, the Miller indices, floats for the shifts, booleans for top, and
             the oriented bulk structure.
   :rtype: list


.. py:function:: flip_struct(struct: pymatgen.core.structure.Structure)

   Flips an atoms object upside down. Normally used to flip slabs.

   :param struct: pymatgen structure object of the surface you want to flip
   :type struct: Structure

   :returns: **flipped_struct** -- pymatgen structure object of the flipped surface.
   :rtype: Structure


.. py:function:: is_structure_invertible(struct: pymatgen.core.structure.Structure)

   This function figures out whether or not an `Structure`
   object has symmetricity. In this function, the affine matrix is a rotation
   matrix that is multiplied with the XYZ positions of the crystal. If the z,z
   component of that is negative, it means symmetry operation exist, it could
   be a mirror operation, or one that involves multiple rotations/etc.
   Regardless, it means that the top becomes the bottom and vice-versa, and the
   structure is the symmetric. i.e. structure_XYZ = structure_XYZ*M.

   In short:  If this function returns `False`, then the input structure can
   be flipped in the z-direction to create a new structure.

   :param struct: pymatgen structure object of the slab.
   :type struct: Structure

   :returns: * A boolean indicating whether or not your `ase.Atoms` object is
             * *symmetric in z-direction (i.e. symmetric with respect to x-y plane).*


.. py:function:: standardize_bulk(atoms: ase.Atoms)

   There are many ways to define a bulk unit cell. If you change the unit
   cell itself but also change the locations of the atoms within the unit
   cell, you can effectively get the same bulk structure. To address this,
   there is a standardization method used to reduce the degrees of freedom
   such that each unit cell only has one "true" configuration. This
   function will align a unit cell you give it to fit within this
   standardization.

   :param atoms: `ase.Atoms` object of the bulk you want to standardize.
   :type atoms: ase.Atoms

   :returns: **standardized_struct** -- pymatgen structure object of the standardized bulk.
   :rtype: Structure


