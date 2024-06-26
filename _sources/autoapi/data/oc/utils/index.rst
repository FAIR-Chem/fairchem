data.oc.utils
=============

.. py:module:: data.oc.utils


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/data/oc/utils/flag_anomaly/index
   /autoapi/data/oc/utils/vasp/index


Classes
-------

.. autoapisummary::

   data.oc.utils.DetectTrajAnomaly


Package Contents
----------------

.. py:class:: DetectTrajAnomaly(init_atoms, final_atoms, atoms_tag, final_slab_atoms=None, surface_change_cutoff_multiplier=1.5, desorption_cutoff_multiplier=1.5)

   .. py:method:: is_adsorbate_dissociated()

      Tests if the initial adsorbate connectivity is maintained.

      :returns: True if the connectivity was not maintained, otherwise False
      :rtype: (bool)



   .. py:method:: has_surface_changed()

      Tests bond breaking / forming events within a tolerance on the surface so
      that systems with significant adsorbate induces surface changes may be discarded
      since the reference to the relaxed slab may no longer be valid.

      :returns: True if the surface is reconstructed, otherwise False
      :rtype: (bool)



   .. py:method:: is_adsorbate_desorbed()

      If the adsorbate binding atoms have no connection with slab atoms,
      consider it desorbed.

      :returns: True if there is desorption, otherwise False
      :rtype: (bool)



   .. py:method:: _get_connectivity(atoms, cutoff_multiplier=1.0)

      Generate the connectivity of an atoms obj.

      :param atoms: object which will have its connectivity considered
      :type atoms: ase.Atoms
      :param cutoff_multiplier: cushion for small atom movements when assessing
                                atom connectivity
      :type cutoff_multiplier: float, optional

      :returns: The connectivity matrix of the atoms object.
      :rtype: (np.ndarray)



   .. py:method:: is_adsorbate_intercalated()

      Ensure the adsorbate isn't interacting with an atom that is not allowed to relax.

      :returns: True if any adsorbate atom neighbors a frozen atom, otherwise False
      :rtype: (bool)



