cattsunami.core
===============

.. py:module:: cattsunami.core


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/cattsunami/core/autoframe/index
   /autoapi/cattsunami/core/ocpneb/index
   /autoapi/cattsunami/core/reaction/index


Classes
-------

.. autoapisummary::

   cattsunami.core.OCPNEB
   cattsunami.core.Reaction


Package Contents
----------------

.. py:class:: OCPNEB(images, checkpoint_path, k=0.1, fmax=0.05, climb=False, parallel=False, remove_rotation_and_translation=False, world=None, dynamic_relaxation=True, scale_fmax=0.0, method='aseneb', allow_shared_calculator=False, precon=None, cpu=False, batch_size=4)

   Bases: :py:obj:`ase.neb.DyNEB`


   .. py:attribute:: batch_size


   .. py:attribute:: trainer


   .. py:attribute:: a2g


   .. py:attribute:: intermediate_energies
      :value: []



   .. py:attribute:: intermediate_forces
      :value: []



   .. py:attribute:: cached
      :value: False



   .. py:method:: load_checkpoint(checkpoint_path: str) -> None

      Load existing trained model

      :param checkpoint_path: string
                              Path to trained model



   .. py:method:: get_forces()

      Evaluate and return the forces.



   .. py:method:: set_positions(positions)


   .. py:method:: get_precon_forces(forces, energies, images)


.. py:class:: Reaction(reaction_db_path: str, adsorbate_db_path: str, reaction_id_from_db: int | None = None, reaction_str_from_db: str | None = None, reaction_type: str | None = None)

   Initialize Reaction object


   .. py:attribute:: reaction_db_path


   .. py:method:: get_desorption_mapping(reactant)

      Get mapping for desorption reaction



