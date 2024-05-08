:py:mod:`fairchem.applications.CatTSunami.ocpneb.core.ocpneb`
=============================================================

.. py:module:: fairchem.applications.CatTSunami.ocpneb.core.ocpneb


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   fairchem.applications.CatTSunami.ocpneb.core.ocpneb.OCPNEB




.. py:class:: OCPNEB(images, checkpoint_path, k=0.1, fmax=0.05, climb=False, parallel=False, remove_rotation_and_translation=False, world=None, dynamic_relaxation=True, scale_fmax=0.0, method='aseneb', allow_shared_calculator=False, precon=None, cpu=False, batch_size=4)


   Bases: :py:obj:`ase.neb.DyNEB`

   .. py:method:: load_checkpoint(checkpoint_path: str) -> None

      Load existing trained model

      :param checkpoint_path: string
                              Path to trained model


   .. py:method:: get_forces()

      Evaluate and return the forces.


   .. py:method:: set_positions(positions)


   .. py:method:: get_precon_forces(forces, energies, images)



