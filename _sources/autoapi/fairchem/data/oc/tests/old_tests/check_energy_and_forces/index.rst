:py:mod:`fairchem.data.oc.tests.old_tests.check_energy_and_forces`
==================================================================

.. py:module:: fairchem.data.oc.tests.old_tests.check_energy_and_forces


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   fairchem.data.oc.tests.old_tests.check_energy_and_forces.check_relaxed_forces
   fairchem.data.oc.tests.old_tests.check_energy_and_forces.check_adsorption_energy
   fairchem.data.oc.tests.old_tests.check_energy_and_forces.check_DFT_energy
   fairchem.data.oc.tests.old_tests.check_energy_and_forces.check_positions_across_frames_are_different
   fairchem.data.oc.tests.old_tests.check_energy_and_forces.read_pkl
   fairchem.data.oc.tests.old_tests.check_energy_and_forces.run_checks
   fairchem.data.oc.tests.old_tests.check_energy_and_forces.create_parser



Attributes
~~~~~~~~~~

.. autoapisummary::

   fairchem.data.oc.tests.old_tests.check_energy_and_forces.parser


.. py:function:: check_relaxed_forces(sid, path, thres)

   Check all forces in the final frame of adslab is less than a threshold.


.. py:function:: check_adsorption_energy(sid, path, ref_energy, adsorption_energy)


.. py:function:: check_DFT_energy(sid, path, e_tol=0.05)

   Given a relaxation trajectory, check to see if 1. final energy is less than the initial
   energy, raise error if not. 2) If the energy decreases throuhghout a trajectory (small spikes are okay).
   And 3) if 2 fails, check if it's just a matter of tolerance being too strict by
   considering only the first quarter of the trajectory and sampling every 10th frame
   to check for _almost_ monotonic decrease in energies.
   If any frame(i+1) energy is higher than frame(i) energy, flag it and plot the trajectory.


.. py:function:: check_positions_across_frames_are_different(sid, path)

   Given a relaxation trajectory, make sure positions for two consecutive
   frames are not identical.


.. py:function:: read_pkl(fname)


.. py:function:: run_checks(args)


.. py:function:: create_parser()


.. py:data:: parser

   

