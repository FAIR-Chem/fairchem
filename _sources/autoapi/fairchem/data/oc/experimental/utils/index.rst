:py:mod:`fairchem.data.oc.experimental.utils`
=============================================

.. py:module:: fairchem.data.oc.experimental.utils


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   fairchem.data.oc.experimental.utils.v0_check
   fairchem.data.oc.experimental.utils.restart_bug_check
   fairchem.data.oc.experimental.utils.plot_traj



.. py:function:: v0_check(full_traj, initial)

   Checks whether the initial structure as gathered from the POSCAR input file
   is in agreement with the initial image of the full trajectory. If not, the
   trajectory comes fro the V0 dataset which failed to save intermediate
   checkpoints.

   Args
   full_traj (list of Atoms objects): Calculated full trajectory.
   initial (Atoms object): Starting image provided by POSCAR..


.. py:function:: restart_bug_check(full_traj)

   Observed that some of the trajectories had a strange identically cyclical
   behavior - suggesting that a checkpoint was restarted from an earlier
   checkpoint rather than the latest. Checks whether the trajectory provided
   falls within that bug.

   Args
   full_traj (list of Atoms objects): Calculated full trajectory.


.. py:function:: plot_traj(traj, fname)

   Plots the energy profile of a given trajectory

   Args
   traj (list of Atoms objects): Full trajectory to be plotted
   fname (str): Filename to be used as title and save figure as.


