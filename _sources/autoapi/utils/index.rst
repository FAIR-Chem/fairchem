:py:mod:`utils`
===============

.. py:module:: utils


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   utils.converged_oszicar
   utils.count_scf



.. py:function:: converged_oszicar(path, nelm=60, ediff=0.0001, idx=0)

   --- FOR VASP USERS ---

   Given a folder containing DFT outputs, ensures the system has converged
   electronically.

   :param path: Path to DFT outputs.
   :param nelm: Maximum number of electronic steps used.
   :param ediff: Energy difference condition for terminating the electronic loop.
   :param idx: Frame to check for electronic convergence. 0 for SP, -1 for RX.


.. py:function:: count_scf(path)

   --- FOR VASP USERS ---

   Given a folder containing DFT outputs, compute total ionic and SCF steps

   :param path: Path to DFT outputs.

   :returns: Total number of electronic steps performed.
             ionic_steps (int): Total number of ionic steps performed.
   :rtype: scf_steps (int)


