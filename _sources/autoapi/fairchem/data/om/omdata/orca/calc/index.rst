:py:mod:`fairchem.data.om.omdata.orca.calc`
===========================================

.. py:module:: fairchem.data.om.omdata.orca.calc


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   fairchem.data.om.omdata.orca.calc.write_orca_inputs



Attributes
~~~~~~~~~~

.. autoapisummary::

   fairchem.data.om.omdata.orca.calc.ORCA_FUNCTIONAL
   fairchem.data.om.omdata.orca.calc.ORCA_BASIS
   fairchem.data.om.omdata.orca.calc.ORCA_SIMPLE_INPUT
   fairchem.data.om.omdata.orca.calc.ORCA_BLOCKS
   fairchem.data.om.omdata.orca.calc.ORCA_ASE_SIMPLE_INPUT
   fairchem.data.om.omdata.orca.calc.OPT_PARAMETERS


.. py:data:: ORCA_FUNCTIONAL
   :value: 'wB97M-V'

   

.. py:data:: ORCA_BASIS
   :value: 'def2-TZVPD'

   

.. py:data:: ORCA_SIMPLE_INPUT
   :value: ['EnGrad', 'RIJCOSX', 'def2/J', 'NoUseSym', 'DIIS', 'NOSOSCF', 'NormalConv', 'DEFGRID3', 'ALLPOP', 'NBO']

   

.. py:data:: ORCA_BLOCKS
   :value: ['%scf Convergence Tight maxiter 300 end', '%elprop Dipole true Quadrupole true end', '%nbo...

   

.. py:data:: ORCA_ASE_SIMPLE_INPUT

   

.. py:data:: OPT_PARAMETERS

   

.. py:function:: write_orca_inputs(atoms, output_directory, charge=0, mult=1, orcasimpleinput=ORCA_ASE_SIMPLE_INPUT, orcablocks=' '.join(ORCA_BLOCKS))

   One-off method to be used if you wanted to write inputs for an arbitrary
   system. Primarily used for debugging.


