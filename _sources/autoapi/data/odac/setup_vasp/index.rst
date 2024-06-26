data.odac.setup_vasp
====================

.. py:module:: data.odac.setup_vasp


Attributes
----------

.. autoapisummary::

   data.odac.setup_vasp.mof


Functions
---------

.. autoapisummary::

   data.odac.setup_vasp.setup_vasp_calc_mof
   data.odac.setup_vasp.setup_vasp_mof_and_ads


Module Contents
---------------

.. py:function:: setup_vasp_calc_mof(atoms: ase.Atoms, path: pathlib.Path)

   Create a VASP calculator for MOF relaxation and write VASP input files to path.


.. py:function:: setup_vasp_mof_and_ads(atoms: ase.Atoms, path: pathlib.Path)

   Create a VASP calculator for MOF + Adsorbate(s) relaxation and write VASP input files to path.
   For these relaxations, the MOF has already been pre-relaxed.


.. py:data:: mof

