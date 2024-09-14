data.oc.utils.vasp
==================

.. py:module:: data.oc.utils.vasp

.. autoapi-nested-parse::

   This submodule contains the scripts that the we used to run VASP.

   Note that some of these scripts were taken and modified from
   [GASpy](https://github.com/ulissigroup/GASpy) with permission of authors.



Attributes
----------

.. autoapisummary::

   data.oc.utils.vasp.__author__
   data.oc.utils.vasp.__email__


Functions
---------

.. autoapisummary::

   data.oc.utils.vasp._clean_up_inputs
   data.oc.utils.vasp.calculate_surface_k_points
   data.oc.utils.vasp.write_vasp_input_files


Module Contents
---------------

.. py:data:: __author__
   :value: 'Kevin Tran'


.. py:data:: __email__
   :value: 'ktran@andrew.cmu.edu'


.. py:function:: _clean_up_inputs(atoms, vasp_flags)

   Parses the inputs and makes sure some things are straightened out.

   Arg:
       atoms       `ase.Atoms` object of the structure we want to relax
       vasp_flags  A dictionary of settings we want to pass to the `Vasp`
                   calculator
   :returns:

             atoms       `ase.Atoms` object of the structure we want to relax, but
                         with the unit vectors fixed (if needed)
             vasp_flags  A modified version of the 'vasp_flags' argument


.. py:function:: calculate_surface_k_points(atoms)

   For surface calculations, it's a good practice to calculate the k-point
   mesh given the unit cell size. We do that on-the-spot here.

   Arg:
       atoms   `ase.Atoms` object of the structure we want to relax
   :returns: k_pts   A 3-tuple of integers indicating the k-point mesh to use


.. py:function:: write_vasp_input_files(atoms, outdir='.', vasp_flags=None, pp_setups='minimal', pp_env='VASP_PP_PATH')

   Effectively goes through the same motions as the `run_vasp` function,
   except it only writes the input files instead of running.

   :param atoms       `ase.Atoms` object that we want to relax.:
   :param outdir      A string indicating where you want to save the input files.: Defaults to '.'
   :param vasp_flags  A dictionary of settings we want to pass to the `Vasp`: calculator. Defaults to a standerd set of values if `None`
   :param pp_setups   Pseudopotential setups to use - https: //gitlab.com/ase/ase/-/blob/master/ase/calculators/vasp/setups.py
   :param pp_env      Environment variable to read for pseudopotentials.:


