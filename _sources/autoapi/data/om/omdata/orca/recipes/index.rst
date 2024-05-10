:py:mod:`data.om.omdata.orca.recipes`
=====================================

.. py:module:: data.om.omdata.orca.recipes


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   data.om.omdata.orca.recipes.single_point_calculation
   data.om.omdata.orca.recipes.ase_relaxation



.. py:function:: single_point_calculation(atoms, charge, spin_multiplicity, xc=ORCA_FUNCTIONAL, basis=ORCA_BASIS, orcasimpleinput=None, orcablocks=None, nprocs=12, outputdir=os.getcwd(), **calc_kwargs)

   Wrapper around QUACC's static job to standardize single-point calculations.
   See github.com/Quantum-Accelerators/quacc/blob/main/src/quacc/recipes/orca/core.py#L22
   for more details.

   :param atoms: Atoms object
   :type atoms: Atoms
   :param charge: Charge of system
   :type charge: int
   :param spin_multiplicity: Multiplicity of the system
   :type spin_multiplicity: int
   :param xc: Exchange-correlaction functional
   :type xc: str
   :param basis: Basis set
   :type basis: str
   :param orcasimpleinput: List of `orcasimpleinput` settings for the calculator
   :type orcasimpleinput: list
   :param orcablocks: List of `orcablocks` swaps for the calculator
   :type orcablocks: list
   :param nprocs: Number of processes to parallelize across
   :type nprocs: int
   :param outputdir: Directory to move results to upon completion
   :type outputdir: str
   :param calc_kwargs: Additional kwargs for the custom Orca calculator


.. py:function:: ase_relaxation(atoms, charge, spin_multiplicity, xc=ORCA_FUNCTIONAL, basis=ORCA_BASIS, orcasimpleinput=None, orcablocks=None, nprocs=12, opt_params=None, outputdir=os.getcwd(), **calc_kwargs)

   Wrapper around QUACC's ase_relax_job to standardize geometry optimizations.
   See github.com/Quantum-Accelerators/quacc/blob/main/src/quacc/recipes/orca/core.py#L22
   for more details.

   :param atoms: Atoms object
   :type atoms: Atoms
   :param charge: Charge of system
   :type charge: int
   :param spin_multiplicity: Multiplicity of the system
   :type spin_multiplicity: int
   :param xc: Exchange-correlaction functional
   :type xc: str
   :param basis: Basis set
   :type basis: str
   :param orcasimpleinput: List of `orcasimpleinput` settings for the calculator
   :type orcasimpleinput: list
   :param orcablocks: List of `orcablocks` swaps for the calculator
   :type orcablocks: list
   :param nprocs: Number of processes to parallelize across
   :type nprocs: int
   :param opt_params: Dictionary of optimizer parameters
   :type opt_params: dict
   :param outputdir: Directory to move results to upon completion
   :type outputdir: str
   :param calc_kwargs: Additional kwargs for the custom Orca calculator


