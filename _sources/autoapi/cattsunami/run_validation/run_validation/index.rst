cattsunami.run_validation.run_validation
========================================

.. py:module:: cattsunami.run_validation.run_validation

.. autoapi-nested-parse::

   A python script to run a validation of the ML NEB model on a set of NEB calculations.
   This script has not been written to run in parallel, but should be modified to do so.



Attributes
----------

.. autoapisummary::

   cattsunami.run_validation.run_validation.parser


Functions
---------

.. autoapisummary::

   cattsunami.run_validation.run_validation.get_results_sp
   cattsunami.run_validation.run_validation.get_results_ml
   cattsunami.run_validation.run_validation.all_converged
   cattsunami.run_validation.run_validation.both_barrierless
   cattsunami.run_validation.run_validation.both_barriered
   cattsunami.run_validation.run_validation.barrierless_converged
   cattsunami.run_validation.run_validation.is_failed_sp
   cattsunami.run_validation.run_validation.parse_neb_info
   cattsunami.run_validation.run_validation.get_single_point


Module Contents
---------------

.. py:function:: get_results_sp(df2: pandas.DataFrame)

   Get the % success and % convergence for the model considered with
   single points performed on the transition states.

   :param df2: The dataframe containing the results of the
               NEB calculations.
   :type df2: pd.DataFrame

   :returns:

             a tuple of strings containing the % success and
                 % convergence
   :rtype: (tuple[str])


.. py:function:: get_results_ml(df2)

   Get the % success and % convergence for the model considered with
   just ML energy and force calls.

   :param df2: The dataframe containing the results of the
               NEB calculations.
   :type df2: pd.DataFrame

   :returns:

             a tuple of strings containing the % success and
                 % convergence
   :rtype: (tuple[str])


.. py:function:: all_converged(row, ml=True)

   Dataframe function which makes the job of filtering to get % success cleaner.
   It assesses the convergence.

   :param row: the dataframe row which the function is applied to
   :param ml: boolean value. If `True` just the ML NEB and DFT NEB convergence are
              considered. If `False`, the single point convergence is also considered.

   :returns: whether the system is converged
   :rtype: bool


.. py:function:: both_barrierless(row)

   Dataframe function which makes the job of filtering to get % success cleaner.
   It assesses if both DFT and ML find a barrierless transition state.

   :param row: the dataframe row which the function is applied to

   :returns: True if both ML and DFT find a barrierless transition state, False otherwise
   :rtype: bool


.. py:function:: both_barriered(row)

   Dataframe function which makes the job of filtering to get % success cleaner.
   It assesses if both DFT and ML find a barriered transition state.

   :param row: the dataframe row which the function is applied to

   :returns: True if both ML and DFT find a barriered transition state, False otherwise
   :rtype: bool


.. py:function:: barrierless_converged(row)

   Dataframe function which makes the job of filtering to get % success cleaner.
   It assesses if both DFT and ML find a barrierless, converged transition state.

   :param row: the dataframe row which the function is applied to

   :returns:

             True if both ML and DFT find a barrierless converged transition state,
                  False otherwise
   :rtype: bool


.. py:function:: is_failed_sp(row)

   Dataframe function which makes the job of filtering to get % success cleaner.
   It assesses if the single point failed.

   :param row: the dataframe row which the function is applied to

   :returns: True if ths single point failed, otherwise False
   :rtype: bool


.. py:function:: parse_neb_info(neb_frames: list, calc, conv: bool, entry: dict)

   At the conclusion of the ML NEB, this function processes the important
   results and adds them to the entry dictionary.

   :param neb_frames: the ML relaxed NEB frames
   :type neb_frames: list[ase.Atoms]
   :param calc: the ocp ase Atoms calculator
   :param conv: whether or not the NEB achieved forces below the threshold within
                the number of allowed steps
   :type conv: bool
   :param entry: the entry corresponding to the NEB performed
   :type entry: dict


.. py:function:: get_single_point(atoms: ase.Atoms, vasp_dir: str, vasp_flags: dict, vasp_command: str)

   Gets a single point on the atoms passed.

   :param atoms: the atoms object on which the single point will be performed
   :type atoms: ase.Atoms
   :param vasp_dir: the path where the vasp files should be written
   :type vasp_dir: str
   :param vasp_flags: a dictionary of the vasp INCAR flags
   :param vasp_command: the
   :type vasp_command: str


.. py:data:: parser

