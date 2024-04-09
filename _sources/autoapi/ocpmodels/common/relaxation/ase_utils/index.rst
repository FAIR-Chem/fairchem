:py:mod:`ocpmodels.common.relaxation.ase_utils`
===============================================

.. py:module:: ocpmodels.common.relaxation.ase_utils

.. autoapi-nested-parse::

   Copyright (c) Facebook, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



   Utilities to interface OCP models/trainers with the Atomic Simulation
   Environment (ASE)



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   ocpmodels.common.relaxation.ase_utils.OCPCalculator



Functions
~~~~~~~~~

.. autoapisummary::

   ocpmodels.common.relaxation.ase_utils.batch_to_atoms



.. py:function:: batch_to_atoms(batch)


.. py:class:: OCPCalculator(config_yml: Optional[str] = None, checkpoint_path: Optional[str] = None, model_name: Optional[str] = None, local_cache: Optional[str] = None, trainer: Optional[str] = None, cutoff: int = 6, max_neighbors: int = 50, cpu: bool = True, seed: Optional[int] = None)


   Bases: :py:obj:`ase.calculators.calculator.Calculator`

   Base-class for all ASE calculators.

   A calculator must raise PropertyNotImplementedError if asked for a
   property that it can't calculate.  So, if calculation of the
   stress tensor has not been implemented, get_stress(atoms) should
   raise PropertyNotImplementedError.  This can be achieved simply by not
   including the string 'stress' in the list implemented_properties
   which is a class member.  These are the names of the standard
   properties: 'energy', 'forces', 'stress', 'dipole', 'charges',
   'magmom' and 'magmoms'.

   .. py:attribute:: implemented_properties
      :value: ['energy', 'forces']

      

   .. py:method:: load_checkpoint(checkpoint_path: str, checkpoint: Dict = {}) -> None

      Load existing trained model

      :param checkpoint_path: string
                              Path to trained model


   .. py:method:: calculate(atoms: ase.Atoms, properties, system_changes) -> None

      Do the calculation.

      properties: list of str
          List of what needs to be calculated.  Can be any combination
          of 'energy', 'forces', 'stress', 'dipole', 'charges', 'magmom'
          and 'magmoms'.
      system_changes: list of str
          List of what has changed since last calculation.  Can be
          any combination of these six: 'positions', 'numbers', 'cell',
          'pbc', 'initial_charges' and 'initial_magmoms'.

      Subclasses need to implement this, but can ignore properties
      and system_changes if they want.  Calculated properties should
      be inserted into results dictionary like shown in this dummy
      example::

          self.results = {'energy': 0.0,
                          'forces': np.zeros((len(atoms), 3)),
                          'stress': np.zeros(6),
                          'dipole': np.zeros(3),
                          'charges': np.zeros(len(atoms)),
                          'magmom': 0.0,
                          'magmoms': np.zeros(len(atoms))}

      The subclass implementation should first call this
      implementation to set the atoms attribute and create any missing
      directories.



