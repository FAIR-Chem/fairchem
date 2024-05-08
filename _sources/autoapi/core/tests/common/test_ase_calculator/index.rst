:py:mod:`core.tests.common.test_ase_calculator`
===============================================

.. py:module:: core.tests.common.test_ase_calculator

.. autoapi-nested-parse::

   Copyright (c) Facebook, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   core.tests.common.test_ase_calculator.atoms
   core.tests.common.test_ase_calculator.checkpoint_path
   core.tests.common.test_ase_calculator.test_calculator_setup
   core.tests.common.test_ase_calculator.test_relaxation_final_energy
   core.tests.common.test_ase_calculator.test_random_seed_final_energy



.. py:function:: atoms() -> ase.Atoms


.. py:function:: checkpoint_path(request, tmp_path)


.. py:function:: test_calculator_setup(checkpoint_path)


.. py:function:: test_relaxation_final_energy(atoms, tmp_path, snapshot) -> None


.. py:function:: test_random_seed_final_energy(atoms, tmp_path)


