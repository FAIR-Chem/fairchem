core.common.relaxation.ase_utils
================================

.. py:module:: core.common.relaxation.ase_utils

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



   Utilities to interface OCP models/trainers with the Atomic Simulation
   Environment (ASE)



Attributes
----------

.. autoapisummary::

   core.common.relaxation.ase_utils.ASE_PROP_RESHAPE


Classes
-------

.. autoapisummary::

   core.common.relaxation.ase_utils.OCPCalculator


Functions
---------

.. autoapisummary::

   core.common.relaxation.ase_utils.batch_to_atoms


Module Contents
---------------

.. py:data:: ASE_PROP_RESHAPE

.. py:function:: batch_to_atoms(batch: torch_geometric.data.Batch, results: dict[str, torch.Tensor] | None = None, wrap_pos: bool = True, eps: float = 1e-07) -> list[ase.Atoms]

   Convert a data batch to ase Atoms

   :param batch: data batch
   :param results: dictionary with predicted result tensors that will be added to a SinglePointCalculator. If no results
                   are given no calculator will be added to the atoms objects.
   :param wrap_pos: wrap positions back into the cell.
   :param eps: Small number to prevent slightly negative coordinates from being wrapped.

   :returns: list of Atoms


.. py:class:: OCPCalculator(config_yml: str | None = None, checkpoint_path: str | pathlib.Path | None = None, model_name: str | None = None, local_cache: str | None = None, trainer: str | None = None, cpu: bool = True, seed: int | None = None, only_output: list[str] | None = None)

   Bases: :py:obj:`ase.calculators.calculator.Calculator`


   ASE based calculator using an OCP model


   .. py:attribute:: _reshaped_props


   .. py:attribute:: config


   .. py:attribute:: trainer


   .. py:attribute:: a2g


   .. py:attribute:: implemented_properties

      Properties calculator can handle (energy, forces, ...)


   .. py:method:: load_checkpoint(checkpoint_path: str, checkpoint: dict | None = None) -> None

      Load existing trained model

      :param checkpoint_path: string
                              Path to trained model
      :param checkpoint: dict
                         A pretrained checkpoint dict



   .. py:method:: calculate(atoms: ase.Atoms | torch_geometric.data.Batch, properties, system_changes) -> None

      Calculate implemented properties for a single Atoms object or a Batch of them.



