core.common.relaxation.optimizable
==================================

.. py:module:: core.common.relaxation.optimizable

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.

   Code based on ase.optimize



Attributes
----------

.. autoapisummary::

   core.common.relaxation.optimizable.ALL_CHANGES


Classes
-------

.. autoapisummary::

   core.common.relaxation.optimizable.Optimizable
   core.common.relaxation.optimizable.OptimizableBatch
   core.common.relaxation.optimizable.OptimizableUnitCellBatch


Functions
---------

.. autoapisummary::

   core.common.relaxation.optimizable.compare_batches


Module Contents
---------------

.. py:class:: Optimizable

.. py:data:: ALL_CHANGES
   :type:  set[str]

.. py:function:: compare_batches(batch1: torch_geometric.data.Batch | None, batch2: torch_geometric.data.Batch, tol: float = 1e-06, excluded_properties: set[str] | None = None) -> list[str]

   Compare properties between two batches

   :param batch1: atoms batch
   :param batch2: atoms batch
   :param tol: tolerance used to compare equility of floating point properties
   :param excluded_properties: list of properties to exclude from comparison

   :returns: list of system changes, property names that are differente between batch1 and batch2


.. py:class:: OptimizableBatch(batch: torch_geometric.data.Batch, trainer: fairchem.core.trainers.BaseTrainer, transform: torch.nn.Module | None = None, mask_converged: bool = True, numpy: bool = False, masked_eps: float = 1e-08)

   Bases: :py:obj:`ase.optimize.optimize.Optimizable`


   A Batch version of ase Optimizable Atoms

   This class can be used with ML relaxations in fairchem.core.relaxations.ml_relaxation
   or in ase relaxations classes, i.e. ase.optimize.lbfgs


   .. py:attribute:: ignored_changes
      :type:  ClassVar[set[str]]


   .. py:attribute:: batch


   .. py:attribute:: trainer


   .. py:attribute:: transform


   .. py:attribute:: numpy


   .. py:attribute:: mask_converged


   .. py:attribute:: _cached_batch
      :value: None



   .. py:attribute:: _update_mask
      :value: None



   .. py:attribute:: torch_results


   .. py:attribute:: results


   .. py:attribute:: _eps


   .. py:attribute:: otf_graph
      :value: True



   .. py:property:: device


   .. py:property:: batch_indices

      Get the batch indices specifying which position/force corresponds to which batch.


   .. py:property:: converged_mask


   .. py:property:: update_mask


   .. py:method:: check_state(batch: torch_geometric.data.Batch, tol: float = 1e-12) -> bool

      Check for any system changes since last calculation.



   .. py:method:: _predict() -> None

      Run prediction if batch has any changes.



   .. py:method:: get_property(name, no_numpy: bool = False) -> torch.Tensor | numpy.typing.NDArray

      Get a predicted property by name.



   .. py:method:: get_positions() -> torch.Tensor | numpy.typing.NDArray

      Get the batch positions



   .. py:method:: set_positions(positions: torch.Tensor | numpy.typing.NDArray) -> None

      Set the atom positions in the batch.



   .. py:method:: get_forces(apply_constraint: bool = False, no_numpy: bool = False) -> torch.Tensor | numpy.typing.NDArray

      Get predicted batch forces.



   .. py:method:: get_potential_energy(**kwargs) -> torch.Tensor | numpy.typing.NDArray

      Get predicted energy as the sum of all batch energies.



   .. py:method:: get_potential_energies() -> torch.Tensor | numpy.typing.NDArray

      Get the predicted energy for each system in batch.



   .. py:method:: get_cells() -> torch.Tensor

      Get batch crystallographic cells.



   .. py:method:: set_cells(cells: torch.Tensor | numpy.typing.NDArray) -> None

      Set batch cells.



   .. py:method:: get_volumes() -> torch.Tensor

      Get a tensor of volumes for each cell in batch



   .. py:method:: iterimages() -> torch_geometric.data.Batch


   .. py:method:: get_max_forces(forces: torch.Tensor | None = None, apply_constraint: bool = False) -> torch.Tensor

      Get the maximum forces per structure in batch



   .. py:method:: converged(forces: torch.Tensor | numpy.typing.NDArray | None, fmax: float, max_forces: torch.Tensor | None = None) -> bool

      Check if norm of all predicted forces are below fmax



   .. py:method:: get_atoms_list() -> list[ase.Atoms]

      Get ase Atoms objects corresponding to the batch



   .. py:method:: update_graph()

      Update the graph if model does not use otf_graph.



   .. py:method:: __len__() -> int


.. py:class:: OptimizableUnitCellBatch(batch: torch_geometric.data.Batch, trainer: fairchem.core.trainers.BaseTrainer, transform: torch.nn.Module | None = None, numpy: bool = False, mask_converged: bool = True, mask: collections.abc.Sequence[bool] | None = None, cell_factor: float | torch.Tensor | None = None, hydrostatic_strain: bool = False, constant_volume: bool = False, scalar_pressure: float = 0.0, masked_eps: float = 1e-08)

   Bases: :py:obj:`OptimizableBatch`


   Modify the supercell and the atom positions in relaxations.

   Based on ase UnitCellFilter to work on data batches


   .. py:attribute:: orig_cells


   .. py:attribute:: stress
      :value: None



   .. py:attribute:: hydrostatic_strain


   .. py:attribute:: constant_volume


   .. py:attribute:: pressure


   .. py:attribute:: cell_factor


   .. py:attribute:: _batch_trace


   .. py:attribute:: _batch_diag


   .. py:property:: batch_indices

      Get the batch indices specifying which position/force corresponds to which batch.

      We augment this to specify the batch indices for augmented positions and forces.


   .. py:method:: deform_grad()

      Get the cell deformation matrix



   .. py:method:: get_positions()

      Get positions and cell deformation gradient.



   .. py:method:: set_positions(positions: torch.Tensor | numpy.typing.NDArray)

      Set positions and cell.

      positions has shape (natoms + ncells * 3, 3).
      the first natoms rows are the positions of the atoms, the last nsystems * three rows are the deformation tensor
      for each cell.



   .. py:method:: get_potential_energy(**kwargs)

      returns potential energy including enthalpy PV term.



   .. py:method:: get_forces(apply_constraint: bool = False, no_numpy: bool = False) -> torch.Tensor | numpy.typing.NDArray

      Get forces and unit cell stress.



   .. py:method:: __len__()


