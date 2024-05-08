:py:mod:`core.models.gemnet_gp.utils`
=====================================

.. py:module:: core.models.gemnet_gp.utils

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   core.models.gemnet_gp.utils.read_json
   core.models.gemnet_gp.utils.update_json
   core.models.gemnet_gp.utils.write_json
   core.models.gemnet_gp.utils.read_value_json
   core.models.gemnet_gp.utils.ragged_range
   core.models.gemnet_gp.utils.repeat_blocks
   core.models.gemnet_gp.utils.calculate_interatomic_vectors
   core.models.gemnet_gp.utils.inner_product_normalized
   core.models.gemnet_gp.utils.mask_neighbors



.. py:function:: read_json(path: str)


.. py:function:: update_json(path: str, data) -> None


.. py:function:: write_json(path: str, data) -> None


.. py:function:: read_value_json(path: str, key)


.. py:function:: ragged_range(sizes)

   Multiple concatenated ranges.

   .. rubric:: Examples

   sizes = [1 4 2 3]
   Return: [0  0 1 2 3  0 1  0 1 2]


.. py:function:: repeat_blocks(sizes: torch.Tensor, repeats, continuous_indexing: bool = True, start_idx: int = 0, block_inc: int = 0, repeat_inc: int = 0) -> torch.Tensor

   Repeat blocks of indices.
   Adapted from https://stackoverflow.com/questions/51154989/numpy-vectorized-function-to-repeat-blocks-of-consecutive-elements

   continuous_indexing: Whether to keep increasing the index after each block
   start_idx: Starting index
   block_inc: Number to increment by after each block,
              either global or per block. Shape: len(sizes) - 1
   repeat_inc: Number to increment by after each repetition,
               either global or per block

   .. rubric:: Examples

   sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = False
   Return: [0 0 0  0 1 2 0 1 2  0 1 0 1 0 1]
   sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True
   Return: [0 0 0  1 2 3 1 2 3  4 5 4 5 4 5]
   sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
   repeat_inc = 4
   Return: [0 4 8  1 2 3 5 6 7  4 5 8 9 12 13]
   sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
   start_idx = 5
   Return: [5 5 5  6 7 8 6 7 8  9 10 9 10 9 10]
   sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
   block_inc = 1
   Return: [0 0 0  2 3 4 2 3 4  6 7 6 7 6 7]
   sizes = [0,3,2] ; repeats = [3,2,3] ; continuous_indexing = True
   Return: [0 1 2 0 1 2  3 4 3 4 3 4]
   sizes = [2,3,2] ; repeats = [2,0,2] ; continuous_indexing = True
   Return: [0 1 0 1  5 6 5 6]


.. py:function:: calculate_interatomic_vectors(R: torch.Tensor, id_s: torch.Tensor, id_t: torch.Tensor, offsets_st: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]

   Calculate the vectors connecting the given atom pairs,
   considering offsets from periodic boundary conditions (PBC).

   :param R: Atom positions.
   :type R: Tensor, shape = (nAtoms, 3)
   :param id_s: Indices of the source atom of the edges.
   :type id_s: Tensor, shape = (nEdges,)
   :param id_t: Indices of the target atom of the edges.
   :type id_t: Tensor, shape = (nEdges,)
   :param offsets_st: PBC offsets of the edges.
                      Subtract this from the correct direction.
   :type offsets_st: Tensor, shape = (nEdges,)

   :returns: **(D_st, V_st)** --

             D_st: Tensor, shape = (nEdges,)
                 Distance from atom t to s.
             V_st: Tensor, shape = (nEdges,)
                 Unit direction from atom t to s.
   :rtype: tuple


.. py:function:: inner_product_normalized(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor

   Calculate the inner product between the given normalized vectors,
   giving a result between -1 and 1.


.. py:function:: mask_neighbors(neighbors: torch.Tensor, edge_mask: torch.Tensor)


