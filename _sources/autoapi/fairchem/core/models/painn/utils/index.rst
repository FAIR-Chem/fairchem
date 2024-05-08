:py:mod:`fairchem.core.models.painn.utils`
==========================================

.. py:module:: fairchem.core.models.painn.utils

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   fairchem.core.models.painn.utils.repeat_blocks
   fairchem.core.models.painn.utils.get_edge_id



.. py:function:: repeat_blocks(sizes, repeats, continuous_indexing: bool = True, start_idx: int = 0, block_inc: int = 0, repeat_inc: int = 0) -> torch.Tensor

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


.. py:function:: get_edge_id(edge_idx, cell_offsets, num_atoms: int)


