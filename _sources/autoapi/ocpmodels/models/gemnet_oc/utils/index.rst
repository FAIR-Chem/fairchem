:py:mod:`ocpmodels.models.gemnet_oc.utils`
==========================================

.. py:module:: ocpmodels.models.gemnet_oc.utils

.. autoapi-nested-parse::

   Copyright (c) Facebook, Inc. and its affiliates.
   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   ocpmodels.models.gemnet_oc.utils.ragged_range
   ocpmodels.models.gemnet_oc.utils.repeat_blocks
   ocpmodels.models.gemnet_oc.utils.masked_select_sparsetensor_flat
   ocpmodels.models.gemnet_oc.utils.calculate_interatomic_vectors
   ocpmodels.models.gemnet_oc.utils.inner_product_clamped
   ocpmodels.models.gemnet_oc.utils.get_angle
   ocpmodels.models.gemnet_oc.utils.vector_rejection
   ocpmodels.models.gemnet_oc.utils.get_projected_angle
   ocpmodels.models.gemnet_oc.utils.mask_neighbors
   ocpmodels.models.gemnet_oc.utils.get_neighbor_order
   ocpmodels.models.gemnet_oc.utils.get_inner_idx
   ocpmodels.models.gemnet_oc.utils.get_edge_id



.. py:function:: ragged_range(sizes)

   Multiple concatenated ranges.

   .. rubric:: Examples

   sizes = [1 4 2 3]
   Return: [0  0 1 2 3  0 1  0 1 2]


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


.. py:function:: masked_select_sparsetensor_flat(src, mask) -> torch_sparse.SparseTensor


.. py:function:: calculate_interatomic_vectors(R, id_s, id_t, offsets_st)

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


.. py:function:: inner_product_clamped(x, y) -> torch.Tensor

   Calculate the inner product between the given normalized vectors,
   giving a result between -1 and 1.


.. py:function:: get_angle(R_ac, R_ab) -> torch.Tensor

   Calculate angles between atoms c -> a <- b.

   :param R_ac: Vector from atom a to c.
   :type R_ac: Tensor, shape = (N, 3)
   :param R_ab: Vector from atom a to b.
   :type R_ab: Tensor, shape = (N, 3)

   :returns: **angle_cab** -- Angle between atoms c <- a -> b.
   :rtype: Tensor, shape = (N,)


.. py:function:: vector_rejection(R_ab, P_n)

   Project the vector R_ab onto a plane with normal vector P_n.

   :param R_ab: Vector from atom a to b.
   :type R_ab: Tensor, shape = (N, 3)
   :param P_n: Normal vector of a plane onto which to project R_ab.
   :type P_n: Tensor, shape = (N, 3)

   :returns: **R_ab_proj** -- Projected vector (orthogonal to P_n).
   :rtype: Tensor, shape = (N, 3)


.. py:function:: get_projected_angle(R_ab, P_n, eps: float = 0.0001) -> torch.Tensor

   Project the vector R_ab onto a plane with normal vector P_n,
   then calculate the angle w.r.t. the (x [cross] P_n),
   or (y [cross] P_n) if the former would be ill-defined/numerically unstable.

   :param R_ab: Vector from atom a to b.
   :type R_ab: Tensor, shape = (N, 3)
   :param P_n: Normal vector of a plane onto which to project R_ab.
   :type P_n: Tensor, shape = (N, 3)
   :param eps: Norm of projection below which to use the y-axis instead of x.
   :type eps: float

   :returns: **angle_ab** -- Angle on plane w.r.t. x- or y-axis.
   :rtype: Tensor, shape = (N)


.. py:function:: mask_neighbors(neighbors, edge_mask)


.. py:function:: get_neighbor_order(num_atoms: int, index, atom_distance) -> torch.Tensor

   Give a mask that filters out edges so that each atom has at most
   `max_num_neighbors_threshold` neighbors.


.. py:function:: get_inner_idx(idx, dim_size)

   Assign an inner index to each element (neighbor) with the same index.
   For example, with idx=[0 0 0 1 1 1 1 2 2] this returns [0 1 2 0 1 2 3 0 1].
   These indices allow reshape neighbor indices into a dense matrix.
   idx has to be sorted for this to work.


.. py:function:: get_edge_id(edge_idx, cell_offsets, num_atoms: int)


