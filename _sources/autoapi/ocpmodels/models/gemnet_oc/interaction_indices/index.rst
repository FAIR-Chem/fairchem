:py:mod:`ocpmodels.models.gemnet_oc.interaction_indices`
========================================================

.. py:module:: ocpmodels.models.gemnet_oc.interaction_indices

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.
   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   ocpmodels.models.gemnet_oc.interaction_indices.get_triplets
   ocpmodels.models.gemnet_oc.interaction_indices.get_mixed_triplets
   ocpmodels.models.gemnet_oc.interaction_indices.get_quadruplets



.. py:function:: get_triplets(graph, num_atoms: int)

   Get all input edges b->a for each output edge c->a.
   It is possible that b=c, as long as the edges are distinct
   (i.e. atoms b and c stem from different unit cells).

   :param graph: Contains the graph's edge_index.
   :type graph: dict of torch.Tensor
   :param num_atoms: Total number of atoms.
   :type num_atoms: int

   :returns:

             in: torch.Tensor, shape (num_triplets,)
                 Indices of input edge b->a of each triplet b->a<-c
             out: torch.Tensor, shape (num_triplets,)
                 Indices of output edge c->a of each triplet b->a<-c
             out_agg: torch.Tensor, shape (num_triplets,)
                 Indices enumerating the intermediate edges of each output edge.
                 Used for creating a padded matrix and aggregating via matmul.
   :rtype: Dictionary containing the entries


.. py:function:: get_mixed_triplets(graph_in, graph_out, num_atoms, to_outedge=False, return_adj=False, return_agg_idx=False)

   Get all output edges (ingoing or outgoing) for each incoming edge.
   It is possible that in atom=out atom, as long as the edges are distinct
   (i.e. they stem from different unit cells). In edges and out edges stem
   from separate graphs (hence "mixed") with shared atoms.

   :param graph_in: Contains the input graph's edge_index and cell_offset.
   :type graph_in: dict of torch.Tensor
   :param graph_out: Contains the output graph's edge_index and cell_offset.
                     Input and output graphs use the same atoms, but different edges.
   :type graph_out: dict of torch.Tensor
   :param num_atoms: Total number of atoms.
   :type num_atoms: int
   :param to_outedge: Whether to map the output to the atom's outgoing edges a->c
                      instead of the ingoing edges c->a.
   :type to_outedge: bool
   :param return_adj: Whether to output the adjacency (incidence) matrix between output
                      edges and atoms adj_edges.
   :type return_adj: bool
   :param return_agg_idx: Whether to output the indices enumerating the intermediate edges
                          of each output edge.
   :type return_agg_idx: bool

   :returns:

             in: torch.Tensor, shape (num_triplets,)
                 Indices of input edges
             out: torch.Tensor, shape (num_triplets,)
                 Indices of output edges
             adj_edges: SparseTensor, shape (num_edges, num_atoms)
                 Adjacency (incidence) matrix between output edges and atoms,
                 with values specifying the input edges.
                 Only returned if return_adj is True.
             out_agg: torch.Tensor, shape (num_triplets,)
                 Indices enumerating the intermediate edges of each output edge.
                 Used for creating a padded matrix and aggregating via matmul.
                 Only returned if return_agg_idx is True.
   :rtype: Dictionary containing the entries


.. py:function:: get_quadruplets(main_graph, qint_graph, num_atoms)

   Get all d->b for each edge c->a and connection b->a
   Careful about periodic images!
   Separate interaction cutoff not supported.

   :param main_graph: Contains the main graph's edge_index and cell_offset.
                      The main graph defines which edges are embedded.
   :type main_graph: dict of torch.Tensor
   :param qint_graph: Contains the quadruplet interaction graph's edge_index and
                      cell_offset. main_graph and qint_graph use the same atoms,
                      but different edges.
   :type qint_graph: dict of torch.Tensor
   :param num_atoms: Total number of atoms.
   :type num_atoms: int

   :returns:

             triplet_in['in']: torch.Tensor, shape (nTriplets,)
                 Indices of input edge d->b in triplet d->b->a.
             triplet_in['out']: torch.Tensor, shape (nTriplets,)
                 Interaction indices of output edge b->a in triplet d->b->a.
             triplet_out['in']: torch.Tensor, shape (nTriplets,)
                 Interaction indices of input edge b->a in triplet c->a<-b.
             triplet_out['out']: torch.Tensor, shape (nTriplets,)
                 Indices of output edge c->a in triplet c->a<-b.
             out: torch.Tensor, shape (nQuadruplets,)
                 Indices of output edge c->a in quadruplet
             trip_in_to_quad: torch.Tensor, shape (nQuadruplets,)
                 Indices to map from input triplet d->b->a
                 to quadruplet d->b->a<-c.
             trip_out_to_quad: torch.Tensor, shape (nQuadruplets,)
                 Indices to map from output triplet c->a<-b
                 to quadruplet d->b->a<-c.
             out_agg: torch.Tensor, shape (num_triplets,)
                 Indices enumerating the intermediate edges of each output edge.
                 Used for creating a padded matrix and aggregating via matmul.
   :rtype: Dictionary containing the entries


