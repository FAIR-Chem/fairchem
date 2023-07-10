"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch_scatter import segment_coo
from torch_sparse import SparseTensor

from .utils import get_inner_idx, masked_select_sparsetensor_flat


def get_triplets(graph, num_atoms: int):
    """
    Get all input edges b->a for each output edge c->a.
    It is possible that b=c, as long as the edges are distinct
    (i.e. atoms b and c stem from different unit cells).

    Arguments
    ---------
    graph: dict of torch.Tensor
        Contains the graph's edge_index.
    num_atoms: int
        Total number of atoms.

    Returns
    -------
    Dictionary containing the entries:
        in: torch.Tensor, shape (num_triplets,)
            Indices of input edge b->a of each triplet b->a<-c
        out: torch.Tensor, shape (num_triplets,)
            Indices of output edge c->a of each triplet b->a<-c
        out_agg: torch.Tensor, shape (num_triplets,)
            Indices enumerating the intermediate edges of each output edge.
            Used for creating a padded matrix and aggregating via matmul.
    """
    idx_s, idx_t = graph["edge_index"]  # c->a (source=c, target=a)
    num_edges = idx_s.size(0)

    value = torch.arange(num_edges, device=idx_s.device, dtype=idx_s.dtype)
    # Possibly contains multiple copies of the same edge (for periodic interactions)
    adj = SparseTensor(
        row=idx_t,
        col=idx_s,
        value=value,
        sparse_sizes=(num_atoms, num_atoms),
    )
    adj_edges = adj[idx_t]

    # Edge indices (b->a, c->a) for triplets.
    idx = {}
    idx["in"] = adj_edges.storage.value()
    idx["out"] = adj_edges.storage.row()

    # Remove self-loop triplets
    # Compare edge indices, not atom indices to correctly handle periodic interactions
    mask = idx["in"] != idx["out"]
    idx["in"] = idx["in"][mask]
    idx["out"] = idx["out"][mask]

    # idx['out'] has to be sorted for this
    idx["out_agg"] = get_inner_idx(idx["out"], dim_size=num_edges)

    return idx


def get_mixed_triplets(
    graph_in,
    graph_out,
    num_atoms,
    to_outedge=False,
    return_adj=False,
    return_agg_idx=False,
):
    """
    Get all output edges (ingoing or outgoing) for each incoming edge.
    It is possible that in atom=out atom, as long as the edges are distinct
    (i.e. they stem from different unit cells). In edges and out edges stem
    from separate graphs (hence "mixed") with shared atoms.

    Arguments
    ---------
    graph_in: dict of torch.Tensor
        Contains the input graph's edge_index and cell_offset.
    graph_out: dict of torch.Tensor
        Contains the output graph's edge_index and cell_offset.
        Input and output graphs use the same atoms, but different edges.
    num_atoms: int
        Total number of atoms.
    to_outedge: bool
        Whether to map the output to the atom's outgoing edges a->c
        instead of the ingoing edges c->a.
    return_adj: bool
        Whether to output the adjacency (incidence) matrix between output
        edges and atoms adj_edges.
    return_agg_idx: bool
        Whether to output the indices enumerating the intermediate edges
        of each output edge.

    Returns
    -------
    Dictionary containing the entries:
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
    """
    idx_out_s, idx_out_t = graph_out["edge_index"]
    # c->a (source=c, target=a)
    idx_in_s, idx_in_t = graph_in["edge_index"]
    num_edges = idx_out_s.size(0)

    value_in = torch.arange(
        idx_in_s.size(0), device=idx_in_s.device, dtype=idx_in_s.dtype
    )
    # This exploits that SparseTensor can have multiple copies of the same edge!
    adj_in = SparseTensor(
        row=idx_in_t,
        col=idx_in_s,
        value=value_in,
        sparse_sizes=(num_atoms, num_atoms),
    )
    if to_outedge:
        adj_edges = adj_in[idx_out_s]
    else:
        adj_edges = adj_in[idx_out_t]

    # Edge indices (b->a, c->a) for triplets.
    idx_in = adj_edges.storage.value()
    idx_out = adj_edges.storage.row()

    # Remove self-loop triplets c->a<-c or c<-a<-c
    # Check atom as well as cell offset
    if to_outedge:
        idx_atom_in = idx_in_s[idx_in]
        idx_atom_out = idx_out_t[idx_out]
        cell_offsets_sum = (
            graph_out["cell_offset"][idx_out] + graph_in["cell_offset"][idx_in]
        )
    else:
        idx_atom_in = idx_in_s[idx_in]
        idx_atom_out = idx_out_s[idx_out]
        cell_offsets_sum = (
            graph_out["cell_offset"][idx_out] - graph_in["cell_offset"][idx_in]
        )
    mask = (idx_atom_in != idx_atom_out) | torch.any(
        cell_offsets_sum != 0, dim=-1
    )

    idx = {}
    if return_adj:
        idx["adj_edges"] = masked_select_sparsetensor_flat(adj_edges, mask)
        idx["in"] = idx["adj_edges"].storage.value().clone()
        idx["out"] = idx["adj_edges"].storage.row()
    else:
        idx["in"] = idx_in[mask]
        idx["out"] = idx_out[mask]

    if return_agg_idx:
        # idx['out'] has to be sorted
        idx["out_agg"] = get_inner_idx(idx["out"], dim_size=num_edges)

    return idx


def get_quadruplets(
    main_graph,
    qint_graph,
    num_atoms,
):
    """
    Get all d->b for each edge c->a and connection b->a
    Careful about periodic images!
    Separate interaction cutoff not supported.

    Arguments
    ---------
    main_graph: dict of torch.Tensor
        Contains the main graph's edge_index and cell_offset.
        The main graph defines which edges are embedded.
    qint_graph: dict of torch.Tensor
        Contains the quadruplet interaction graph's edge_index and
        cell_offset. main_graph and qint_graph use the same atoms,
        but different edges.
    num_atoms: int
        Total number of atoms.

    Returns
    -------
    Dictionary containing the entries:
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
    """
    idx_s, _ = main_graph["edge_index"]
    idx_qint_s, _ = qint_graph["edge_index"]
    # c->a (source=c, target=a)
    num_edges = idx_s.size(0)
    idx = {}

    idx["triplet_in"] = get_mixed_triplets(
        main_graph,
        qint_graph,
        num_atoms,
        to_outedge=True,
        return_adj=True,
    )
    # Input triplets d->b->a

    idx["triplet_out"] = get_mixed_triplets(
        qint_graph,
        main_graph,
        num_atoms,
        to_outedge=False,
    )
    # Output triplets c->a<-b

    # ---------------- Quadruplets -----------------
    # Repeat indices by counting the number of input triplets per
    # intermediate edge ba. segment_coo assumes sorted idx['triplet_in']['out']
    ones = (
        idx["triplet_in"]["out"]
        .new_ones(1)
        .expand_as(idx["triplet_in"]["out"])
    )
    num_trip_in_per_inter = segment_coo(
        ones, idx["triplet_in"]["out"], dim_size=idx_qint_s.size(0)
    )

    num_trip_out_per_inter = num_trip_in_per_inter[idx["triplet_out"]["in"]]
    idx["out"] = torch.repeat_interleave(
        idx["triplet_out"]["out"], num_trip_out_per_inter
    )
    idx_inter = torch.repeat_interleave(
        idx["triplet_out"]["in"], num_trip_out_per_inter
    )
    idx["trip_out_to_quad"] = torch.repeat_interleave(
        torch.arange(
            len(idx["triplet_out"]["out"]),
            device=idx_s.device,
            dtype=idx_s.dtype,
        ),
        num_trip_out_per_inter,
    )

    # Generate input indices by using the adjacency
    # matrix idx['triplet_in']['adj_edges']
    idx["triplet_in"]["adj_edges"].set_value_(
        torch.arange(
            len(idx["triplet_in"]["in"]),
            device=idx_s.device,
            dtype=idx_s.dtype,
        ),
        layout="coo",
    )
    adj_trip_in_per_trip_out = idx["triplet_in"]["adj_edges"][
        idx["triplet_out"]["in"]
    ]
    # Rows in adj_trip_in_per_trip_out are intermediate edges ba
    idx["trip_in_to_quad"] = adj_trip_in_per_trip_out.storage.value()
    idx_in = idx["triplet_in"]["in"][idx["trip_in_to_quad"]]

    # Remove quadruplets with c == d
    # Triplets should already ensure that a != d and b != c
    # Compare atom indices and cell offsets
    idx_atom_c = idx_s[idx["out"]]
    idx_atom_d = idx_s[idx_in]

    cell_offset_cd = (
        main_graph["cell_offset"][idx_in]
        + qint_graph["cell_offset"][idx_inter]
        - main_graph["cell_offset"][idx["out"]]
    )
    mask_cd = (idx_atom_c != idx_atom_d) | torch.any(
        cell_offset_cd != 0, dim=-1
    )

    idx["out"] = idx["out"][mask_cd]
    idx["trip_out_to_quad"] = idx["trip_out_to_quad"][mask_cd]
    idx["trip_in_to_quad"] = idx["trip_in_to_quad"][mask_cd]

    # idx['out'] has to be sorted for this
    idx["out_agg"] = get_inner_idx(idx["out"], dim_size=num_edges)

    return idx
