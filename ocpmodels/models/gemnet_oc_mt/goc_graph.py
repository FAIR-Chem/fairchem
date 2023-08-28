from dataclasses import dataclass
from functools import wraps
from typing import Callable, ParamSpec, TypedDict, cast

import numpy as np
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.nn import radius_graph
from torch_geometric.utils import sort_edge_index
from torch_scatter import segment_coo
from typing_extensions import NotRequired

from ocpmodels.common.utils import get_pbc_distances

from .radius_graph import radius_graph_pbc
from .utils import (
    get_edge_id,
    get_max_neighbors_mask,
    mask_neighbors,
    repeat_blocks,
)


class Graph(TypedDict):
    edge_index: torch.Tensor  # 2 e
    distance: torch.Tensor  # e
    vector: torch.Tensor  # e 3
    cell_offset: torch.Tensor  # e 3
    num_neighbors: torch.Tensor  # b

    cutoff: torch.Tensor  # b
    max_neighbors: torch.Tensor  # b

    id_swap_edge_index: NotRequired[torch.Tensor]  # e


@dataclass(frozen=True, kw_only=True)
class Cutoffs:
    main: float
    aeaint: float
    qint: float
    aint: float

    @classmethod
    def from_constant(cls, value: float):
        return cls(main=value, aeaint=value, qint=value, aint=value)


@dataclass(frozen=True, kw_only=True)
class MaxNeighbors:
    main: int
    aeaint: int
    qint: int
    aint: int

    @classmethod
    def from_goc_base_proportions(cls, max_neighbors: int):
        """
        GOC base proportions:
            max_neighbors: 30
            max_neighbors_qint: 8
            max_neighbors_aeaint: 20
            max_neighbors_aint: 1000
        """
        return cls(
            main=max_neighbors,
            aeaint=int(max_neighbors * 20 / 30),
            qint=int(max_neighbors * 8 / 30),
            aint=int(max_neighbors * 1000 / 30),
        )


def _select_symmetric_edges(tensor, mask, reorder_idx, opposite_neg):
    """Use a mask to remove values of removed edges and then
    duplicate the values for the correct edge direction.

    Arguments
    ---------
    tensor: torch.Tensor
        Values to symmetrize for the new tensor.
    mask: torch.Tensor
        Mask defining which edges go in the correct direction.
    reorder_idx: torch.Tensor
        Indices defining how to reorder the tensor values after
        concatenating the edge values of both directions.
    opposite_neg: bool
        Whether the edge in the opposite direction should use the
        negative tensor value.

    Returns
    -------
    tensor_ordered: torch.Tensor
        A tensor with symmetrized values.
    """
    # Mask out counter-edges
    tensor_directed = tensor[mask]
    # Concatenate counter-edges after normal edges
    sign = 1 - 2 * opposite_neg
    tensor_cat = torch.cat([tensor_directed, sign * tensor_directed])
    # Reorder everything so the edges of every image are consecutive
    tensor_ordered = tensor_cat[reorder_idx]
    return tensor_ordered


def symmetrize_edges(graph: Graph, num_atoms: int):
    """
    Symmetrize edges to ensure existence of counter-directional edges.

    Some edges are only present in one direction in the data,
    since every atom has a maximum number of neighbors.
    We only use i->j edges here. So we lose some j->i edges
    and add others by making it symmetric.
    """
    new_graph = graph.copy()

    # Generate mask
    mask_sep_atoms = graph["edge_index"][0] < graph["edge_index"][1]
    # Distinguish edges between the same (periodic) atom by ordering the cells
    cell_earlier = (
        (graph["cell_offset"][:, 0] < 0)
        | (
            (graph["cell_offset"][:, 0] == 0)
            & (graph["cell_offset"][:, 1] < 0)
        )
        | (
            (graph["cell_offset"][:, 0] == 0)
            & (graph["cell_offset"][:, 1] == 0)
            & (graph["cell_offset"][:, 2] < 0)
        )
    )
    mask_same_atoms = graph["edge_index"][0] == graph["edge_index"][1]
    mask_same_atoms &= cell_earlier
    mask = mask_sep_atoms | mask_same_atoms

    # Mask out counter-edges
    edge_index_directed = graph["edge_index"][
        mask[None, :].expand(2, -1)
    ].view(2, -1)

    # Concatenate counter-edges after normal edges
    edge_index_cat = torch.cat(
        [edge_index_directed, edge_index_directed.flip(0)],
        dim=1,
    )

    # Count remaining edges per image
    batch_edge = torch.repeat_interleave(
        torch.arange(
            graph["num_neighbors"].size(0),
            device=graph["edge_index"].device,
        ),
        graph["num_neighbors"],
    )
    batch_edge = batch_edge[mask]
    # segment_coo assumes sorted batch_edge
    # Factor 2 since this is only one half of the edges
    ones = batch_edge.new_ones(1).expand_as(batch_edge)
    new_graph["num_neighbors"] = 2 * segment_coo(
        ones, batch_edge, dim_size=graph["num_neighbors"].size(0)
    )

    # Create indexing array
    edge_reorder_idx = repeat_blocks(
        torch.div(new_graph["num_neighbors"], 2, rounding_mode="floor"),
        repeats=2,
        continuous_indexing=True,
        repeat_inc=edge_index_directed.size(1),
    )

    # Reorder everything so the edges of every image are consecutive
    new_graph["edge_index"] = edge_index_cat[:, edge_reorder_idx]
    new_graph["cell_offset"] = _select_symmetric_edges(
        graph["cell_offset"], mask, edge_reorder_idx, True
    )
    new_graph["distance"] = _select_symmetric_edges(
        graph["distance"], mask, edge_reorder_idx, False
    )
    new_graph["vector"] = _select_symmetric_edges(
        graph["vector"], mask, edge_reorder_idx, True
    )

    # Indices for swapping c->a and a->c (for symmetric MP)
    # To obtain these efficiently and without any index assumptions,
    # we get order the counter-edge IDs and then
    # map this order back to the edge IDs.
    # Double argsort gives the desired mapping
    # from the ordered tensor to the original tensor.
    edge_ids = get_edge_id(
        new_graph["edge_index"], new_graph["cell_offset"], num_atoms
    )
    order_edge_ids = torch.argsort(edge_ids)
    inv_order_edge_ids = torch.argsort(order_edge_ids)
    edge_ids_counter = get_edge_id(
        new_graph["edge_index"].flip(0),
        -new_graph["cell_offset"],
        num_atoms,
    )
    order_edge_ids_counter = torch.argsort(edge_ids_counter)
    id_swap_edge_index = order_edge_ids_counter[inv_order_edge_ids]

    new_graph["id_swap_edge_index"] = id_swap_edge_index

    return cast(Graph, new_graph)


def tag_mask(data: Data, graph: Graph, *, tags: list[int]):
    tags_ = torch.tensor(tags, dtype=torch.long, device=data.tags.device)

    # Only use quadruplets for certain tags
    tags_s = data.tags[graph["edge_index"][0]]
    tags_t = data.tags[graph["edge_index"][1]]
    tag_mask_s = (tags_s[..., None] == tags_).any(dim=-1)
    tag_mask_t = (tags_t[..., None] == tags_).any(dim=-1)
    tag_mask = tag_mask_s | tag_mask_t

    graph["edge_index"] = graph["edge_index"][:, tag_mask]
    graph["cell_offset"] = graph["cell_offset"][tag_mask, :]
    graph["distance"] = graph["distance"][tag_mask]
    graph["vector"] = graph["vector"][tag_mask, :]

    return graph


def _generate_graph(
    data: Data,
    *,
    cutoff: float,
    max_neighbors: int,
    pbc: bool,
):
    if pbc:
        edge_index, cell_offsets, neighbors = radius_graph_pbc(
            data, cutoff, max_neighbors
        )

        out = get_pbc_distances(
            data.pos,
            edge_index,
            data.cell,
            cell_offsets,
            neighbors,
            return_offsets=True,
            return_distance_vec=True,
        )

        edge_index: torch.Tensor = out["edge_index"]
        edge_dist: torch.Tensor = out["distances"]
        cell_offset_distances: torch.Tensor = out["offsets"]
        distance_vec: torch.Tensor = out["distance_vec"]
    else:
        edge_index = radius_graph(
            data.pos,
            r=cutoff,
            batch=data.batch,
            max_num_neighbors=max_neighbors,
        )

        j, i = edge_index
        distance_vec = data.pos[j] - data.pos[i]

        edge_dist = distance_vec.norm(dim=-1)
        cell_offsets = torch.zeros(
            edge_index.shape[1], 3, device=data.pos.device
        )
        cell_offset_distances = torch.zeros_like(
            cell_offsets, device=data.pos.device
        )
        neighbors = edge_index.shape[1]

    return (
        edge_index,
        edge_dist,
        distance_vec,
        cell_offsets,
        cell_offset_distances,
        neighbors,
    )


def generate_graph(
    data: Data,
    *,
    cutoff: float,
    max_neighbors: int,
    pbc: bool,
    symmetrize: bool = False,
    filter_tags: list[int] | None = None,
    sort_edges: bool = False,
):
    (
        edge_index,
        edge_dist,
        distance_vec,
        cell_offsets,
        _,  # cell offset distances
        num_neighbors,
    ) = _generate_graph(
        data,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        pbc=pbc,
    )
    # These vectors actually point in the opposite direction.
    # But we want to use col as idx_t for efficient aggregation.
    edge_vector = -distance_vec / edge_dist[:, None]
    # cell_offsets = -cell_offsets  # a - c + offset

    graph: Graph = {
        "edge_index": edge_index,
        "distance": edge_dist,
        "vector": edge_vector,
        "cell_offset": cell_offsets,
        "num_neighbors": num_neighbors,
        "cutoff": torch.tensor(
            cutoff, dtype=torch.float, device=data.pos.device
        ),
        "max_neighbors": torch.tensor(
            max_neighbors, dtype=torch.long, device=data.pos.device
        ),
    }

    if symmetrize:
        graph = symmetrize_edges(graph, data.pos.shape[0])

    if filter_tags is not None:
        graph = tag_mask(data, graph, tags=filter_tags)

    if sort_edges:
        graph["edge_index"], [
            graph["distance"],
            graph["vector"],
            graph["cell_offset"],
        ] = sort_edge_index(
            graph["edge_index"],
            [
                graph["distance"],
                graph["vector"],
                graph["cell_offset"],
            ],
            num_nodes=data.pos.shape[0],
            sort_by_row=False,
        )

        graph["num_neighbors"] = torch.full_like(
            graph["num_neighbors"], graph["edge_index"].shape[1]
        )

    return graph


def _subselect_edges(
    data: Data,
    graph: Graph,
    cutoff: float | None = None,
    max_neighbors: int | None = None,
):
    """Subselect edges using a stricter cutoff and max_neighbors."""
    subgraph = graph.copy()

    if cutoff is not None:
        edge_mask = subgraph["distance"] <= cutoff

        subgraph["edge_index"] = subgraph["edge_index"][:, edge_mask]
        subgraph["cell_offset"] = subgraph["cell_offset"][edge_mask]
        subgraph["num_neighbors"] = mask_neighbors(
            subgraph["num_neighbors"], edge_mask
        )
        subgraph["distance"] = subgraph["distance"][edge_mask]
        subgraph["vector"] = subgraph["vector"][edge_mask]

    if max_neighbors is not None:
        subgraph["max_neighbors"] = torch.tensor(
            max_neighbors, dtype=torch.long, device=data.pos.device
        )
        edge_mask, subgraph["num_neighbors"] = get_max_neighbors_mask(
            natoms=torch.tensor(
                [data.natoms], dtype=torch.long, device=data.pos.device
            )
            if not torch.is_tensor(data.natoms)
            else data.natoms.view(-1),
            index=subgraph["edge_index"][1],
            atom_distance=subgraph["distance"],
            max_num_neighbors_threshold=max_neighbors,
        )
        if not torch.all(edge_mask):
            subgraph["edge_index"] = subgraph["edge_index"][:, edge_mask]
            subgraph["cell_offset"] = subgraph["cell_offset"][edge_mask]
            subgraph["distance"] = subgraph["distance"][edge_mask]
            subgraph["vector"] = subgraph["vector"][edge_mask]

    empty_image = subgraph["num_neighbors"] == 0
    if torch.any(empty_image):
        raise ValueError(f"An image has no neighbors: {data}")
    return subgraph


def subselect_graph(
    data: Data,
    graph: Graph,
    cutoff: float,
    max_neighbors: int,
    cutoff_orig: float,
    max_neighbors_orig: int,
):
    """If the new cutoff and max_neighbors is different from the original,
    subselect the edges of a given graph.
    """
    # Check if embedding edges are different from interaction edges
    if np.isclose(cutoff, cutoff_orig):
        select_cutoff = None
    else:
        select_cutoff = cutoff
    if max_neighbors == max_neighbors_orig:
        select_neighbors = None
    else:
        select_neighbors = max_neighbors

    graph = _subselect_edges(
        data=data,
        graph=graph,
        cutoff=select_cutoff,
        max_neighbors=select_neighbors,
    )
    return graph


def generate_graphs(
    data: Data,
    *,
    cutoffs: Cutoffs | Callable[[Data], Cutoffs],
    max_neighbors: MaxNeighbors | Callable[[Data], MaxNeighbors],
    pbc: bool,
    symmetrize_main: bool = False,
    qint_tags: list[int] | None = [1, 2],
):
    if callable(cutoffs):
        cutoffs = cutoffs(data)
    if callable(max_neighbors):
        max_neighbors = max_neighbors(data)

    assert cutoffs.main <= cutoffs.aint
    assert cutoffs.aeaint <= cutoffs.aint
    assert cutoffs.qint <= cutoffs.aint

    assert max_neighbors.main <= max_neighbors.aint
    assert max_neighbors.aeaint <= max_neighbors.aint
    assert max_neighbors.qint <= max_neighbors.aint

    main_graph = generate_graph(
        data,
        cutoff=cutoffs.main,
        max_neighbors=max_neighbors.main,
        pbc=pbc,
        symmetrize=symmetrize_main,
    )
    a2a_graph = generate_graph(
        data,
        cutoff=cutoffs.aint,
        max_neighbors=max_neighbors.aint,
        pbc=pbc,
    )
    a2ee2a_graph = generate_graph(
        data,
        cutoff=cutoffs.aeaint,
        max_neighbors=max_neighbors.aeaint,
        pbc=pbc,
    )
    qint_graph = generate_graph(
        data,
        cutoff=cutoffs.qint,
        max_neighbors=max_neighbors.qint,
        pbc=pbc,
        filter_tags=qint_tags,
    )

    graphs = {
        "main": main_graph,
        "a2a": a2a_graph,
        "a2ee2a": a2ee2a_graph,
        "qint": qint_graph,
    }
    return graphs


P = ParamSpec("P")


def with_goc_graphs(
    cutoffs: Cutoffs | Callable[[Data], Cutoffs],
    max_neighbors: MaxNeighbors | Callable[[Data], MaxNeighbors],
    pbc: bool,
    symmetrize_main: bool = False,
    qint_tags: list[int] | None = [1, 2],
):
    def decorator(func: Callable[P, Data]):
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Data:
            data = func(*args, **kwargs)

            graphs = generate_graphs(
                data,
                cutoffs=cutoffs,
                max_neighbors=max_neighbors,
                pbc=pbc,
                symmetrize_main=symmetrize_main,
                qint_tags=qint_tags,
            )
            for graph_type, graph in graphs.items():
                for key, value in graph.items():
                    setattr(data, f"{graph_type}_{key}", value)

            return data

        return wrapper

    return decorator


class Graphs(TypedDict):
    main: Graph
    a2a: Graph
    a2ee2a: Graph
    qint: Graph


GRAPH_TYPES = ["main", "a2a", "a2ee2a", "qint"]


def graphs_from_batch(data: Data | Batch) -> Graphs:
    global GRAPH_TYPES

    graphs = {
        graph_type: {
            "edge_index": getattr(data, f"{graph_type}_edge_index"),
            "distance": getattr(data, f"{graph_type}_distance"),
            "vector": getattr(data, f"{graph_type}_vector"),
            "cell_offset": getattr(data, f"{graph_type}_cell_offset"),
            "num_neighbors": getattr(
                data, f"{graph_type}_num_neighbors", None
            ),
            "cutoff": getattr(data, f"{graph_type}_cutoff", None),
            "max_neighbors": getattr(
                data, f"{graph_type}_max_neighbors", None
            ),
            "id_swap_edge_index": getattr(
                data, f"{graph_type}_id_swap_edge_index", None
            ),
        }
        for graph_type in GRAPH_TYPES
    }
    # remove None values
    graphs = {
        graph_type: {
            key: value for key, value in graph.items() if value is not None
        }
        for graph_type, graph in graphs.items()
    }
    return cast(Graphs, graphs)


def graphs_to_batch(data: Data | Batch, graphs: Graphs):
    global GRAPH_TYPES

    for graph_type in GRAPH_TYPES:
        for key, value in graphs[graph_type].items():
            setattr(data, f"{graph_type}_{key}", value)

    return data
