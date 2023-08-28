from functools import partial

import torch
from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge

from ocpmodels.models.gemnet_oc_mt.goc_graph import (
    Cutoffs,
    Graph,
    MaxNeighbors,
    generate_graph,
    subselect_graph,
    tag_mask,
)

from .config import MultiTaskConfig


def _process_aint_graph(
    graph: Graph,
    config: MultiTaskConfig,
    *,
    training: bool,
):
    if config.edge_dropout:
        graph["edge_index"], mask = dropout_edge(
            graph["edge_index"],
            p=config.edge_dropout,
            training=training,
        )
        graph["distance"] = graph["distance"][mask]
        graph["vector"] = graph["vector"][mask]
        graph["cell_offset"] = graph["cell_offset"][mask]

        if "id_swap_edge_index" in graph:
            graph["id_swap_edge_index"] = graph["id_swap_edge_index"][mask]

    return graph


def _pre_generate_graph_transform(
    data: Data,
    config: MultiTaskConfig,
    *,
    training: bool,
):
    if config.node_dropout and training:
        node_mask = (
            torch.rand(data.pos.shape[0], device=data.pos.device)
            > config.node_dropout
        )  # (n_nodes,)
        # make sure that at least four nodes are left
        if (n_masked := node_mask.sum()) < 4:
            (unmasked_idx,) = torch.nonzero(~node_mask, as_tuple=True)
            node_mask[unmasked_idx[: 4 - n_masked]] = True
            assert node_mask.sum() >= 4, f"{node_mask.sum()=} < 4"

        data.pos = data.pos[node_mask]
        data.force = data.force[node_mask]
        data.atomic_numbers = data.atomic_numbers[node_mask]
        data.fixed = data.fixed[node_mask]
        data.tags = data.tags[node_mask]
        data.natoms = node_mask.sum().item()

    return data


def _generate_graphs(
    data: Data,
    config: MultiTaskConfig,
    cutoffs: Cutoffs,
    max_neighbors: MaxNeighbors,
    pbc: bool,
    *,
    training: bool,
):
    aint_graph = generate_graph(
        data, cutoff=cutoffs.aint, max_neighbors=max_neighbors.aint, pbc=pbc
    )
    aint_graph = _process_aint_graph(aint_graph, config, training=training)
    subselect = partial(
        subselect_graph,
        data,
        aint_graph,
        cutoff_orig=cutoffs.aint,
        max_neighbors_orig=max_neighbors.aint,
    )
    main_graph = subselect(cutoffs.main, max_neighbors.main)
    aeaint_graph = subselect(cutoffs.aeaint, max_neighbors.aeaint)
    qint_graph = subselect(cutoffs.qint, max_neighbors.qint)

    # We can't do this at the data level: This is because the batch collate_fn doesn't know
    # that it needs to increment the "id_swap" indices as it collates the data.
    # So we do this at the graph level (which is done in the GemNetOC `get_graphs_and_indices` method).
    # main_graph = symmetrize_edges(main_graph, num_atoms=data.pos.shape[0])
    qint_graph = tag_mask(
        data, qint_graph, tags=self.config.backbone.qint_tags
    )

    graphs = {
        "main": main_graph,
        "a2a": aint_graph,
        "a2ee2a": aeaint_graph,
        "qint": qint_graph,
    }

    for graph_type, graph in graphs.items():
        graph["num_neighbors"] = graph["edge_index"].shape[1]
        for key, value in graph.items():
            setattr(data, f"{graph_type}_{key}", value)

    return data


def oc20_transform(data: Data, *, config: MultiTaskConfig, training: bool):
    # convert back these keys into required format for collation
    data.natoms = int(
        data.natoms.item() if torch.is_tensor(data) else data.natoms
    )

    data.atomic_numbers = data.atomic_numbers.long()
    data.tags = data.tags.long()
    try:
        if not torch.is_tensor(data.y):
            data.y = torch.tensor(data.y)
        data.y = data.y.view(-1)
    except:
        if not torch.is_tensor(data.y_relaxed):
            data.y_relaxed = torch.tensor(data.y_relaxed)
        data.y = data.y_relaxed.view(-1)
    data.name = "oc20"

    data = _generate_graphs(
        data,
        config,
        cutoffs=Cutoffs.from_constant(12.0),
        max_neighbors=MaxNeighbors.from_goc_base_proportions(30),
        pbc=True,
        training=training,
    )
    return data


def oc22_transform(data: Data, *, config: MultiTaskConfig, training: bool):
    # convert back these keys into required format for collation
    data.natoms = int(
        data.natoms.item() if torch.is_tensor(data) else data.natoms
    )

    data.atomic_numbers = data.atomic_numbers.long()
    data.tags = data.tags.long()
    try:
        data.y = torch.tensor(float(data.y)).view(-1)
    except:
        data.y = torch.tensor(float(data.y_relaxed)).view(-1)
    data.name = "oc22"

    data = _generate_graphs(
        data,
        config,
        cutoffs=Cutoffs.from_constant(12.0),
        max_neighbors=MaxNeighbors.from_goc_base_proportions(30),
        pbc=True,
        training=training,
    )
    return data


def _set_inf_cell(data: Data, max_length: float = 1000.0):
    data.cell = (torch.eye(3) * max_length).unsqueeze(dim=0)
    return data


def ani1x_transform(data: Data, *, config: MultiTaskConfig, training: bool):
    data.y = data.y.view(-1).float()
    if not hasattr(data, "sid"):
        data.sid = data.absolute_idx
    if not hasattr(data, "natoms"):
        data.natoms = data.num_nodes

    # data.fixed = torch.ones(data.natoms)
    data.fixed = torch.zeros(data.natoms, dtype=torch.bool)

    data.tags = 2 * torch.ones(data.natoms)
    data.tags = data.tags.long()
    data.name = "ani1x"

    data = _set_inf_cell(data)
    data = _generate_graphs(
        data,
        config,
        cutoffs=Cutoffs.from_constant(8.0),
        max_neighbors=MaxNeighbors.from_goc_base_proportions(30),
        pbc=False,
        training=training,
    )
    return data


def transition1x_transform(
    data: Data, *, config: MultiTaskConfig, training: bool
):
    data.y = data.y.view(-1).float()
    if not hasattr(data, "sid"):
        data.sid = data.absolute_idx
    if not hasattr(data, "natoms"):
        data.natoms = data.num_nodes

    # data.fixed = torch.ones(data.natoms)
    data.fixed = torch.zeros(data.natoms, dtype=torch.bool)

    data.tags = 2 * torch.ones(data.natoms)
    data.tags = data.tags.long()
    data.name = "transition1x"

    data = _set_inf_cell(data)
    data = _generate_graphs(
        data,
        config,
        cutoffs=Cutoffs.from_constant(8.0),
        max_neighbors=MaxNeighbors.from_goc_base_proportions(30),
        pbc=False,
        training=training,
    )
    return data
