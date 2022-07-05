""" Rewire each 3D molecular graph
"""

from copy import deepcopy
from time import time

import torch
from torch import cat, isin, tensor, where
from torch_geometric.utils import coalesce, remove_self_loops


def remove_tag0_nodes(data):
    """Delete sub-surface (tag == 0) nodes and rewire accordingly the graph

    Args:
        data (torch_geometric.Data): the data batch to re-wire

    Returns:
        torch_geometric.Data: the data rewired data batch
    """
    # non sub-surface atoms
    non_sub = torch.where(data.tags != 0)[0]
    src_is_not_sub = torch.isin(data.edge_index[0], non_sub)
    target_is_not_sub = torch.isin(data.edge_index[1], non_sub)
    neither_is_sub = src_is_not_sub * target_is_not_sub

    # per-atom tensors
    data.pos = data.pos[non_sub, :]
    data.atomic_numbers = data.atomic_numbers[non_sub]
    data.batch = data.batch[non_sub]
    data.force = data.force[non_sub, :]
    data.fixed = data.fixed[non_sub]
    data.tags = data.tags[non_sub]
    data.pos_relaxed = data.pos_relaxed[non_sub, :]

    # per-edge tensors
    data.edge_index = data.edge_index[:, neither_is_sub]
    data.cell_offsets = data.cell_offsets[neither_is_sub, :]
    data.distances = data.distances[neither_is_sub]
    # re-index adj matrix, given some nodes were deleted
    num_nodes = data.natoms.sum().item()
    mask = torch.zeros(num_nodes, dtype=torch.bool, device=data.edge_index.device)
    mask[non_sub] = 1
    assoc = torch.full((num_nodes,), -1, dtype=torch.long, device=mask.device)
    assoc[mask] = torch.arange(mask.sum(), device=assoc.device)
    data.edge_index = assoc[data.edge_index]

    # per-graph tensors
    batch_size = max(data.batch).item() + 1
    data.ptr = torch.tensor(
        [0] + [data.natoms[:i].sum() for i in range(1, batch_size + 1)],
        dtype=data.ptr.dtype,
        device=data.ptr.device,
    )
    data.natoms = torch.tensor(
        [(data.batch == i).sum() for i in range(batch_size)],
        dtype=data.natoms.dtype,
        device=data.natoms.device,
    )
    _, data.neighbors = torch.unique(
        data.batch[data.edge_index[0, :]], return_counts=True
    )

    return data


def one_supernode_per_graph(data):
    """Generate a single supernode representing all tag0 atoms

    Args:
        data (data.Data): single batch of graphs
    """
    b = deepcopy(data)
    t0 = time()

    batch_size = max(data.batch).item() + 1
    device = data.edge_index.device

    # ids of sub-surface nodes, per batch
    sub_nodes = [
        where((data.tags == 0) * (data.batch == i))[0] for i in range(batch_size)
    ]

    # idem for non-sub-surface nodes
    non_sub_nodes = [
        where((data.tags != 0) * (data.batch == i))[0] for i in range(batch_size)
    ]

    # super node index per batch: they are last in their batch (after removal of tag0 nodes)
    new_sn_ids = [
        sum([len(nsn) for nsn in non_sub_nodes[: i + 1]]) + i for i in range(batch_size)
    ]
    # define new number of atoms per batch
    data.ptr = tensor(
        [0] + [nsi + 1 for nsi in new_sn_ids], dtype=data.ptr.dtype, device=device
    )
    data.natoms = data.ptr[1:] - data.ptr[:-1]

    # super node position for a batch is the mean of its aggregates
    sn_pos = [data.pos[sub_nodes[i]].mean(0) for i in range(batch_size)]
    # target relaxed position is the mean of the super node's aggregates
    sn_pos_relaxed = [data.pos_relaxed[sub_nodes[i]].mean(0) for i in range(batch_size)]
    # the super node force is the mean of the force applied to its aggregates
    sn_force = [data.force[sub_nodes[i]].mean(0) for i in range(batch_size)]

    # per-atom tensors

    # learn a new embedding to each supernode
    data.atomic_numbers = cat(
        [
            cat([data.atomic_numbers[non_sub_nodes[i]], tensor([84], device=device)])
            for i in range(batch_size)
        ]
    )

    # position excludes sub-surface atoms but includes extra super-nodes
    data.pos = cat(
        [
            cat([data.pos[non_sub_nodes[i]], sn_pos[i][None, :]])
            for i in range(batch_size)
        ]
    )
    data.pos_relaxed = cat(
        [
            cat([data.pos_relaxed[non_sub_nodes[i]], sn_pos_relaxed[i][None, :]])
            for i in range(batch_size)
        ]
    )
    # idem
    data.force = cat(
        [
            cat([data.force[non_sub_nodes[i]], sn_force[i][None, :]])
            for i in range(batch_size)
        ]
    )
    # idem, sn position is fixed
    data.fixed = cat(
        [
            cat(
                [
                    data.fixed[non_sub_nodes[i]],
                    tensor([1.0], dtype=data.fixed.dtype, device=device),
                ]
            )
            for i in range(batch_size)
        ]
    )
    # idem, sn have tag0
    data.tags = cat(
        [
            cat(
                [
                    data.tags[non_sub_nodes[i]],
                    tensor([0], dtype=data.tags.dtype, device=device),
                ]
            )
            for i in range(batch_size)
        ]
    )
    # batch
    data.batch = torch.zeros(data.ptr[-1], dtype=data.batch.dtype, device=device)
    for i, p in enumerate(data.ptr[:-1]):
        data.batch[torch.arange(p, data.ptr[i + 1], dtype=torch.long)] = tensor(
            i, dtype=data.batch.dtype
        )

    # list of edge indices per batch
    ei_batch_ids = [
        (b.ptr[i] <= data.edge_index[0]) * (data.edge_index[0] < b.ptr[i + 1])
        for i in range(batch_size)
    ]
    # boolean: src node is not sub-surface node
    src_is_not_sub = [
        isin(data.edge_index[0][ei_batch_ids[i]], ns)
        for i, ns in enumerate(non_sub_nodes)
    ]
    # boolean: target node is not  sub-surface node
    target_is_not_sub = [
        isin(data.edge_index[1][ei_batch_ids[i]], ns)
        for i, ns in enumerate(non_sub_nodes)
    ]
    # boolean: edges for which NOT both nodes are sub-surface atoms
    # so tag0--tag1/2 + tag1/2--tag1/2 edges
    not_both_are_sub = [
        torch.logical_or(s, t) for s, t in zip(src_is_not_sub, target_is_not_sub)
    ]

    # Adapt cell offsets
    new_cell_offsets = data.cell_offsets[cat(not_both_are_sub)]
    # -----------------------------
    # -----  Graph re-wiring  -----
    # -----------------------------
    # edge-index without tag0-tag0 edges
    ei_not_both = data.edge_index.clone()[:, cat(not_both_are_sub)]
    # store batch index of new adj
    batch_idx_adj = b.batch[ei_not_both][0]
    # number of nodes in this batch: all existing + batch_size supernodes
    num_nodes = b.ptr[-1].item()
    # re-index edges
    mask = torch.zeros(num_nodes, dtype=torch.bool, device=data.edge_index.device)
    mask[cat(non_sub_nodes)] = 1  # mask is 0 for sub-surface atoms
    assoc = torch.full((num_nodes,), -1, dtype=torch.long, device=mask.device)
    assoc[mask] = cat(
        [
            torch.arange(data.ptr[e], data.ptr[e + 1] - 1, device=assoc.device)
            for e in range(batch_size)
        ]
    )
    # re-index only edges for which not both nodes are sub-surface atoms
    ei_sn = assoc[ei_not_both]
    # Adapt cell_offsets: add [0,0,0] for supernode related edges
    new_cell_offsets[isin(ei_sn, torch.tensor(-1)).any(dim=0)] = torch.tensor([0, 0, 0])
    # Replace index -1 by supernode index
    ei_sn = where(
        isin(ei_sn, torch.tensor(-1)), torch.tensor(new_sn_ids)[batch_idx_adj], ei_sn
    )
    # Remove duplicate entries
    ei_sn, new_cell_offsets = coalesce(
        remove_self_loops(ei_sn)[0], edge_attr=new_cell_offsets, reduce="min"
    )
    data.edge_index = ei_sn.to(dtype=data.edge_index.dtype)
    data.cell_offsets = new_cell_offsets.to(dtype=data.cell_offsets.dtype)

    # distances
    distance = torch.sqrt(
        ((data.pos[data.edge_index[0, :]] - data.pos[data.edge_index[1, :]]) ** 2).sum(
            -1
        )
    )
    data.distances = distance.to(dtype=data.distances.dtype)

    # neighbors
    _, data.neighbors = torch.unique(
        data.batch[data.edge_index[0, :]], return_counts=True
    )

    # Time
    tf = time()
    print(f"Total processing time: {tf-t0:.5f}")
    print(f"Total processing time per batch: {(tf-t0) / batch_size:.5f}")


def one_supernode_per_atom_type(data):
    """Create one supernode for each sub-surface atom type
    and remove all such tag-0 atoms.

    Args:
        data (torch_geometric.Data): the data batch to re-wire

    Returns:
        torch_geometric.Data: the data rewired data batch
    """

    batch_size = max(data.batch).item() + 1
    device = data.edge_index.device

    # ids of sub-surface nodes, per batch
    sub_nodes = [
        torch.where((data.tags == 0) * (data.batch == i))[0] for i in range(batch_size)
    ]
    # idem for non-sub-surface nodes
    non_sub_nodes = [
        torch.where((data.tags != 0) * (data.batch == i))[0] for i in range(batch_size)
    ]
    # atom types per supernode
    atom_types = [
        torch.unique(data.atomic_numbers[(data.tags == 0) * (data.batch == i)])
        for i in range(batch_size)
    ]
    # number of supernodes per batch
    num_supernodes = [atom_types[i].shape[0] for i in range(batch_size)]
    # total_num_supernodes = sum(num_supernodes)
    # indexes of nodes belonging to each supernode
    supernodes_composition = [
        [
            torch.where(
                (data.atomic_numbers == an) * (data.tags == 0) * (data.batch == i)
            )[0]
            for an in atom_types[i]
        ]
        for i in range(batch_size)
    ]
    # supernode indexes
    sn_idxes = [
        [data.ptr[1:][i] + sn for sn in range(num_supernodes[i])]
        for i in range(len(num_supernodes))
    ]

    # supernode positions
    supernodes_pos = [
        data.pos[sn, :].mean(0)[None, :]
        for sublist in supernodes_composition
        for sn in sublist
    ]

    ### Compute supernode edge-index
    ei_batch_ids = [
        (data.ptr[i] <= data.edge_index[0]) * (data.edge_index[0] < data.ptr[i + 1])
        for i in range(batch_size)
    ]
    # list of graph level adj.
    ei_batch = [data.edge_index[:, ei_batch_ids[i]] for i in range(batch_size)]

    # Define new edge_index matrix per batch
    for i in range(batch_size):
        for j, sc in enumerate(supernodes_composition[i]):
            ei_batch[i] = torch.where(
                torch.isin(ei_batch[i], sc), sn_idxes[i][j], ei_batch[i]
            )

    # Remove self loops and duplicates
    data.edge_index = [coalesce(remove_self_loops(adj)[0]) for adj in ei_batch]

    # re-index batch adj matrix one by one
    max_num_nodes = 0
    for i in range(batch_size):
        num_nodes = data.ptr[i + 1] + num_supernodes[i]
        mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
        mask[sub_nodes[i]] = 0
        # mask = mask[data.ptr[i]:]
        mask[: data.ptr[i]] = torch.zeros(data.ptr[i], dtype=torch.bool, device=device)
        assoc = torch.full((mask.shape[0],), -1, dtype=torch.long, device=mask.device)
        assoc[mask] = torch.arange(
            start=max_num_nodes, end=max_num_nodes + mask.sum(), device=assoc.device
        )
        max_num_nodes = max(assoc) + 1
        data.edge_index[i] = assoc[data.edge_index[i]]

    # number of atoms per graph in the batch
    data.ptr = torch.tensor(
        [0] + [nsi.max() + 1 for nsi in data.edge_index],
        dtype=data.ptr.dtype,
        device=device,
    )
    data.natoms = data.ptr[1:] - data.ptr[:-1]

    # neighbors
    data.neighbors = [adj.shape[1] for adj in data.edge_index]

    # Concat edge_index into one
    data.edge_index = torch.cat(data.edge_index, dim=1)

    # batch
    data.batch = torch.cat(
        [
            torch.tensor(i).expand(non_sub_nodes[i].shape[0] + num_supernodes[i])
            for i in range(batch_size)
        ]
    )
    # data.batch = torch.cat([torch.tensor(i).expand(data.natoms[i])
    #     for  i in range(batch_size) ])

    # tags
    data.tags = torch.cat(
        [
            torch.cat(
                [
                    data.tags[non_sub_nodes[i]],
                    torch.tensor([0], dtype=data.tags.dtype, device=device).expand(
                        num_supernodes[i]
                    ),
                ]
            )
            for i in range(batch_size)
        ]
    )

    # SNs are last in their batch
    data.atomic_numbers = cat(
        [
            cat([data.atomic_numbers[non_sub_nodes[i]], atom_types[i]])
            for i in range(batch_size)
        ]
    )

    # position exclude the sub-surface atoms but include extra super-nodes
    # supernodes_pos = [pos for pos in supernodes_pos]
    data.pos = cat(
        [
            cat(
                [
                    data.pos[non_sub_nodes[i]],
                    cat(supernodes_pos[i : i + num_supernodes[i]]),
                ]
            )
            for i in range(batch_size)
        ]
    )

    # pos relaxed
    data.pos_relaxed = data.pos

    # the force applied on the super node is the mean of the force applied
    # to its aggregates (per batch)
    sn_force = [
        data.force[supernodes_composition[i][j]].mean(0)[None, :]
        for i in range(batch_size)
        for j in range(num_supernodes[i])
    ]
    data.force = cat(
        [
            cat(
                [data.force[non_sub_nodes[i]], cat(sn_force[i : i + num_supernodes[i]])]
            )
            for i in range(batch_size)
        ]
    )

    # fixed atoms
    data.fixed = cat(
        [
            cat(
                [
                    data.fixed[non_sub_nodes[i]],
                    tensor([1.0], dtype=data.fixed.dtype, device=device).expand(
                        num_supernodes[i]
                    ),
                ]
            )
            for i in range(batch_size)
        ]
    )

    # distances
    data.distances = torch.sqrt(
        ((data.pos[data.edge_index[0, :]] - data.pos[data.edge_index[1, :]]) ** 2).sum(
            -1
        )
    )

    # cell offsets
    # TODO: very tricky here, no correspondance because of re-indexing.
    # do it before without supernodes
    data.cell_offsets = torch.zeros(data.distances.shape[0], 3)
    # lost correspondance because of re-indexing

    # torch.argwhere(torch.all(data.edge_index.T==torch.tensor([n,atom_idx]), dim=1)).squeeze()
    # indices = torch.argwhere(torch.all(torch.isin(data.edge_index.T, concat_reindexed_clean_edge_index.T),dim=1))
    # stored_cell_offsets = data.cell_offsets[indices.squeeze()]
    # torch.all(torch.isin(concat_reindexed_clean_edge_index.T, data.edge_index.T),dim=1)

    # TODO: tests
    # create tests for shape of each parameter
    # for number of Os or some properties.
