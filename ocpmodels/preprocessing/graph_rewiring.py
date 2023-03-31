""" Rewire each 3D molecular graph
"""

from copy import deepcopy

import torch
from torch import cat, isin, tensor, where
from torch_geometric.utils import coalesce, remove_self_loops, sort_edge_index


def remove_tag0_nodes(data):
    """Delete sub-surface (tag == 0) nodes and rewire accordingly the graph

    Args:
        data (torch_geometric.Data): the data batch to re-wire

    Returns:
        torch_geometric.Data: the data rewired data batch
    """
    device = data.edge_index.device

    # non sub-surface atoms
    non_sub = torch.where(data.tags != 0)[0]
    src_is_not_sub = torch.isin(data.edge_index[0], non_sub)
    target_is_not_sub = torch.isin(data.edge_index[1], non_sub)
    neither_is_sub = src_is_not_sub * target_is_not_sub

    # per-atom tensors
    data.pos = data.pos[non_sub, :]
    data.atomic_numbers = data.atomic_numbers[non_sub]
    data.batch = data.batch[non_sub]
    if hasattr(data, "force"):
        data.force = data.force[non_sub, :]
    if hasattr(data, "fixed"):
        data.fixed = data.fixed[non_sub]
    data.tags = data.tags[non_sub]
    if hasattr(data, "pos_relaxed"):
        data.pos_relaxed = data.pos_relaxed[non_sub, :]

    # per-edge tensors
    data.edge_index = data.edge_index[:, neither_is_sub]
    data.cell_offsets = data.cell_offsets[neither_is_sub, :]
    data.distances = data.distances[neither_is_sub]
    # re-index adj matrix, given some nodes were deleted
    num_nodes = data.natoms.sum().item()
    mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    mask[non_sub] = 1
    assoc = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    assoc[mask] = torch.arange(mask.sum(), device=device)
    data.edge_index = assoc[data.edge_index]

    # per-graph tensors
    batch_size = max(data.batch).item() + 1
    data.natoms = torch.tensor(
        [(data.batch == i).sum() for i in range(batch_size)],
        dtype=data.natoms.dtype,
        device=device,
    )
    data.ptr = torch.tensor(
        [0] + [data.natoms[:i].sum() for i in range(1, batch_size + 1)],
        dtype=data.ptr.dtype,
        device=device,
    )
    _, data.neighbors = torch.unique(
        data.batch[data.edge_index[0, :]], return_counts=True
    )

    return data


def one_supernode_per_graph(data, cutoff=6.0, verbose=False):
    """Generate a single supernode representing all tag0 atoms

    Args:
        data (data.Data): single batch of graphs
    """
    batch_size = max(data.batch).item() + 1
    device = data.edge_index.device
    original_ptr = deepcopy(data.ptr)

    # ids of sub-surface nodes, per batch
    sub_nodes = [
        where((data.tags == 0) * (data.batch == i))[0] for i in range(batch_size)
    ]

    # idem for non-sub-surface nodes
    non_sub_nodes = [
        where((data.tags != 0) * (data.batch == i))[0] for i in range(batch_size)
    ]

    # super node index per batch: they are last in their batch
    # (after removal of tag0 nodes)
    new_sn_ids = [
        sum([len(nsn) for nsn in non_sub_nodes[: i + 1]]) + i for i in range(batch_size)
    ]
    # define new number of atoms per batch
    data.ptr = tensor(
        [0] + [nsi + 1 for nsi in new_sn_ids], dtype=data.ptr.dtype, device=device
    )
    data.natoms = data.ptr[1:] - data.ptr[:-1]
    # Store number of nodes each supernode contains
    data.subnodes = torch.tensor(
        [len(sub) for sub in sub_nodes], dtype=torch.long, device=device
    )

    # super node position for a batch is the mean of its aggregates
    # sn_pos = [data.pos[sub_nodes[i]].mean(0) for i in range(batch_size)]
    sn_pos = [
        torch.cat(
            [
                data.pos[sub_nodes[i], :2].mean(0),
                data.pos[sub_nodes[i], 2].max().unsqueeze(0),
            ],
            dim=0,
        )
        for i in range(batch_size)
    ]
    # the super node force is the mean of the force applied to its aggregates
    if hasattr(data, "force"):
        sn_force = [data.force[sub_nodes[i]].mean(0) for i in range(batch_size)]
        data.force = cat(
            [
                cat([data.force[non_sub_nodes[i]], sn_force[i][None, :]])
                for i in range(batch_size)
            ]
        )

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
    # relaxed position for supernode is the same as initial position
    if hasattr(data, "pos_relaxed"):
        data.pos_relaxed = cat(
            [
                cat([data.pos_relaxed[non_sub_nodes[i]], sn_pos[i][None, :]])
                for i in range(batch_size)
            ]
        )

    # idem, sn position is fixed
    if hasattr(data, "fixed"):
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

    # Edge-index and cell_offsets
    batch_idx_adj = data.batch[data.edge_index][0]
    ei_sn = data.edge_index.clone()
    new_cell_offsets = data.cell_offsets.clone()
    # number of nodes in this batch: all existing + batch_size supernodes
    num_nodes = original_ptr[-1].item()
    # Re-index
    mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    mask[cat(non_sub_nodes)] = 1  # mask is 0 for sub-surface atoms
    assoc = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    assoc[mask] = cat(
        [
            torch.arange(data.ptr[e], data.ptr[e + 1] - 1, device=device)
            for e in range(batch_size)
        ]
    )
    # re-index only edges for which not both nodes are sub-surface atoms
    ei_sn = assoc[ei_sn]

    # Adapt cell_offsets: add [0,0,0] for supernode related edges
    is_minus_one = isin(ei_sn, torch.tensor(-1, device=device))
    new_cell_offsets[is_minus_one.any(dim=0)] = torch.tensor([0, 0, 0], device=device)
    # Replace index -1 by supernode index
    ei_sn = where(
        is_minus_one,
        torch.tensor(new_sn_ids, device=device)[batch_idx_adj],
        ei_sn,
    )
    # Remove self loops
    ei_sn, new_cell_offsets = remove_self_loops(ei_sn, new_cell_offsets)

    # Remove tag0 related duplicates
    # First, store tag 1/2 adjacency
    new_non_sub_nodes = where(data.tags != 0)[0]
    tag12_ei = ei_sn[:, torch.isin(ei_sn, new_non_sub_nodes).all(dim=0)]
    tag12_cell_offsets_ei = new_cell_offsets[
        torch.isin(ei_sn, new_non_sub_nodes).all(dim=0), :
    ]
    # Remove duplicate in supernode adjacency
    indxes = torch.isin(ei_sn, torch.tensor(new_sn_ids).to(device=ei_sn.device)).any(
        dim=0
    )
    ei_sn, new_cell_offsets = coalesce(
        ei_sn[:, indxes], edge_attr=new_cell_offsets[indxes, :], reduce="min"
    )
    # Merge back both
    ei_sn = torch.cat([tag12_ei, ei_sn], dim=1)
    new_cell_offsets = torch.cat([tag12_cell_offsets_ei, new_cell_offsets], dim=0)
    ei_sn, new_cell_offsets = sort_edge_index(ei_sn, edge_attr=new_cell_offsets)

    # Remove duplicate entries
    # ei_sn, new_cell_offsets = coalesce(
    #   ei_sn, edge_attr=new_cell_offsets, reduce="min",
    # )

    # ensure correct type
    data.edge_index = ei_sn.to(dtype=data.edge_index.dtype)
    data.cell_offsets = new_cell_offsets.to(dtype=data.cell_offsets.dtype)

    # distances
    data.distances = torch.sqrt(
        ((data.pos[data.edge_index[0, :]] - data.pos[data.edge_index[1, :]]) ** 2).sum(
            -1
        )
    ).to(dtype=data.distances.dtype)

    # batch
    data.batch = torch.zeros(data.ptr[-1], dtype=data.batch.dtype, device=device)
    for i, p in enumerate(data.ptr[:-1]):
        data.batch[
            torch.arange(p, data.ptr[i + 1], dtype=torch.long, device=device)
        ] = tensor(i, dtype=data.batch.dtype, device=device)

    return adjust_cutoff_distances(data, new_sn_ids, cutoff)


def one_supernode_per_atom_type(data, cutoff=6.0):
    """Create one supernode for each sub-surface atom type
    and remove all such tag-0 atoms.

    Args:
        data (torch_geometric.Data): the data batch to re-wire

    Returns:
        torch_geometric.Data: the data rewired data batch
    """
    batch_size = max(data.batch).item() + 1
    device = data.edge_index.device
    original_ptr = deepcopy(data.ptr)

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
    total_num_supernodes = sum(num_supernodes)
    # indexes of nodes belonging to each supernode
    supernodes_composition = [
        torch.where((data.atomic_numbers == an) * (data.tags == 0) * (data.batch == i))[
            0
        ]
        for i in range(batch_size)
        for an in atom_types[i]
    ]
    # Store number of nodes each supernode regroups
    data.subnodes = torch.tensor(
        [len(sub) for sub in supernodes_composition], dtype=torch.long, device=device
    )

    # super node index per batch: they are last in their batch
    # (after removal of tag0 nodes)
    new_sn_ids = [
        [
            sum([len(nsn) for nsn in non_sub_nodes[: i + 1]]) + j
            for j in range(sum(num_supernodes[:i]), sum(num_supernodes[: i + 1]))
        ]
        for i in range(batch_size)
    ]
    # Concat version
    new_sn_ids_cat = [s for sn in new_sn_ids for s in sn]

    # supernode positions
    supernodes_pos = [
        torch.cat(
            [data.pos[sn, :2].mean(0), data.pos[sn, 2].max().unsqueeze(0)], dim=0
        )[None, :]
        for sn in supernodes_composition
    ]

    # number of atoms per graph in the batch
    data.ptr = torch.tensor(
        [0] + [max(nsi) + 1 for nsi in new_sn_ids],
        dtype=data.ptr.dtype,
        device=device,
    )
    data.natoms = data.ptr[1:] - data.ptr[:-1]

    # batch
    data.batch = torch.cat(
        [
            torch.tensor(i, device=device).expand(
                non_sub_nodes[i].shape[0] + num_supernodes[i]
            )
            for i in range(batch_size)
        ]
    )

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

    # re-index edges
    num_nodes = original_ptr[-1]  # + sum(num_supernodes)
    mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    mask[cat(non_sub_nodes)] = 1  # mask is 0 for sub-surface atoms
    assoc = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    assoc[mask] = cat(
        [
            torch.arange(
                data.ptr[e], data.ptr[e + 1] - num_supernodes[e], device=device
            )
            for e in range(batch_size)
        ]
    )
    # Set corresponding supernode index to subatoms
    for i, sn in enumerate(supernodes_composition):
        assoc[sn] = new_sn_ids_cat[i]
    # Re-index
    data.edge_index = assoc[data.edge_index]

    # Adapt cell_offsets: add [0,0,0] for supernode related edges
    data.cell_offsets[
        isin(data.edge_index, torch.tensor(new_sn_ids_cat, device=device)).any(dim=0)
    ] = torch.tensor([0, 0, 0], device=device)

    # Remove self loops and duplicates
    data.edge_index, data.cell_offsets = remove_self_loops(
        data.edge_index, data.cell_offsets
    )

    # Remove tag0 related duplicates
    # First, store tag 1/2 adjacency
    new_non_sub_nodes = where(data.tags != 0)[0]
    tag12_ei = data.edge_index[
        :, torch.isin(data.edge_index, new_non_sub_nodes).all(dim=0)
    ]
    tag12_cell_offsets_ei = data.cell_offsets[
        torch.isin(data.edge_index, new_non_sub_nodes).all(dim=0), :
    ]
    # Remove duplicate in supernode adjacency
    indxes = torch.isin(
        data.edge_index, torch.tensor(new_sn_ids_cat).to(device=data.edge_index.device)
    ).any(dim=0)
    data.edge_index, data.cell_offsets = coalesce(
        data.edge_index[:, indxes], edge_attr=data.cell_offsets[indxes, :], reduce="min"
    )
    # Merge back both
    data.edge_index = torch.cat([tag12_ei, data.edge_index], dim=1)
    data.cell_offsets = torch.cat([tag12_cell_offsets_ei, data.cell_offsets], dim=0)
    data.edge_index, data.cell_offsets = sort_edge_index(
        data.edge_index, edge_attr=data.cell_offsets
    )

    # SNs are last in their batch
    data.atomic_numbers = cat(
        [
            cat([data.atomic_numbers[non_sub_nodes[i]], atom_types[i]])
            for i in range(batch_size)
        ]
    )

    # position exclude the sub-surface atoms but include extra super-nodes
    acc_num_supernodes = [0] + [sum(num_supernodes[: i + 1]) for i in range(batch_size)]
    data.pos = cat(
        [
            cat(
                [
                    data.pos[non_sub_nodes[i]],
                    cat(
                        supernodes_pos[
                            acc_num_supernodes[i] : acc_num_supernodes[i + 1]
                        ]
                    ),
                ]
            )
            for i in range(batch_size)
        ]
    )

    # pos relaxed
    if hasattr(data, "pos_relaxed"):
        data.pos_relaxed = cat(
            [
                cat(
                    [
                        data.pos_relaxed[non_sub_nodes[i]],
                        cat(
                            supernodes_pos[
                                acc_num_supernodes[i] : acc_num_supernodes[i + 1]
                            ]
                        ),
                    ]
                )
                for i in range(batch_size)
            ]
        )

    # the force applied on the super node is the mean of the force applied
    # to its aggregates (per batch)
    if hasattr(data, "force"):
        sn_force = [
            data.force[supernodes_composition[i]].mean(0)[None, :]
            for i in range(total_num_supernodes)
        ]
        data.force = cat(
            [
                cat(
                    [
                        data.force[non_sub_nodes[i]],
                        cat(
                            sn_force[acc_num_supernodes[i] : acc_num_supernodes[i + 1]]
                        ),
                    ]
                )
                for i in range(batch_size)
            ]
        )

    # fixed atoms
    if hasattr(data, "fixed"):
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
    # TODO: compute with cell_offsets
    data.distances = torch.sqrt(
        ((data.pos[data.edge_index[0, :]] - data.pos[data.edge_index[1, :]]) ** 2).sum(
            -1
        )
    )

    return adjust_cutoff_distances(data, new_sn_ids_cat, cutoff)


def one_supernode_per_atom_type_dist(data, cutoff=6.0):
    """Create one supernode for each sub-surface atom type
    and remove all such tag-0 atoms.
    Distance to supernode is defined as min. dist of subnodes
    instead of dist. to new positions

    Args:
        data (torch_geometric.Data): the data batch to re-wire

    Returns:
        torch_geometric.Data: the data rewired data batch
    """
    batch_size = max(data.batch).item() + 1
    device = data.edge_index.device
    original_ptr = deepcopy(data.ptr)

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
    total_num_supernodes = sum(num_supernodes)
    # indexes of nodes belonging to each supernode
    supernodes_composition = [
        torch.where((data.atomic_numbers == an) * (data.tags == 0) * (data.batch == i))[
            0
        ]
        for i in range(batch_size)
        for an in atom_types[i]
    ]
    # Store number of nodes each supernode regroups
    data.subnodes = torch.tensor(
        [len(sub) for sub in supernodes_composition], dtype=torch.long, device=device
    )

    # super node index per batch: they are last in their batch
    # (after removal of tag0 nodes)
    new_sn_ids = [
        [
            sum([len(nsn) for nsn in non_sub_nodes[: i + 1]]) + j
            for j in range(sum(num_supernodes[:i]), sum(num_supernodes[: i + 1]))
        ]
        for i in range(batch_size)
    ]
    # Concat version
    new_sn_ids_cat = [s for sn in new_sn_ids for s in sn]

    # supernode positions
    # supernodes_pos = [data.pos[sn, :].mean(0)[None, :]
    #   for sn in supernodes_composition]
    supernodes_pos = [
        torch.cat(
            [data.pos[sn, :2].mean(0), data.pos[sn, 2].max().unsqueeze(0)], dim=0
        )[None, :]
        for sn in supernodes_composition
    ]

    # number of atoms per graph in the batch
    data.ptr = torch.tensor(
        [0] + [max(nsi) + 1 for nsi in new_sn_ids],
        dtype=data.ptr.dtype,
        device=device,
    )
    data.natoms = data.ptr[1:] - data.ptr[:-1]

    # batch
    data.batch = torch.cat(
        [
            torch.tensor(i, device=device).expand(
                non_sub_nodes[i].shape[0] + num_supernodes[i]
            )
            for i in range(batch_size)
        ]
    )

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

    # re-index edges
    num_nodes = original_ptr[-1]  # + sum(num_supernodes)
    mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    mask[cat(non_sub_nodes)] = 1  # mask is 0 for sub-surface atoms
    assoc = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    assoc[mask] = cat(
        [
            torch.arange(
                data.ptr[e], data.ptr[e + 1] - num_supernodes[e], device=device
            )
            for e in range(batch_size)
        ]
    )
    # Set corresponding supernode index to subatoms
    for i, sn in enumerate(supernodes_composition):
        assoc[sn] = new_sn_ids_cat[i]
    # Re-index
    data.edge_index = assoc[data.edge_index]

    # Adapt cell_offsets: add [0,0,0] for supernode related edges
    data.cell_offsets[
        isin(data.edge_index, torch.tensor(new_sn_ids_cat, device=device)).any(dim=0)
    ] = torch.tensor([0, 0, 0], device=device)

    # Define dist(sn-node) as min(dist(subnode_i, node))
    data.edge_index, offsets_and_distances = remove_self_loops(
        data.edge_index, cat([data.distances.unsqueeze(1), data.cell_offsets], dim=1)
    )
    # De-cat offsets and distances
    data.cell_offsets = offsets_and_distances[:, 1:]
    data.distances = offsets_and_distances[:, 0]
    #
    _, data.distances = coalesce(
        data.edge_index, edge_attr=data.distances, reduce="min"
    )

    # Remove tag0 related duplicates
    # First, store tag 1/2 adjacency
    new_non_sub_nodes = where(data.tags != 0)[0]
    tag12_ei = data.edge_index[
        :, torch.isin(data.edge_index, new_non_sub_nodes).all(dim=0)
    ]
    tag12_cell_offsets_ei = data.cell_offsets[
        torch.isin(data.edge_index, new_non_sub_nodes).all(dim=0), :
    ]
    # Remove duplicate in supernode adjacency
    indxes = torch.isin(
        data.edge_index, torch.tensor(new_sn_ids_cat).to(device=data.edge_index.device)
    ).any(dim=0)
    data.edge_index, data.cell_offsets = coalesce(
        data.edge_index[:, indxes], edge_attr=data.cell_offsets[indxes, :], reduce="min"
    )
    # Merge back both
    data.edge_index = torch.cat([tag12_ei, data.edge_index], dim=1)
    data.cell_offsets = torch.cat([tag12_cell_offsets_ei, data.cell_offsets], dim=0)
    data.edge_index, data.cell_offsets = sort_edge_index(
        data.edge_index, edge_attr=data.cell_offsets
    )

    # data.edge_index, data.cell_offsets = coalesce(
    #     data.edge_index, edge_attr=data.cell_offsets, reduce="min"
    # )

    # SNs are last in their batch
    data.atomic_numbers = cat(
        [
            cat([data.atomic_numbers[non_sub_nodes[i]], atom_types[i]])
            for i in range(batch_size)
        ]
    )

    # position exclude the sub-surface atoms but include extra super-nodes
    acc_num_supernodes = [0] + [sum(num_supernodes[: i + 1]) for i in range(batch_size)]
    data.pos = cat(
        [
            cat(
                [
                    data.pos[non_sub_nodes[i]],
                    cat(
                        supernodes_pos[
                            acc_num_supernodes[i] : acc_num_supernodes[i + 1]
                        ]
                    ),
                ]
            )
            for i in range(batch_size)
        ]
    )

    # pos relaxed
    if hasattr(data, "pos_relaxed"):
        data.pos_relaxed = cat(
            [
                cat(
                    [
                        data.pos_relaxed[non_sub_nodes[i]],
                        cat(
                            supernodes_pos[
                                acc_num_supernodes[i] : acc_num_supernodes[i + 1]
                            ]
                        ),
                    ]
                )
                for i in range(batch_size)
            ]
        )

    # the force applied on the super node is the mean of the force applied
    # to its aggregates (per batch)
    if hasattr(data, "force"):
        sn_force = [
            data.force[supernodes_composition[i]].mean(0)[None, :]
            for i in range(total_num_supernodes)
        ]
        data.force = cat(
            [
                cat(
                    [
                        data.force[non_sub_nodes[i]],
                        cat(
                            sn_force[acc_num_supernodes[i] : acc_num_supernodes[i + 1]]
                        ),
                    ]
                )
                for i in range(batch_size)
            ]
        )

    # fixed atoms
    if hasattr(data, "fixed"):
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

    return adjust_cutoff_distances(data, new_sn_ids_cat, cutoff)


def adjust_cutoff_distances(data, sn_indxes, cutoff=6.0):
    # remove long edges (> cutoff), for sn related edges only
    sn_indxes = torch.isin(
        data.edge_index, torch.tensor(sn_indxes).to(device=data.edge_index.device)
    ).any(dim=0)
    cutoff_mask = torch.logical_not((data.distances > cutoff) * sn_indxes)
    data.edge_index = data.edge_index[:, cutoff_mask]
    data.cell_offsets = data.cell_offsets[cutoff_mask, :]
    data.distances = data.distances[cutoff_mask]
    _, data.neighbors = torch.unique(
        data.batch[data.edge_index[0, :]], return_counts=True
    )
    return data
