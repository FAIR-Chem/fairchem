""" Rewire each 3D molecular graph
"""

import torch
from torch import cat, isin, tensor, where
from torch_geometric.utils import remove_self_loops, sort_edge_index


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


def one_supernode_per_atom_type_draft(data):
    """Create one supernode for each sub-surface atom type
    and remove all such tag-0 atoms.

    Args:
        data (torch_geometric.Data): the data batch to re-wire

    Returns:
        torch_geometric.Data: the data rewired data batch
    """
    b = data

    batch_size = max(b.batch).item() + 1
    device = b.edge_index.device

    # ids of sub-surface nodes, per batch
    sub_nodes = [
        torch.where((b.tags == 0) * (b.batch == i))[0] for i in range(batch_size)
    ]
    # idem for non-sub-surface nodes
    non_sub_nodes = [
        torch.where((b.tags != 0) * (b.batch == i))[0] for i in range(batch_size)
    ]
    # atom types per supernode
    atom_types = [
        torch.unique(b.atomic_numbers[(b.tags == 0) * (b.batch == i)])
        for i in range(batch_size)
    ]
    # number of supernodes per batch
    num_supernodes = [atom_types[i].shape[0] for i in range(batch_size)]
    total_num_supernodes = sum(num_supernodes)
    # indexes of nodes belonging to each supernode
    supernodes_composition = [
        [
            torch.where((b.atomic_numbers == an) * (b.tags == 0) * (b.batch == i))[0]
            for an in atom_types[i]
        ]
        for i in range(batch_size)
    ]
    # supernode indexes
    sn_idxes = [
        b.ptr[1:][i] + sn
        for i in range(len(num_supernodes))
        for sn in range(num_supernodes[i])
    ]

    ### Define attributes of supernode using supernodes_composition
    # Tags,

    ### Compute supernode positions
    supernodes_pos = [
        b.pos[sn, :][0] for sublist in supernodes_composition for sn in sublist
    ]

    #### Compute supernode edge-index
    # TODO: replace in edge-index supernode's subnodes by supernode idx
    # for tag0-tag1/2 edges only. Same for distances.
    ei_batch_ids = [
        (b.ptr[i] <= b.edge_index[0]) * (b.edge_index[0] < b.ptr[i + 1])
        for i in range(batch_size)
    ]
    # list of graph level adj.
    ei_batch = [b.edge_index[:, ei_batch_ids[i]] for i in range(batch_size)]
    # boolean src node is not sub per batch
    src_is_sub = [
        torch.isin(b.edge_index[0][ei_batch_ids[i]], ns)
        for i, ns in enumerate(sub_nodes)
    ]
    # boolean target node is not sub per batch
    target_is_sub = [
        torch.isin(b.edge_index[1][ei_batch_ids[i]], ns)
        for i, ns in enumerate(sub_nodes)
    ]
    # Select edges between supernode component and tag1/2 atoms
    # source_is_sub_target_not = [torch.logical_and(s, ~t) for s, t in zip(src_is_sub, target_is_sub)]
    # target_is_sub_source_not = [torch.logical_and(~s, t) for s, t in zip(src_is_sub, target_is_sub)]
    # one_is_sub_other_not = [torch.logical_or(~s, t) for s, t in zip(source_is_sub_target_not, target_is_sub_source_not)]
    only_one_is_sub = [
        torch.logical_or((s & ~t), (~s & t)) for s, t in zip(src_is_sub, target_is_sub)
    ]

    #
    supernode_ei_batch = []
    for i in range(batch_size):
        for j in range(sum(num_supernodes[:i]), sum(num_supernodes[: i + 1])):
            supernode_ei_batch.append(
                ei_batch[i][
                    :,
                    torch.logical_or(
                        torch.isin(ei_batch[i][0], supernodes_composition[j]),
                        torch.isin(ei_batch[i][1], supernodes_composition[j]),
                    ),
                ]
            )

    # Compute supernode distances
    supernode_distances = b.distances[torch.cat(only_one_is_sub)]

    #############

    # Define new edge_index matrix per batch
    for i in range(batch_size):
        for j, sc in enumerate(supernodes_composition[i]):
            ei_batch[i] = torch.where(
                torch.isin(ei_batch[i], sc), sn_idxes[j], ei_batch[i]
            )
    # # Define new edge_index matrix per batch
    # new_edge_index = [ torch.where(torch.isin(ei_batch[i], sc), sn_idxes[j], ei_batch[i])
    #     for i in range(batch_size) for j, sc in enumerate(supernodes_composition[i])
    # ]
    # Remove self loops and duplicates
    clean_new_edge_index = [
        torch.unique(remove_self_loops(adj)[0], dim=1) for adj in ei_batch
    ]

    # re-index adj matrix, given some nodes were deleted and some added
    # Do for each batch otherwise problem of indices with supernodes
    # Redef their index to be above batch max idx.
    num_nodes = data.natoms.sum().item()  # udpate data.natoms with supernodes.
    mask = torch.zeros(num_nodes, dtype=torch.bool, device=data.edge_index.device)
    mask[torch.cat(non_sub_nodes)] = 1
    assoc = torch.full((num_nodes,), -1, dtype=torch.long, device=mask.device)
    assoc[mask] = torch.arange(mask.sum(), device=assoc.device)
    reindex_clean_new_edge_index = assoc[clean_new_edge_index]

    # Concat into one
    concat_clean_edge_index = torch.cat(clean_new_edge_index, dim=1)

    # Distances
    distance = torch.sqrt(
        (
            (
                data.pos[concat_clean_edge_index[0, :]]
                - data.pos[concat_clean_edge_index[1, :]]
            )
            ** 2
        ).sum(-1)
    )

    # test
    assert ((data.tags == 0) & (data.batch == 0)).nonzero().shape[0] == torch.cat(
        supernodes_composition[0]
    ).shape[0]
    # test
    torch.isin(ei_batch[i], sc).sum()  # vs
    torch.isin(ei_batch[i], sn_idxes[j]).sum()  # vs
    torch.isin(clean_new_edge_index[i], sn_idxes[j]).sum()
    # test
    torch.where(torch.all(ei_batch[0].T == torch.tensor([40, 41]), dim=1))[0]  # vs
    torch.where(torch.all(clean_new_edge_index[0].T == torch.tensor([40, 41]), dim=1))[
        0
    ]

    return data


def one_supernode_per_atom_type(data):
    """Create one supernode for each sub-surface atom type
    and remove all such tag-0 atoms.

    Args:
        data (torch_geometric.Data): the data batch to re-wire

    Returns:
        torch_geometric.Data: the data rewired data batch
    """
    b = data

    batch_size = max(b.batch).item() + 1
    device = b.edge_index.device

    # ids of sub-surface nodes, per batch
    sub_nodes = [
        torch.where((b.tags == 0) * (b.batch == i))[0] for i in range(batch_size)
    ]
    # idem for non-sub-surface nodes
    non_sub_nodes = [
        torch.where((b.tags != 0) * (b.batch == i))[0] for i in range(batch_size)
    ]
    # atom types per supernode
    atom_types = [
        torch.unique(b.atomic_numbers[(b.tags == 0) * (b.batch == i)])
        for i in range(batch_size)
    ]
    # number of supernodes per batch
    num_supernodes = [atom_types[i].shape[0] for i in range(batch_size)]
    total_num_supernodes = sum(num_supernodes)
    # indexes of nodes belonging to each supernode
    supernodes_composition = [
        [
            torch.where((b.atomic_numbers == an) * (b.tags == 0) * (b.batch == i))[0]
            for an in atom_types[i]
        ]
        for i in range(batch_size)
    ]
    # supernode indexes
    sn_idxes = [
        [b.ptr[1:][i] + sn for sn in range(num_supernodes[i])]
        for i in range(len(num_supernodes))
    ]

    # supernode positions
    supernodes_pos = [
        b.pos[sn, :].mean() for sublist in supernodes_composition for sn in sublist
    ]

    ### Compute supernode edge-index
    ei_batch_ids = [
        (b.ptr[i] <= b.edge_index[0]) * (b.edge_index[0] < b.ptr[i + 1])
        for i in range(batch_size)
    ]
    # list of graph level adj.
    ei_batch = [b.edge_index[:, ei_batch_ids[i]] for i in range(batch_size)]

    # Define new edge_index matrix per batch
    for i in range(batch_size):
        for j, sc in enumerate(supernodes_composition[i]):
            ei_batch[i] = torch.where(
                torch.isin(ei_batch[i], sc), sn_idxes[i][j], ei_batch[i]
            )

    # Remove self loops and duplicates
    clean_new_edge_index = [
        torch.unique(remove_self_loops(adj)[0], dim=1) for adj in ei_batch
    ]
    # TODO: coalescence

    # re-index batch adj matrix one by one
    max_num_nodes = 0
    reindexed_clean_edge_index = clean_new_edge_index.copy()
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
        reindexed_clean_edge_index[i] = assoc[clean_new_edge_index[i]]

    # number of atoms per graph in the batch
    data.new_ptr = torch.tensor(
        [0] + [nsi.max() + 1 for nsi in reindexed_clean_edge_index],
        dtype=b.ptr.dtype,
        device=device,
    )
    data.new_natoms = data.new_ptr[1:] - data.new_ptr[:-1]

    # neighbors
    data.new_neighbors = [adj.shape[1] for adj in reindexed_clean_edge_index]

    # Concat edge_index into one
    concat_reindexed_clean_edge_index = torch.cat(reindexed_clean_edge_index, dim=1)

    # batch
    data.new_batch = torch.cat(
        [
            torch.tensor(i).expand(non_sub_nodes[i].shape[0] + num_supernodes[i])
            for i in range(batch_size)
        ]
    )
    # data.new_batch = torch.cat([torch.tensor(i).expand(data.new_natoms[i])
    #     for  i in range(batch_size) ])

    # tags
    data.new_tags = torch.cat(
        [
            torch.cat(
                [
                    b.tags[non_sub_nodes[i]],
                    torch.tensor([0], dtype=b.tags.dtype, device=device).expand(
                        num_supernodes[i]
                    ),
                ]
            )
            for i in range(batch_size)
        ]
    )

    # position exclude the sub-surface atoms but include an extra super-node
    # data.pos = cat(
    #     [
    #         cat([b.pos[non_sub_nodes[i]], supernodes_pos[i][None, :]])
    #         for i in range(batch_size) for j in range(num_supernodes)
    #     ]
    # )

    # pos
    data.pos_relaxed = cat(
        [
            cat([b.pos_relaxed[non_sub_nodes[i]], supernodes_pos[i][None, :]])
            for i in range(batch_size)
        ]
    )

    # number of neigbors
    #  _, data.new_neighbors = torch.unique(
    #         data.new_batch[data.edge_index[0, :]], return_counts=True
    #     )

    ### Define attributes of supernode using supernodes_composition
    # Tags, batch, etc.
