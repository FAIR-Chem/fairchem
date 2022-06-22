""" Rewire each 3D molecular graph
"""

import torch


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
        torch.unique(b.atomic_numbers[(b.tags == 0) * (b.batch == i)]) for i in range(batch_size)
    ]
    # number of supernodes per batch
    num_supernodes = [
        atom_types[i].shape[0] for i in range(batch_size)
    ]
    total_num_supernodes = sum(num_supernodes)
    # indexes of nodes belonging to each supernode
    supernodes_composition = [
        torch.where((b.atomic_numbers == an) * (b.tags == 0) * (b.batch == i))[0] for i in range(batch_size) for an in atom_types[i]
    ]
    assert total_num_supernodes == len(supernodes_composition)

    ### Define attributes of supernode using supernodes_composition
    # Tags, 

    ### Compute supernode positions 
    supernodes_pos = [
        b.pos[sn, :][0] for sn in supernodes_composition
    ]

    #### Compute supernode edge-index
    # TODO: replace in edge-index supernode's subnodes by supernode idx
    # for tag0-tag1/2 edges only. Same for distances. 
    ei_batch_ids = [
        (b.ptr[i] <= b.edge_index[0]) * (b.edge_index[0] < b.ptr[i + 1])
        for i in range(batch_size)
    ] 
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
    only_one_is_sub = [torch.logical_or((s & ~t), (~s & t)) for s, t in zip(src_is_sub, target_is_sub)]

    supernode_ei_batch = []
    for i in range(batch_size):
        for j in range(sum(num_supernodes[:i]), sum(num_supernodes[:i+1])):
            supernode_ei_batch.append(ei_batch[i][:,
                torch.logical_or(torch.isin(ei_batch[i][0],
                    supernodes_composition[j]), torch.isin(ei_batch[i][1], 
                    supernodes_composition[j]))])

    # Compute supernode distances
    supernode_distances = b.distances[torch.cat(only_one_is_sub)]

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