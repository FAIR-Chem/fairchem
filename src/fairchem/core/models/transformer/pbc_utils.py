import torch

from torch_cluster import radius_graph

def radius_graph_pbc(
    data,
    radius,
):
    device = data.pos.device
    batch_size = len(data.natoms)

    # position of the atoms
    atom_pos = data.pos

    # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
    num_atoms_per_image = data.natoms
    num_atoms_per_image_sqr = (num_atoms_per_image**2).long()

    # index offset between images
    index_offset = torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image

    index_offset_expand = torch.repeat_interleave(index_offset, num_atoms_per_image_sqr)
    num_atoms_per_image_expand = torch.repeat_interleave(
        num_atoms_per_image, num_atoms_per_image_sqr
    )

    # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
    # that is used to compute indices for the pairs of atoms. This is a very convoluted way to implement
    # the following (but 10x faster since it removes the for loop)
    # for batch_idx in range(batch_size):
    #    batch_count = torch.cat([batch_count, torch.arange(num_atoms_per_image_sqr[batch_idx], device=device)], dim=0)
    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = (
        torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
    )
    index_sqr_offset = torch.repeat_interleave(
        index_sqr_offset, num_atoms_per_image_sqr
    )
    atom_count_sqr = torch.arange(num_atom_pairs, device=device) - index_sqr_offset

    # Compute the indices for the pairs of atoms (using division and mod)
    # If the systems get too large this apporach could run into numerical precision issues
    index1 = (
        torch.div(atom_count_sqr, num_atoms_per_image_expand, rounding_mode="floor")
    ) + index_offset_expand
    index2 = (atom_count_sqr % num_atoms_per_image_expand) + index_offset_expand
    # Get the positions for each atom
    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)

    # Tensor of unit cells
    cells_per_dim = [
        torch.arange(-1, 2, device=device, dtype=torch.float) for _ in range(3)
    ]
    unit_cell = torch.cartesian_prod(*cells_per_dim)
    num_cells = len(unit_cell)
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(batch_size, -1, -1)

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(data.cell, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms_per_image_sqr, dim=0
    )

    # Expand the positions and indices for the 27 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    index1 = index1.view(-1, 1).repeat(1, num_cells)
    index2 = index2.view(-1, 1).repeat(1, num_cells)
    src_index = index2 + data.natoms.sum() * torch.arange(num_cells, device=device)[None]
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the squared distance between atoms
    dist = torch.linalg.norm(pos1 - pos2, dim=1)

    # select the one with minimum distance
    argmin = torch.argmin(dist, dim=1, keepdim=True)
    dist = dist.gather(1, argmin).squeeze(1)
    index1 = index1.gather(1, argmin).squeeze(1)
    index2 = index2.gather(1, argmin).squeeze(1)
    src_index = src_index.gather(1, argmin).squeeze(1)
    pos2 = pos2.gather(2, argmin[..., None].expand(-1, 3, -1)).squeeze(2)

    # Remove pairs that are too far apart
    mask_within_radius = torch.le(dist, radius)

    index1 = torch.masked_select(index1, mask_within_radius)
    index2 = torch.masked_select(index2, mask_within_radius)
    src_index = torch.masked_select(src_index, mask_within_radius)
    src_pos = torch.masked_select(pos2, mask_within_radius[:, None]).view(-1, 3)
    dist = torch.masked_select(dist, mask_within_radius)

    # sort the indices
    unique_src_index, src_index = torch.unique(src_index, sorted=True, return_inverse=True)
    indx = torch.arange(src_index.size(0), dtype=src_index.dtype, device=src_index.device)
    src_index_fliped, indx = src_index.flip(0), indx.flip(0)
    indx = src_index_fliped.new_empty(unique_src_index.size(0)).scatter_(0, src_index_fliped, indx)
    
    # get position and indicies
    src_pos = src_pos[indx]
    org_to_src = index2[indx]

    return index1, index2, src_index, dist, src_pos, org_to_src

def build_radius_graph(
    data,
    radius,
    use_pbc=False,
):
    if use_pbc:
        return radius_graph_pbc(data, radius)
    else:
        edge_index = radius_graph(
            data.pos, 
            radius, 
            data.batch,
            flow="target_to_source",
            max_num_neighbors=data.natoms.max(),
        )
        dist = torch.linalg.norm(data.pos[edge_index[0]] - data.pos[edge_index[1]], dim=-1)
        return edge_index[0], edge_index[1], edge_index[1], dist, data.pos, torch.arange(data.pos.size(0), device=dist.device)