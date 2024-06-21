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

    cross_a2a3 = torch.cross(data.cell[:, 1], data.cell[:, 2], dim=-1)
    cell_vol = torch.sum(data.cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)

    inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol, p=2, dim=-1)
    rep_a1 = torch.ceil(radius * inv_min_dist_a1)

    cross_a3a1 = torch.cross(data.cell[:, 2], data.cell[:, 0], dim=-1)
    inv_min_dist_a2 = torch.norm(cross_a3a1 / cell_vol, p=2, dim=-1)
    rep_a2 = torch.ceil(radius * inv_min_dist_a2)

    cross_a1a2 = torch.cross(data.cell[:, 0], data.cell[:, 1], dim=-1)
    inv_min_dist_a3 = torch.norm(cross_a1a2 / cell_vol, p=2, dim=-1)
    rep_a3 = torch.ceil(radius * inv_min_dist_a3)

    # Take the max over all images for uniformity. This is essentially padding.
    # Note that this can significantly increase the number of computed distances
    # if the required repetitions are very different between images
    # (which they usually are). Changing this to sparse (scatter) operations
    # might be worth the effort if this function becomes a bottleneck.
    max_rep = [rep_a1.max(), rep_a2.max(), rep_a3.max()]

    # Tensor of unit cells
    cells_per_dim = [
        torch.arange(-rep, rep + 1, device=device, dtype=torch.float) for rep in max_rep
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
    index3 = index2 + data.natoms.sum() * torch.arange(num_cells, device=device)[None]
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the squared distance between atoms
    dist = torch.linalg.norm(pos1 - pos2, dim=1)

    # Remove pairs that are too far apart
    mask_within_radius = torch.le(dist, radius)

    index1 = torch.masked_select(index1, mask_within_radius)
    index2 = torch.masked_select(index2, mask_within_radius)
    index3 = torch.masked_select(index3, mask_within_radius)
    src_pos = torch.masked_select(pos2.permute(0, 2, 1), mask_within_radius[..., None]).view(-1, 3)
    dist = torch.masked_select(dist, mask_within_radius)

    # sort the indices
    unique_index3, index3 = torch.unique(index3, sorted=True, return_inverse=True)
    indx = torch.arange(index3.size(0), dtype=index3.dtype, device=index3.device)
    index3_fliped, indx = index3.flip(0), indx.flip(0)
    indx = index3_fliped.new_empty(unique_index3.size(0)).scatter_(0, index3_fliped, indx)
    
    # get position and indicies
    src_pos = src_pos[indx]
    org_to_src = index2[indx]

    return index1, index3, dist, src_pos, org_to_src

def build_radius_graph(
    data,
    radius,
    use_pbc=False,
):
    device=data.pos.device
    if use_pbc:
        row_index, col_index, dist, col_pos, to_col_index = radius_graph_pbc(data, radius)
    else:
        row_index, col_index = radius_graph(
            data.pos, 
            radius, 
            data.batch,
            flow="target_to_source",
            max_num_neighbors=data.natoms.max(),
        )
        dist = torch.linalg.norm(data.pos[row_index] - data.pos[col_index], dim=-1)
        col_pos = data.pos
        to_col_index = None

    if dist.size(0) % 4 != 0:
        _, indicies = torch.topk(dist, dist.size(0) % 4, largest=True)
        mask = torch.ones(dist.size(0), device=device, dtype=torch.bool)
        mask.index_fill_(0, indicies, False)
        row_index, col_index, dist = row_index[mask], col_index[mask], dist[mask]
        
    return row_index, col_index, dist, col_pos, to_col_index