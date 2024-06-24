import torch

from torch_cluster import radius_graph
from torch_scatter import scatter_min

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
        torch.cat([
            torch.arange(0, rep + 1, device=device, dtype=torch.float),
            torch.arange(-rep, 0, device=device, dtype=torch.float)
        ]) for rep in max_rep
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
    ).permute(2, 0, 1)

    # Expand the positions and indices for the 27 cells
    pos1 = pos1.expand(num_cells, -1, -1)
    pos2 = pos2.expand(num_cells, -1, -1)
    index1 = index1.expand(num_cells, -1)
    index2 = index2.expand(num_cells, -1)
    index3 = index2 + data.natoms.sum() * torch.arange(num_cells, device=device)[:, None]
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the squared distance between atoms
    dist = torch.linalg.norm(pos1 - pos2, dim=-1)

    # Remove pairs that are too far apart
    mask_within_radius = torch.le(dist, radius)

    index1 = torch.masked_select(index1, mask_within_radius)
    index2 = torch.masked_select(index2, mask_within_radius)
    index3 = torch.masked_select(index3, mask_within_radius)
    src_pos = torch.masked_select(pos2, mask_within_radius[..., None]).view(-1, 3)
    dist = torch.masked_select(dist, mask_within_radius)

    # sort according to row index
    indx = torch.argsort(index1, stable=True)
    index1, index2, index3, src_pos, dist = (
        index1[indx], index2[indx], index3[indx], src_pos[indx], dist[indx]
    )

    # sort the indices
    unique_index3, index3 = torch.unique(index3, sorted=True, return_inverse=True)
    indx = torch.arange(index3.size(0), dtype=index3.dtype, device=index3.device)
    index3_fliped, indx = index3.flip(0), indx.flip(0)
    indx = index3_fliped.new_empty(unique_index3.size(0)).scatter_(0, index3_fliped, indx)
    
    # get position and indicies
    src_pos = src_pos[indx]

    # construct edges
    edge_index = torch.stack([index1, index2])
    src_index = torch.stack([index1, index3])

    # count number of unique elements in edge index
    unique_edges, inverse, multiplicity = torch.unique(
        edge_index, dim=1, sorted=True, return_inverse=True, return_counts=True
    )
    multiplicity = multiplicity[inverse]
    num_edges_to_remove = unique_edges.size(1) % 4
    num_dup_edges_to_remove = (src_index.size(1) - num_edges_to_remove) % 4


    # make it multiple of 4 by removing large radius ones
    mask = torch.ones(dist.size(0), device=device, dtype=torch.bool)
    if num_edges_to_remove > 0:
        _, indicies = torch.topk(dist[multiplicity==1], num_edges_to_remove, largest=True)
        mask[torch.arange(dist.size(0), device=device)[multiplicity==1][indicies]] = False
        
    # to avoid cutting the last edge
    _, argmin = scatter_min(dist, inverse, dim=0)
    multiplicity[argmin] = 1 

    if num_dup_edges_to_remove > 0:
        _, indicies = torch.topk(dist[multiplicity>1], num_dup_edges_to_remove, largest=True)
        mask[torch.arange(dist.size(0), device=device)[multiplicity>1][indicies]] = False
    edge_index, src_index, dist = edge_index[:, mask], src_index[:, mask], dist[mask]

    # filter edges
    edge_index, edge_to_src = torch.unique(edge_index, dim=1, sorted=True, return_inverse=True)
    assert edge_index.size(1) % 4 == 0, f"the edge index must be multiple of 4, found {edge_index.size(1)}"
    assert src_index.size(1) % 4 == 0, f"the src index must be multiple of 4, found {src_index.size(1)}"

    return edge_index, src_index, edge_to_src, dist, src_pos

def build_radius_graph(
    data,
    radius,
    use_pbc=False,
):
    device=data.pos.device
    if use_pbc:
        edge_index, src_index, edge_to_src, dist, src_pos = radius_graph_pbc(data, radius)
    else:
        edge_index = radius_graph(
            data.pos, 
            radius, 
            data.batch,
            flow="target_to_source",
            max_num_neighbors=data.natoms.max(),
        )
        dist = torch.linalg.norm(data.pos[edge_index[0]] - data.pos[edge_index[1]], dim=-1)

        if dist.size(0) % 4 != 0:
            _, indicies = torch.topk(dist, dist.size(0) % 4, largest=True)
            mask = torch.ones(dist.size(0), device=device, dtype=torch.bool)
            mask.index_fill_(0, indicies, False)
            edge_index, dist = edge_index[:, mask], dist[mask]

        src_pos = data.pos
        edge_to_src = None
        src_index = edge_index
        
    return edge_index, src_index, edge_to_src, dist, src_pos