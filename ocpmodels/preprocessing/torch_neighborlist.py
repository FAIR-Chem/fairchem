import ase.io
import numpy as np
import torch
from ase import Atoms
from ase.build import add_adsorbate, fcc100
from ase.neighborlist import primitive_neighbor_list


def torch_divmod(a, b, device):
    a = a.to(device)
    b = b.to(device)
    return (a // b, torch.remainder(a, b))


def torch_neighbor_list(
    quantities,
    pbc,
    cell,
    positions,
    cutoff,
    device="cuda",
    numbers=None,
    self_interaction=False,
    use_scaled_positions=False,
    max_nbins=1e6,
):
    if len(positions) == 0:
        empty_types = dict(
            i=(torch.int, (0,)),
            j=(torch.int, (0,)),
            D=(torch.float, (0, 3)),
            d=(torch.float, (0,)),
            S=(torch.int, (0, 3)),
        )
        retvals = []
        for i in quantities:
            dtype, shape = empty_types[i]
            retvals += [torch.tensor([], dtype=dtype).view(shape)]
        if len(retvals) == 1:
            return retvals[0]
        else:
            return tuple(retvals)

    # Compute reciprocal lattice vectors.
    b1_c, b2_c, b3_c = torch.pinverse(cell.view(1, 3, 3)).view(3, 3).t()

    # Compute distances of cell faces.
    l1 = b1_c.norm()
    l2 = b2_c.norm()
    l3 = b3_c.norm()
    face_dist_c = torch.tensor(
        [
            1 / l1 if l1 > 0 else 1,
            1 / l2 if l2 > 0 else 1,
            1 / l3 if l3 > 0 else 1,
        ]
    ).to(device)

    try:
        max_cutoff = 2 * cutoff.max()
    except Exception:
        max_cutoff = cutoff

    # We use a minimum bin size of 3 A
    bin_size = max(max_cutoff, 3)
    # Compute number of bins such that a sphere of radius cutoff fits into
    # eight neighboring bins.
    nbins_c = torch.max(
        (face_dist_c / bin_size).float(), torch.ones(3).to(device)
    )
    nbins = torch.prod(nbins_c)
    # Make sure we limit the amount of memory used by the explicit bins.
    while nbins > max_nbins:
        nbins_c = torch.max(nbins_c // 2, torch.ones(3))
        nbins = torch.prod(nbins_c)

    # Compute over how many bins we need to loop in the neighbor list search.
    neigh_search_x, neigh_search_y, neigh_search_z = torch.ceil(
        bin_size * nbins_c / face_dist_c
    ).int()

    # If we only have a single bin and the system is not periodic, then we
    # do not need to search neighboring bins
    neigh_search_x = 0 if nbins_c[0] == 1 and not pbc[0] else neigh_search_x
    neigh_search_y = 0 if nbins_c[1] == 1 and not pbc[1] else neigh_search_y
    neigh_search_z = 0 if nbins_c[2] == 1 and not pbc[2] else neigh_search_z

    # Sort atoms into bins.
    if use_scaled_positions:
        scaled_positions_ic = positions
        positions = torch.dot(scaled_positions_ic, cell)
    else:
        scaled_positions_ic, _ = torch.solve(positions.t(), cell.t())
        scaled_positions_ic = scaled_positions_ic.t()
    bin_index_ic = torch.floor(scaled_positions_ic * nbins_c).int()
    cell_shift_ic = torch.zeros_like(bin_index_ic)

    for c in range(3):
        if pbc[c]:
            cell_shift_ic[:, c], bin_index_ic[:, c] = torch_divmod(
                bin_index_ic[:, c], nbins_c[c], device
            )
        else:
            bin_index_ic[:, c] = torch.clamp(
                bin_index_ic[:, c], 0, nbins_c[c] - 1
            )

    # Convert Cartesian bin index to unique scalar bin index.
    bin_index_i = bin_index_ic[:, 0] + nbins_c[0] * (
        bin_index_ic[:, 1] + nbins_c[1] * bin_index_ic[:, 2]
    )

    # atom_i contains atom index in new sort order.
    atom_i = torch.argsort(bin_index_i)
    bin_index_i = bin_index_i[atom_i].int()

    # Find max number of atoms per bin
    max_natoms_per_bin = torch.bincount(bin_index_i).max()

    # Sort atoms into bins: atoms_in_bin_ba contains for each bin (identified
    # by its scalar bin index) a list of atoms inside that bin. This list is
    # homogeneous, i.e. has the same size *max_natoms_per_bin* for all bins.
    # The list is padded with -1 values.
    atoms_in_bin_ba = -torch.ones(
        [nbins.int().item(), max_natoms_per_bin.item()], dtype=int
    )
    for i in range(max_natoms_per_bin):
        # Create a mask array that identifies the first atom of each bin.
        mask = torch.cat(
            (
                torch.tensor([True]).to(device),
                bin_index_i[:-1] != bin_index_i[1:],
            )
        )
        # Assign all first atoms.
        atoms_in_bin_ba[bin_index_i[mask].item(), i] = atom_i[mask]

        # Remove atoms that we just sorted into atoms_in_bin_ba. The next
        # "first" atom will be the second and so on.
        mask = torch.logical_not(mask)
        atom_i = atom_i[mask]
        bin_index_i = bin_index_i[mask]

    # Make sure that all atoms have been sorted into bins.
    assert len(atom_i) == 0
    assert len(bin_index_i) == 0

    # Now we construct neighbor pairs by pairing up all atoms within a bin or
    # between bin and neighboring bin. atom_pairs_pn is a helper buffer that
    # contains all potential pairs of atoms between two bins, i.e. it is a list
    # of length max_natoms_per_bin**2.
    # atom_pairs_pn = np.indices((max_natoms_per_bin, max_natoms_per_bin)
    dimensions = tuple((max_natoms_per_bin, max_natoms_per_bin))
    N = len(dimensions)
    shape = (1,) * N
    atom_pairs_pn = torch.empty((N,) + dimensions)
    for i, dim in enumerate(dimensions):
        idx = torch.arange(dim).view(
            shape[:i] + (dim.item(),) + shape[i + 1 :]
        )
        atom_pairs_pn[i] = idx
    atom_pairs_pn = atom_pairs_pn.view(2, -1).long()

    # Initialized empty neighbor list buffers.
    first_at_neightuple_nn = []
    secnd_at_neightuple_nn = []
    cell_shift_vector_x_n = []
    cell_shift_vector_y_n = []
    cell_shift_vector_z_n = []

    # This is the main neighbor list search. We loop over neighboring bins and
    # then construct all possible pairs of atoms between two bins, assuming
    # that each bin contains exactly max_natoms_per_bin atoms. We then throw
    # out pairs involving pad atoms with atom index -1 below.
    binz_xyz, biny_xyz, binx_xyz = torch.meshgrid(
        torch.arange(nbins_c[2]),
        torch.arange(nbins_c[1]),
        torch.arange(nbins_c[0]),
    )

    # First atoms in pair.
    _first_at_neightuple_n = atoms_in_bin_ba[:, atom_pairs_pn[0]]
    for dz in range(-neigh_search_z, neigh_search_z + 1):
        for dy in range(-neigh_search_y, neigh_search_y + 1):
            for dx in range(-neigh_search_x, neigh_search_x + 1):
                shiftx_xyz, neighbinx_xyz = torch_divmod(
                    binx_xyz + dx, nbins_c[0], device
                )
                shifty_xyz, neighbiny_xyz = torch_divmod(
                    biny_xyz + dy, nbins_c[1], device
                )
                shiftz_xyz, neighbinz_xyz = torch_divmod(
                    binz_xyz + dz, nbins_c[2], device
                )
                neighbin_b = (
                    (
                        neighbinx_xyz
                        + nbins_c[0]
                        * (neighbiny_xyz + nbins_c[1] * neighbinz_xyz)
                    )
                    .view(-1)
                    .long()
                )

                # Second atom in pair.
                _secnd_at_neightuple_n = atoms_in_bin_ba[neighbin_b][
                    :, atom_pairs_pn[1]
                ]

                # Shift vectors.
                _cell_shift_vector_x_n = torch.repeat_interleave(
                    shiftx_xyz.reshape(-1, 1), max_natoms_per_bin ** 2, dim=1
                )
                _cell_shift_vector_y_n = torch.repeat_interleave(
                    shifty_xyz.reshape(-1, 1), max_natoms_per_bin ** 2, dim=1
                )
                _cell_shift_vector_z_n = torch.repeat_interleave(
                    shiftz_xyz.reshape(-1, 1), max_natoms_per_bin ** 2, dim=1
                )

                # We have created too many pairs because we assumed each bin
                # has exactly max_natoms_per_bin atoms. Remove all surperfluous
                # pairs. Those are pairs that involve an atom with index -1.
                mask = torch.logical_and(
                    _first_at_neightuple_n != -1, _secnd_at_neightuple_n != -1
                )
                if mask.sum() > 0:
                    first_at_neightuple_nn += [_first_at_neightuple_n[mask]]
                    secnd_at_neightuple_nn += [_secnd_at_neightuple_n[mask]]
                    cell_shift_vector_x_n += [_cell_shift_vector_x_n[mask]]
                    cell_shift_vector_y_n += [_cell_shift_vector_y_n[mask]]
                    cell_shift_vector_z_n += [_cell_shift_vector_z_n[mask]]

    first_at_neightuple_n = torch.cat(first_at_neightuple_nn).to(device)
    secnd_at_neightuple_n = torch.cat(secnd_at_neightuple_nn).to(device)
    cell_shift_vector_n = torch.stack(
        [
            torch.cat(cell_shift_vector_x_n),
            torch.cat(cell_shift_vector_y_n),
            torch.cat(cell_shift_vector_z_n),
        ],
        dim=1,
    ).to(device)

    # Add global cell shift to shift vectors
    cell_shift_vector_n += (
        cell_shift_ic[first_at_neightuple_n]
        - cell_shift_ic[secnd_at_neightuple_n]
    )

    # Remove all self-pairs that do not cross the cell boundary.
    if not self_interaction:
        m = torch.logical_not(
            torch.logical_and(
                first_at_neightuple_n == secnd_at_neightuple_n,
                (cell_shift_vector_n == 0).all(axis=1),
            )
        )
        first_at_neightuple_n = first_at_neightuple_n[m]
        secnd_at_neightuple_n = secnd_at_neightuple_n[m]
        cell_shift_vector_n = cell_shift_vector_n[m]

    # For nonperiodic directions, remove any bonds that cross the domain
    # boundary.
    for c in range(3):
        if not pbc[c]:
            m = cell_shift_vector_n[:, c] == 0
            first_at_neightuple_n = first_at_neightuple_n[m]
            secnd_at_neightuple_n = secnd_at_neightuple_n[m]
            cell_shift_vector_n = cell_shift_vector_n[m]

    # Sort neighbor list.
    i = torch.argsort(first_at_neightuple_n)
    first_at_neightuple_n = first_at_neightuple_n[i]
    secnd_at_neightuple_n = secnd_at_neightuple_n[i]
    cell_shift_vector_n = cell_shift_vector_n[i]

    # Compute distance vectors.
    distance_vector_nc = (
        positions[secnd_at_neightuple_n]
        - positions[first_at_neightuple_n]
        + cell_shift_vector_n.mm(cell.float())
    )
    abs_distance_vector_n = torch.sqrt(
        torch.sum(distance_vector_nc * distance_vector_nc, axis=1)
    )

    # We have still created too many pairs. Only keep those with distance
    # smaller than max_cutoff.
    mask = abs_distance_vector_n < max_cutoff
    first_at_neightuple_n = first_at_neightuple_n[mask]
    secnd_at_neightuple_n = secnd_at_neightuple_n[mask]
    cell_shift_vector_n = cell_shift_vector_n[mask]
    distance_vector_nc = distance_vector_nc[mask]
    abs_distance_vector_n = abs_distance_vector_n[mask]

    # If cutoff is a list that contains atomic radii. Atoms are neighbors
    # if their radii overlap.
    try:
        mask = (
            abs_distance_vector_n
            < cutoff[first_at_neightuple_n] + cutoff[secnd_at_neightuple_n]
        )
        first_at_neightuple_n = first_at_neightuple_n[mask]
        secnd_at_neightuple_n = secnd_at_neightuple_n[mask]
        cell_shift_vector_n = cell_shift_vector_n[mask]
        distance_vector_nc = distance_vector_nc[mask]
        abs_distance_vector_n = abs_distance_vector_n[mask]
    except Exception:
        pass

    # Assemble return tuple.
    retvals = []
    for q in quantities:
        if q == "i":
            retvals += [first_at_neightuple_n]
        elif q == "j":
            retvals += [secnd_at_neightuple_n]
        elif q == "D":
            retvals += [distance_vector_nc]
        elif q == "d":
            retvals += [abs_distance_vector_n]
        elif q == "S":
            retvals += [cell_shift_vector_n]
        else:
            raise ValueError("Unsupported quantity specified.")
    if len(retvals) == 1:
        return retvals[0]
    else:
        return tuple(retvals)


if __name__ == "__main__":
    slab = fcc100("Cu", size=(2, 2, 1), vacuum=10.0)
    slab.set_cell([6, 6, 6])
    slab.pbc = [True, True, False]
    pbc = slab.pbc
    cell = slab.cell
    positions = slab.positions
    cutoffs = [6 / 2] * len(slab)

    # ASE neighborlist to compare against
    pair_first, pair_second, dist = primitive_neighbor_list(
        "ijd",
        pbc,
        cell,
        positions,
        cutoffs,
        numbers=None,
        self_interaction=False,
        use_scaled_positions=False,
    )

    # device = "cpu"
    device = "cpu"
    pbc = torch.tensor(pbc).to(device)
    cell = torch.tensor(cell).to(device)
    positions = torch.tensor(positions).to(device)
    cutoffs = torch.tensor(cutoffs).to(device)

    torch_first, torch_second, torch_dist = torch_neighbor_list(
        "ijd",
        pbc,
        cell,
        positions,
        cutoffs,
        device=device,
        numbers=None,
        self_interaction=False,
        use_scaled_positions=False,
    )
