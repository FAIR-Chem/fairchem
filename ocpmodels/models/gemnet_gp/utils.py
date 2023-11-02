"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import json
from typing import Optional, Tuple

import torch
from torch_scatter import segment_csr


def read_json(path: str):
    """"""
    if not path.endswith(".json"):
        raise UserWarning(f"Path {path} is not a json-path.")

    with open(path, "r") as f:
        content = json.load(f)
    return content


def update_json(path: str, data) -> None:
    """"""
    if not path.endswith(".json"):
        raise UserWarning(f"Path {path} is not a json-path.")

    content = read_json(path)
    content.update(data)
    write_json(path, content)


def write_json(path: str, data) -> None:
    """"""
    if not path.endswith(".json"):
        raise UserWarning(f"Path {path} is not a json-path.")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def read_value_json(path: str, key):
    """"""
    content = read_json(path)

    if key in content.keys():
        return content[key]
    else:
        return None


def ragged_range(sizes):
    """Multiple concatenated ranges.

    Examples
    --------
        sizes = [1 4 2 3]
        Return: [0  0 1 2 3  0 1  0 1 2]
    """
    assert sizes.dim() == 1
    if sizes.sum() == 0:
        return sizes.new_empty(0)

    # Remove 0 sizes
    sizes_nonzero = sizes > 0
    if not torch.all(sizes_nonzero):
        sizes = torch.masked_select(sizes, sizes_nonzero)

    # Initialize indexing array with ones as we need to setup incremental indexing
    # within each group when cumulatively summed at the final stage.
    id_steps = torch.ones(sizes.sum(), dtype=torch.long, device=sizes.device)
    id_steps[0] = 0
    insert_index = sizes[:-1].cumsum(0)
    insert_val = (1 - sizes)[:-1]

    # Assign index-offsetting values
    id_steps[insert_index] = insert_val

    # Finally index into input array for the group repeated o/p
    res = id_steps.cumsum(0)
    return res


def repeat_blocks(
    sizes: torch.Tensor,
    repeats,
    continuous_indexing: bool = True,
    start_idx: int = 0,
    block_inc: int = 0,
    repeat_inc: int = 0,
) -> torch.Tensor:
    """Repeat blocks of indices.
    Adapted from https://stackoverflow.com/questions/51154989/numpy-vectorized-function-to-repeat-blocks-of-consecutive-elements

    continuous_indexing: Whether to keep increasing the index after each block
    start_idx: Starting index
    block_inc: Number to increment by after each block,
               either global or per block. Shape: len(sizes) - 1
    repeat_inc: Number to increment by after each repetition,
                either global or per block

    Examples
    --------
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = False
        Return: [0 0 0  0 1 2 0 1 2  0 1 0 1 0 1]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True
        Return: [0 0 0  1 2 3 1 2 3  4 5 4 5 4 5]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        repeat_inc = 4
        Return: [0 4 8  1 2 3 5 6 7  4 5 8 9 12 13]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        start_idx = 5
        Return: [5 5 5  6 7 8 6 7 8  9 10 9 10 9 10]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        block_inc = 1
        Return: [0 0 0  2 3 4 2 3 4  6 7 6 7 6 7]
        sizes = [0,3,2] ; repeats = [3,2,3] ; continuous_indexing = True
        Return: [0 1 2 0 1 2  3 4 3 4 3 4]
        sizes = [2,3,2] ; repeats = [2,0,2] ; continuous_indexing = True
        Return: [0 1 0 1  5 6 5 6]
    """
    assert sizes.dim() == 1
    assert all(sizes >= 0)

    # Remove 0 sizes
    sizes_nonzero = sizes > 0
    if not torch.all(sizes_nonzero):
        assert block_inc == 0  # Implementing this is not worth the effort
        sizes = torch.masked_select(sizes, sizes_nonzero)
        if isinstance(repeats, torch.Tensor):
            repeats = torch.masked_select(repeats, sizes_nonzero)
        if isinstance(repeat_inc, torch.Tensor):
            repeat_inc = torch.masked_select(repeat_inc, sizes_nonzero)

    if isinstance(repeats, torch.Tensor):
        assert all(repeats >= 0)
        insert_dummy = repeats[0] == 0
        if insert_dummy:
            one = sizes.new_ones(1)
            zero = sizes.new_zeros(1)
            sizes = torch.cat((one, sizes))
            repeats = torch.cat((one, repeats))
            if isinstance(block_inc, torch.Tensor):
                block_inc = torch.cat((zero, block_inc))
            if isinstance(repeat_inc, torch.Tensor):
                repeat_inc = torch.cat((zero, repeat_inc))
    else:
        assert repeats >= 0
        insert_dummy = False

    # Get repeats for each group using group lengths/sizes
    r1 = torch.repeat_interleave(
        torch.arange(len(sizes), device=sizes.device), repeats
    )

    # Get total size of output array, as needed to initialize output indexing array
    N = int((sizes * repeats).sum().item())

    # Initialize indexing array with ones as we need to setup incremental indexing
    # within each group when cumulatively summed at the final stage.
    # Two steps here:
    # 1. Within each group, we have multiple sequences, so setup the offsetting
    # at each sequence lengths by the seq. lengths preceding those.
    id_ar = torch.ones(N, dtype=torch.long, device=sizes.device)
    id_ar[0] = 0
    insert_index = sizes[r1[:-1]].cumsum(0)
    insert_val = (1 - sizes)[r1[:-1]]

    if isinstance(repeats, torch.Tensor) and torch.any(repeats == 0):
        diffs = r1[1:] - r1[:-1]
        indptr = torch.cat((sizes.new_zeros(1), diffs.cumsum(0)))
        if continuous_indexing:
            # If a group was skipped (repeats=0) we need to add its size
            insert_val += segment_csr(sizes[: r1[-1]], indptr, reduce="sum")

        # Add block increments
        if isinstance(block_inc, torch.Tensor):
            insert_val += segment_csr(
                block_inc[: r1[-1]], indptr, reduce="sum"
            )
        else:
            insert_val += block_inc * (indptr[1:] - indptr[:-1])
            if insert_dummy:
                insert_val[0] -= block_inc
    else:
        idx = r1[1:] != r1[:-1]
        if continuous_indexing:
            # 2. For each group, make sure the indexing starts from the next group's
            # first element. So, simply assign 1s there.
            insert_val[idx] = 1

        # Add block increments
        insert_val[idx] += block_inc

    # Add repeat_inc within each group
    if isinstance(repeat_inc, torch.Tensor):
        insert_val += repeat_inc[r1[:-1]]
        if isinstance(repeats, torch.Tensor):
            repeat_inc_inner = repeat_inc[repeats > 0][:-1]
        else:
            repeat_inc_inner = repeat_inc[:-1]
    else:
        insert_val += repeat_inc
        repeat_inc_inner = repeat_inc

    # Subtract the increments between groups
    if isinstance(repeats, torch.Tensor):
        repeats_inner = repeats[repeats > 0][:-1]
    else:
        repeats_inner = repeats
    insert_val[r1[1:] != r1[:-1]] -= repeat_inc_inner * repeats_inner

    # Assign index-offsetting values
    id_ar[insert_index] = insert_val

    if insert_dummy:
        id_ar = id_ar[1:]
        if continuous_indexing:
            id_ar[0] -= 1

    # Set start index now, in case of insertion due to leading repeats=0
    id_ar[0] += start_idx

    # Finally index into input array for the group repeated o/p
    res = id_ar.cumsum(0)
    return res


def calculate_interatomic_vectors(
    R: torch.Tensor,
    id_s: torch.Tensor,
    id_t: torch.Tensor,
    offsets_st: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the vectors connecting the given atom pairs,
    considering offsets from periodic boundary conditions (PBC).

    Parameters
    ----------
        R: Tensor, shape = (nAtoms, 3)
            Atom positions.
        id_s: Tensor, shape = (nEdges,)
            Indices of the source atom of the edges.
        id_t: Tensor, shape = (nEdges,)
            Indices of the target atom of the edges.
        offsets_st: Tensor, shape = (nEdges,)
            PBC offsets of the edges.
            Subtract this from the correct direction.

    Returns
    -------
        (D_st, V_st): tuple
            D_st: Tensor, shape = (nEdges,)
                Distance from atom t to s.
            V_st: Tensor, shape = (nEdges,)
                Unit direction from atom t to s.
    """
    Rs = R[id_s]
    Rt = R[id_t]
    # ReLU prevents negative numbers in sqrt
    if offsets_st is None:
        V_st = Rt - Rs  # s -> t
    else:
        V_st = Rt - Rs + offsets_st  # s -> t
    D_st = torch.sqrt(torch.sum(V_st**2, dim=1))
    V_st = V_st / D_st[..., None]
    return D_st, V_st


def inner_product_normalized(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Calculate the inner product between the given normalized vectors,
    giving a result between -1 and 1.
    """
    return torch.sum(x * y, dim=-1).clamp(min=-1, max=1)


def mask_neighbors(neighbors: torch.Tensor, edge_mask: torch.Tensor):
    neighbors_old_indptr = torch.cat([neighbors.new_zeros(1), neighbors])
    neighbors_old_indptr = torch.cumsum(neighbors_old_indptr, dim=0)
    neighbors = segment_csr(edge_mask.long(), neighbors_old_indptr)
    return neighbors
