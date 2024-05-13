"""
Modified from https://github.com/lucidrains/performer-pytorch/blob/main/performer_pytorch/performer_pytorch.py
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def orthogonal_matrix_chunk(cols, device=None):
    unstructured_block = torch.randn((cols, cols), device=device)
    q, r = torch.linalg.qr(unstructured_block, mode='reduced')
    return q.t()  # [cols, cols]


@torch.no_grad()
def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, device=None):
    """create 2D Gaussian orthogonal matrix"""
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    normalizer = final_matrix.norm(p=2, dim=1, keepdim=True)
    normalizer[normalizer == 0] = 1e-5
    final_matrix = final_matrix / normalizer

    return final_matrix


@torch.no_grad()
def orthogonal_matrix_chunk_batched(bsz, cols, device=None):
    unstructured_block = torch.randn((bsz, cols, cols), device=device)
    q, r = torch.linalg.qr(unstructured_block, mode='reduced')
    return q.transpose(2, 1)  # [bsz, cols, cols]


@torch.no_grad()
def gaussian_orthogonal_random_matrix_batched(nb_samples, nb_rows, nb_columns, device=None, dtype=torch.float32):
    """create 2D Gaussian orthogonal matrix"""
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk_batched(nb_samples, nb_columns, device=device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk_batched(nb_samples, nb_columns, device=device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list, dim=1).type(dtype)
    final_matrix = F.normalize(final_matrix, p=2, dim=2)
    return final_matrix
