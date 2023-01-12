import random
from copy import deepcopy
from itertools import product

import torch

from ocpmodels.common.graph_transforms import RandomRotate


def all_frames(eigenvec, pos, cell, fa_frames="random", pos_3D=None, det_index=0):
    """Compute all frames for a given graph
    Related to frame ambiguity issue

    Args:
        eigenvec (tensor): eigenvectors matrix
        pos (tensor): position vector (X-1t)
        cell (tensor): cell direction (3x3)
        fa_frames: whether to return one random frame (random),
            one deterministic frame (det), all frames (all)
            or SE(3) frames (se3- as prefix).
        pos_3D: 3rd position coordinate of atoms, for 2D FA.

    Returns:
        tensor: lists of 3D positions tensors
    """
    dim = pos.shape[1]  # to differentiate between 2D or 3D case
    plus_minus_list = list(product([1, -1], repeat=dim))
    plus_minus_list = [torch.tensor(x) for x in plus_minus_list]
    all_fa_pos = []
    all_cell = []
    all_rots = []
    se3 = fa_frames in {
        "se3-all",
        "se3-random",
        "se3-det",
        "se3-multiple",
    }
    fa_cell = deepcopy(cell)

    if fa_frames == "det" or fa_frames == "se3-det":
        sum_eigenvec = torch.sum(eigenvec, axis=0)
        plus_minus_list = [torch.where(sum_eigenvec >= 0, 1.0, -1.0)]

    for pm in plus_minus_list:

        # Append new graph positions to list
        new_eigenvec = pm * eigenvec

        # Consider frame if it passes above check
        fa_pos = pos @ new_eigenvec

        if pos_3D is not None:
            full_eigenvec = torch.eye(3)
            fa_pos = torch.cat((fa_pos, pos_3D.unsqueeze(1)), dim=1)
            full_eigenvec[:2, :2] = new_eigenvec
            new_eigenvec = full_eigenvec

        if cell is not None:
            fa_cell = cell @ new_eigenvec

        # Check if determinant is 1 for SE(3) case
        if se3 and not torch.allclose(
            torch.linalg.det(new_eigenvec), torch.tensor(1.0), atol=1e-03
        ):
            continue

        all_fa_pos.append(fa_pos)
        all_cell.append(fa_cell)
        all_rots.append(new_eigenvec.unsqueeze(0))

    # Handle rare case where no R is positive orthogonal
    if all_fa_pos == []:
        all_fa_pos.append(fa_pos)
        all_cell.append(fa_cell)
        all_rots.append(new_eigenvec.unsqueeze(0))

    # Return frame(s) depending on method fa_frames
    if fa_frames == "all" or fa_frames == "se3-all":
        return all_fa_pos, all_cell, all_rots

    if fa_frames == "multiple" or fa_frames == "se3-multiple":
        index = torch.bernoulli(torch.tensor([0.5] * len(all_fa_pos)))
        if index.sum() == 0:
            index = random.randint(0, len(all_fa_pos) - 1)
            return [all_fa_pos[index]], [all_cell[index]], [all_rots[index]]
        if index.sum() == 1:
            return [all_fa_pos[index]], [all_cell[index]], [all_rots[index]]
        else:
            all_fa_pos = [a for a, b in zip(all_fa_pos, index) if b]
            all_cell = [a for a, b in zip(all_cell, index) if b]
            all_rots = [a for a, b in zip(all_rots, index) if b]
            return all_fa_pos, all_cell, all_rots

    elif fa_frames == "det" or fa_frames == "se3-det":
        return [all_fa_pos[det_index]], [all_cell[det_index]], [all_rots[det_index]]

    index = random.randint(0, len(all_fa_pos) - 1)
    return [all_fa_pos[index]], [all_cell[index]], [all_rots[index]]


def check_constraints(eigenval, eigenvec, dim):
    """Check requirements for frame averaging are satisfied

    Args:
        eigenval (tensor): eigenvalues
        eigenvec (tensor): eigenvectors
        dim (int): 2D or 3D frame averaging
    """
    # Check eigenvalues are different
    if dim == 3:
        if (eigenval[1] / eigenval[0] > 0.90) or (eigenval[2] / eigenval[1] > 0.90):
            print("Eigenvalues are quite similar")
    else:
        if eigenval[1] / eigenval[0] > 0.90:
            print("Eigenvalues are quite similar")

    # Check eigenvectors are orthonormal
    if not torch.allclose(eigenvec @ eigenvec.T, torch.eye(dim), atol=1e-03):
        print("Matrix not orthogonal")

    # Check determinant of eigenvectors is 1
    if not torch.allclose(torch.linalg.det(eigenvec), torch.tensor(1.0), atol=1e-03):
        print("Determinant is not 1")


def frame_averaging_3D(g, fa_frames="random"):
    """Computes new positions for the graph atoms,
    using on frame averaging, which builds on PCA.

    Args:
        g (data.Data): input graph
        fa_frames (str): FA method used (random, det, all, se3)

    Returns:
        data.Data: graph with updated positions (and distances)
    """

    # Compute centroid and covariance
    pos = g.pos - g.pos.mean(dim=0, keepdim=True)
    C = torch.matmul(pos.t(), pos)

    # Eigendecomposition
    eigenval, eigenvec = torch.linalg.eigh(C)

    # Sort, if necessary
    idx = eigenval.argsort(descending=True)
    eigenvec = eigenvec[:, idx]
    eigenval = eigenval[idx]

    # Compute fa_pos
    g.fa_pos, g.fa_cell, g.fa_rot = all_frames(
        eigenvec, pos, g.cell if hasattr(g, "cell") else None, fa_frames
    )

    # No need to update distances, they are preserved.

    return g


def frame_averaging_2D(g, fa_frames="random"):
    """Computes new positions for the graph atoms,
    based on a frame averaging building on PCA.

    Args:
        g (data.Data): graph
        fa_frames (str): FA method used (random, det, all, se3)

    Returns:
        _type_: updated positions
    """

    # Compute centroid and covariance
    pos_2D = g.pos[:, :2] - g.pos[:, :2].mean(dim=0, keepdim=True)
    C = torch.matmul(pos_2D.t(), pos_2D)

    # Eigendecomposition
    eigenval, eigenvec = torch.linalg.eigh(C)
    # Sort, if necessary
    idx = eigenval.argsort(descending=True)
    eigenval = eigenval[idx]
    eigenvec = eigenvec[:, idx]

    # Compute all frames
    g.fa_pos, g.fa_cell, g.fa_rot = all_frames(
        eigenvec, pos_2D, g.cell if hasattr(g, "cell") else None, fa_frames, g.pos[:, 2]
    )
    # No need to update distances, they are preserved.

    return g


def data_augmentation(g, *args):
    """Data augmentation where we randomly rotate each graph
    in the dataloader transform

    Args:
        g (data.Data): single graph
        rotation (str, optional): around which axis do we rotate it.
            Defaults to 'z'.

    Returns:
        (data.Data): rotated graph
    """

    # Sampling a random rotation within [-180, 180] for all axes.
    transform = RandomRotate([-180, 180], [2])
    # transform = RandomRotate([-180, 180], [0, 1, 2])  # 3D

    # Rotate graph
    graph_rotated, _, _ = transform(g)

    return graph_rotated
