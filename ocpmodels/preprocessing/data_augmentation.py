import random
from copy import deepcopy
from itertools import product

import torch

from ocpmodels.common.transforms import RandomRotate


def all_frames(eigenvec, pos, cell, fa_frames="random", pos_3D=None):
    """Compute all frames for a given graph
    Related to frame ambiguity issue

    Args:
        eigenvec (tensor): eigenvectors matrix
        pos (tensor): position vector (X-1t)
        cell (tensor): cell direction (3x3)
        fa_frames: whether to return one random frame (random),
            one deterministic frame (det), all frames (all)
            or SE(3) frames (se3- as prefix).
        pos_3D: 3rd position coordinate of atoms

    Returns:
        tensor: lists of 3D positions tensors
    """
    dim = pos.shape[1]  # to differentiate between 2D or 3D case
    plus_minus_list = list(product([1, -1], repeat=dim))
    plus_minus_list = [torch.tensor(x) for x in plus_minus_list]
    all_fa = []
    all_cell = []
    se3 = fa_frames in {
        "se3-all",
        "se3-random",
        "se3-det",
    }
    fa_cell = deepcopy(cell)

    for pm in plus_minus_list:

        # Append new graph positions to list
        new_eigenvec = pm * eigenvec

        # Check if determinant is 1 for SE(3) case
        if se3 and not torch.allclose(
            torch.linalg.det(new_eigenvec), torch.tensor(1.0), atol=1e-03
        ):
            continue

        # Consider frame if it passes above check
        fa_pos = pos @ new_eigenvec

        if pos_3D is not None:
            fa_pos = torch.cat((fa_pos, pos_3D.unsqueeze(1)), dim=1)
            fa_cell[:, :2, :2] = cell[:, :2, :2] @ new_eigenvec
        else:
            fa_cell = new_eigenvec.t() @ cell

        all_fa.append(fa_pos)
        all_cell.append(fa_cell)

    # Handle rare case where no R is positive orthogonal
    if all_fa == []:
        all_fa.append(pos @ eigenvec)
        all_cell.append(cell @ eigenvec)

    # Return frame(s) depending on method fa_frames
    if fa_frames == "all" or fa_frames == "se3-all":
        return all_fa, all_cell

    elif fa_frames == "det" or fa_frames == "se3-det":
        return [all_fa[0]], [all_cell[0]]

    return [random.choice(all_fa)], [random.choice(all_cell)]


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
    eigenval, eigenvec = torch.linalg.eig(C)

    # Check if eigenvec, eigenval are real or complex ?
    if not torch.isreal(eigenvec).all():
        print("Eigenvec is complex")
    else:
        eigenvec = eigenvec.real
        eigenval = eigenval.real

    # Sort, if necessary
    idx = eigenval.argsort(descending=True)
    eigenvec = eigenvec[:, idx]
    eigenval = eigenval[idx]

    # Compute fa_pos
    g.fa_pos, g.fa_cell = all_frames(eigenvec, pos, g.cell, fa_frames)

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
    eigenval, eigenvec = torch.linalg.eig(C)

    # Convert eigenval to real values
    eigenvec = eigenvec.real
    eigenval = eigenval.real

    # Sort, if necessary
    idx = eigenval.argsort(descending=True)
    eigenval = eigenval[idx]
    eigenvec = eigenvec[:, idx]

    # Compute all frames
    g.fa_pos, g.fa_cell = all_frames(eigenvec, pos_2D, g.cell, fa_frames, g.pos[:, 2])
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


def check_pbc_fa(val_loader, cell):
    x_diff, y_diff = [], []
    fa_x_diff, fa_y_diff = [], []
    cell_diff = []
    for batch in val_loader:
        b = batch[0]
        for i in range(batch_size):
            x_diff.append(
                max(b.pos[b.batch == i][:, 0]) - min(b.pos[b.batch == i][:, 0])
            )
            fa_x_diff.append(
                max(b.fa_pos[0][b.batch == i][:, 0])
                - min(b.fa_pos[0][b.batch == i][:, 0])
            )
            y_diff.append(
                max(b.pos[b.batch == i][:, 1]) - min(b.pos[b.batch == i][:, 1])
            )
            fa_y_diff.append(
                max(b.fa_pos[0][b.batch == i][:, 1])
                - min(b.fa_pos[0][b.batch == i][:, 1])
            )
    # Do smth with cell and cell_offsets too.

    # check distances after out_pbc when fa_pos is applied !
    # rotate cell offsets of rotated graphs ?
