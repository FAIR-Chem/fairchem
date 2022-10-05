import random
from itertools import product

import torch
from torch_geometric.data import Batch

from ocpmodels.common.transforms import RandomRotate


def all_frames(eigenvec, pos, fa_frames="random", pos_3D=None):
    """Compute all frames for a given graph
    Related to frame ambiguity issue

    Args:
        eigenvec (tensor): eigenvectors matrix
        pos (tensor): position vector (X-1t)
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
    e3 = fa_frames in {
        "all",
        "random",
        "det",
    }

    for pm in plus_minus_list:

        # Append new graph positions to list
        new_eigenvec = pm * eigenvec

        # Check if determinant is 1
        if not e3 and not torch.allclose(
            torch.linalg.det(new_eigenvec), torch.tensor(1.0), atol=1e-03
        ):
            continue

        # Consider frame if it passes above check
        fa = pos @ new_eigenvec
        if pos_3D is not None:
            all_fa.append(torch.cat((fa, pos_3D.unsqueeze(1)), dim=1))
        else:
            all_fa.append(fa)

    # Handle rare case where no R is positive orthogonal
    if all_fa == []:
        all_fa.append(pos @ eigenvec)

    # Return frame(s) depending on method fa_frames
    if fa_frames == "all" or fa_frames == "se3-all":
        return all_fa

    elif fa_frames == "det" or fa_frames == "se3-det":
        return [all_fa[0]]

    return [random.choice(all_fa)]


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
    g.fa_pos = all_frames(eigenvec, pos, fa_frames)

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
    g.fa_pos = all_frames(eigenvec, pos_2D, fa_frames, g.pos[:, 2])

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
