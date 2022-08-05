from itertools import product
import random
import torch
from torch_geometric.data import Batch


def all_frames(eigenvec, pos):
    """Compute all frames for a given graph
    Related to frame ambiguity issue

    Args:
        eigenvec (tensor): eigenvectors matrix
        pos (tensor): position vector (x-t)

    Returns:
        tensor: lists of 3D positions tensors
    """
    dim = pos.shape[1]  # to differentiate between 2D or 3D case
    plus_minus_list = list(product([-1, 1], repeat=dim))
    plus_minus_list = [torch.tensor(x) for x in plus_minus_list]
    all_fa = []

    for pm in plus_minus_list:

        # Append new graph positions to list
        new_eigenvec = pm * eigenvec

        # Check if eigenv is orthonormal
        if not torch.allclose(
            new_eigenvec @ new_eigenvec.T, torch.eye(dim), atol=1e-05
        ):
            continue

        # Check if determinant is 1
        if not torch.allclose(
            torch.linalg.det(new_eigenvec), torch.tensor(1.0), atol=1e-03
        ):
            continue

        # Consider frame if it passes above checks
        fa = pos @ new_eigenvec
        all_fa.append(fa)

    # Return one frame at random among plausible ones
    return random.choice(all_fa)

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
    if not torch.allclose(eigenvec @ eigenvec.T, torch.eye(dim), atol=1e-05):
        print("Matrix not orthogonal")

    # Check determinant of eigenvectors is 1
    if not torch.allclose(
        torch.linalg.det(eigenvec), torch.tensor(1.0), atol=1e-03
    ):
        print("Determinant is not 1")

def frame_averaging(g, random_sign=False):
    """Computes new positions for the graph atoms,
    using on frame averaging, which builds on PCA.

    Args:
        g (data.Data): input graph
        random_sign (bool): whether to pick sign of U at random

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

    # Compute all frames
    g.pos = all_frames(eigenvec, pos)
    
    # Update distances too
    g.distances = torch.sqrt(
        ((g.pos[g.edge_index[0, :]] - g.pos[g.edge_index[1, :]]) ** 2).sum(
            -1
        )
    ).to(dtype=g.distances.dtype)

    return g

def frame_averaging_2D(g, random_sign=True):
    """Computes new positions for the graph atoms,
    based on a frame averaging building on PCA.

    Args:
        g (_type_): graph
        random_sign (bool): True if we take a random sign for eigenv

    Returns:
        _type_: updated positions
    """

    # Compute centroid and covariance
    pos_2D = g.pos[:, :2] - g.pos[:, :2].mean(dim=0, keepdim=True)
    C = torch.matmul(pos_2D.t(), pos_2D)

    # Eigendecomposition
    eigenval, eigenvec = torch.linalg.eig(C)

    # Check if eigenvec, eigenval are real or complex ?
    # TODO: convert directly to real
    if not torch.isreal(eigenvec).all():
        print("Eigenvec is complex")
    else:
        eigenvec = eigenvec.real
        eigenval = eigenval.real

    # Sort, if necessary
    idx = eigenval.argsort(descending=True)
    eigenval = eigenval[idx]
    eigenvec = eigenvec[:, idx]

    # Compute all frames
    g.pos[:, :2] = all_frames(eigenvec, pos_2D)
    #g.pos = torch.cat((pos_2D, g.pos[:, 2].unsqueeze(1)), dim=1)

    # Update distances too
    g.distances = torch.sqrt(
        ((g.pos[g.edge_index[0, :]] - g.pos[g.edge_index[1, :]]) ** 2).sum(
            -1
        )
    ).to(dtype=g.distances.dtype)

    return g