import math
import random
from copy import deepcopy
from itertools import product

import torch
from minydra import resolved_args
from torch_geometric.data import Batch

from ocpmodels.common.utils import make_script_trainer


def test_rotation_invariance(graph, rotation="z", dim="2D"):
    """Investigate if the frame averaging output is
    the same for the original graph and a rotated version.
    Basically test rotation invariance of such method.

    Args:
        graph (_type_): input molecule we want to rotate
        rotation (str): axis on which we rotate the graph
        dim (str): whether we focus on 2D or 3D frame averaging.

    Returns:
        bool: True if FA yields rotation invariant rep.
    """
    # Frame averaging for original graph
    if dim == "2D":
        graph, _ = frame_averaging_2D(graph, fa_frames="random")
    else:
        graph, _ = frame_averaging_3D(graph, fa_frames="random")

    # Rotate graph
    rotated_graph = deepcopy(graph)
    degrees = (0, 180)
    degree = math.pi * random.uniform(*degrees) / 180.0
    sin, cos = math.sin(degree), math.cos(degree)

    if rotation == "x":
        R = torch.tensor([[1, 0, 0], [0, cos, sin], [0, -sin, cos]])
    elif rotation == "y":
        R = torch.tensor([[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]])
    else:
        R = torch.tensor([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])
    rotated_graph.pos = (graph.pos @ R.T).to(graph.pos.device, graph.pos.dtype)

    # Frame averaging rotated graph
    if dim == "2D":
        rotated_graph, all_fa = frame_averaging_2D(rotated_graph, False)
    else:
        rotated_graph, all_fa = frame_averaging_3D(rotated_graph, False)

    # Check if one of these frames equal the input frame (for rotated examples)
    count = 0  # count times fa is equal to original fa
    for fa in all_fa:
        if not torch.allclose(fa, graph.updated_pos, atol=1e-2):
            count += 1

    # Check if FA is the same for default eigenvector. Shortcut to above.
    # if torch.allclose(rotated_graph.updated_pos, graph.updated_pos, atol=1e-2):
    #    print("Default frame is rotation invariant")

    return count == len(all_fa) - 1


def all_frames_deprecated(eigenvec, pos):
    """Compute all frames for a given graph
    Related to frame ambiguity issue

    Args:
        eigenvec (_type_): eigenvectors matrix
        pos (_type_): position vector (x-t)
        t (_type_, optional): original fa, for rotated matrix.
            Defaults to None.

    Returns:
        _type_: lists of 3D positions tensors
    """
    dim = pos.shape[1]  # to differentiate between 2D or 3D case
    plus_minus_list = list(product([-1, 1], repeat=dim))
    plus_minus_list = [torch.tensor(x) for x in plus_minus_list]
    all_fa = []

    for pm in plus_minus_list:

        # Append new graph positions to list
        new_eigenvec = pm * eigenvec

        # Do not check orthogonality or determinant = 1.

        # Consider frame if it passes above checks
        fa = pos @ new_eigenvec
        all_fa.append(fa)

    return all_fa


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
            print("Issue: eigenvalues too similar")
    else:
        if eigenval[1] / eigenval[0] > 0.90:
            print("Issue: eigenvalues too similar")

    # Check eigenvectors are orthonormal
    if not torch.allclose(eigenvec @ eigenvec.T, torch.eye(dim), atol=1e-05):
        print("Matrix not orthogonal")

    # Check determinant of eigenvectors is 1
    if not torch.allclose(torch.linalg.det(eigenvec), torch.tensor(1.0), atol=1e-03):
        print("Determinant is not 1")


def frame_averaging_3D(g, random_sign=False):
    """Computes new positions for the graph atoms,
    based on a frame averaging building on PCA.

    Args:
        g (_type_): graph
        random_sign (bool): whether to pick sign of U at random

    Returns:
        _type_: updated positions
    """

    # Compute centroid and covariance
    pos = g.pos - g.pos.mean(dim=0, keepdim=True)
    C = torch.matmul(pos.t(), pos)

    # Eigen decomposition
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
    all_fa = all_frames_deprecated(eigenvec, pos)

    # Change signs of eigenvectors
    if random_sign:
        plus_minus = torch.randint(0, 2, (3,))
        plus_minus[plus_minus == 0] = -1
        eigenvec = plus_minus * eigenvec

    # Compute new positions
    g.updated_pos = pos @ eigenvec

    # Check if FA constraints are satisfied
    check_constraints(eigenval, eigenvec, dim=3)

    return g, all_fa


def frame_averaging_2D(g, random_sign=False):
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

    # Eigen decomposition
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
    all_fa = all_frames_deprecated(eigenvec, pos_2D)
    all_fa = [torch.cat((item, g.pos[:, 2].unsqueeze(1)), dim=1) for item in all_fa]

    # TODO: remove, simply select from all_fa. Update g.pos

    # Change signs of eigenvectors
    if random_sign:
        plus_minus = torch.randint(0, 2, (2,))
        plus_minus[plus_minus == 0] = -1
        eigenvec = plus_minus * eigenvec
    # Compute new positions
    g.updated_pos = deepcopy(g.pos)
    g.updated_pos[:, :2] = pos_2D @ eigenvec

    # Check if FA constraints are satisfied
    check_constraints(eigenval, eigenvec, dim=2)

    return g, all_fa


if __name__ == "__main__":

    opts = resolved_args()

    trainer_config = {
        "optim": {
            "num_workers": 4,
            "batch_size": 64,
        },
        "logger": {
            "dummy",
        },
    }

    if opts.victor_local:
        trainer_config["dataset"][0]["src"] = "data/is2re/All/train/data.lmdb"
        trainer_config["dataset"] = trainer_config["dataset"][:1]
        trainer_config["optim"]["num_workers"] = 0
        trainer_config["optim"]["batch_size"] = (
            opts.bs or trainer_config["optim"]["batch_size"]
        )

    trainer = make_script_trainer(overrides=trainer_config)

    for batch in trainer.train_loader:
        break
    b = batch[0]

    count = 0
    count_1 = 0
    for i in range(len(b.sid)):
        g = Batch.get_example(b, i)

        # Check invariance to rotations of 3D frame averaging
        if not test_rotation_invariance(g, rotation="z", dim="3D"):
            print("Not rotation invariant around z")
            count_1 += 1
        if not test_rotation_invariance(g, rotation="y", dim="3D"):
            print("Not rotation invariant around y")
        if not test_rotation_invariance(g, rotation="x", dim="3D"):
            print("Not rotation invariant around x")

        # Check invariance to rotations of frame averaging on 2D only
        if not test_rotation_invariance(g, rotation="z", dim="2D"):
            print("Not rotation invariant around z")
            count += 1
        # Here: we only expect rotation invariance around z

    print(f"Proportion of 2D non-rotation invariance wrt z: {count / i}%")
    print(f"Proportion of 3D non-rotation invariance wrt z: {count_1 / i}%")
