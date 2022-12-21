import math
import random
import sys
from copy import deepcopy
from pathlib import Path

from ocpmodels.preprocessing.frame_averaging import (
    frame_averaging_2D,
    frame_averaging_3D,
)

sys.path.append(str(Path(__file__).resolve().parent.parent))

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
        graph, _ = frame_averaging_2D(graph, fa_frames="det")
    else:
        graph, _ = frame_averaging_3D(graph, fa_frames="det")

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


def custom_rotate_graph(g, degrees=None):
    # Return rotated graph
    rotated_graph = deepcopy(g)
    if degrees is None:
        degrees = (0, 360)
        degree = math.pi * random.uniform(*degrees) / 180.0
    else:
        degree = math.pi * degrees / 180.0
    sin, cos = math.sin(degree), math.cos(degree)
    R = torch.tensor([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])
    rotated_graph.pos = (g.pos @ R.T).to(g.pos.device, g.pos.dtype)
    rotated_graph.cell = torch.matmul(rotated_graph.cell, R)
    return rotated_graph, R


def summary_fa(g):
    pos_2D = g.pos[:, :2] - g.pos[:, :2].mean(dim=0, keepdim=True)
    C = torch.matmul(pos_2D.t(), pos_2D)
    eigenval, eigenvec = torch.linalg.eigh(C)
    idx = eigenval.argsort(descending=True)
    eigenval = eigenval[idx]
    eigenvec = eigenvec[:, idx]
    # sum_eigenvec = torch.sum(eigenvec, axis=0)
    # sign = torch.where(sum_eigenvec >= 0, 1., -1.)
    sign = torch.ones(2)
    new_eigenvec = sign * eigenvec
    fa_pos = g.pos[:, :2] @ new_eigenvec
    return fa_pos, new_eigenvec, sign, eigenval, C


if __name__ == "__main__":

    opts = resolved_args()
    checkpoint_path = None
    config = {}
    # config["graph_rewiring"] = "remove-tag-0"
    config["frame_averaging"] = "2D"
    config["fa_frames"] = "det"
    config["test_ri"] = True
    config["optim"] = {"max_epochs": 0}

    str_args = sys.argv[1:]
    if all("config" not in arg for arg in str_args):
        str_args.append("--is_debug")
        str_args.append("--config=sfarinet-is2re-10k")

    trainer = make_script_trainer(str_args=str_args, overrides=config)

    # Compute rotated graph and corresponding fa_pos, fa_cell, fa_rot
    for i, batch in enumerate(trainer.loaders["val_id"]):
        break
        rotated_batch_1 = trainer.rotate_graph(batch, rotation="z")
        rotated_batch_2 = trainer.rotate_graph(batch, rotation="z")
        fa_pos_batch = batch[0]["fa_pos"][0]
        fa_pos_batch_1 = rotated_batch_1["batch_list"][0]["fa_pos"][0]
        fa_pos_batch_2 = rotated_batch_2["batch_list"][0]["fa_pos"][0]
        pos_diff = torch.sum(torch.abs(fa_pos_batch - fa_pos_batch_1))
        pos_diff2 = torch.sum(torch.abs(fa_pos_batch - fa_pos_batch_2))
        if pos_diff < 1 and pos_diff2 < 1:
            print("PERFECT")
        # Some prints
        print(pos_diff)
        print(pos_diff2)
        print("Position matrices")
        print(fa_pos_batch[:2, :])
        print(fa_pos_batch_1[:2, :])
        print(fa_pos_batch_2[:2, :])
        print("Rotation matrix: \n", rotated_batch_1["rot"])
        print("Rotation matrix 2: \n ", rotated_batch_2["rot"])
        print("PCA matrix of batch: \n ", batch[0]["fa_rot"][0][0])
        print(
            "1st PCA matrix of rotated batch: \n ",
            rotated_batch_1["batch_list"][0]["fa_rot"][0][0],
        )
        print(
            "2st PCA matrix of rotated batch: \n",
            rotated_batch_2["batch_list"][0]["fa_rot"][0][0],
        )

        if i == 10:
            break

    # For a finer analysus

    # Choose one graph only
    b = deepcopy(batch[0])
    # Otherwise can't iterate
    b_fa_pos = b.fa_pos
    b_fa_rot = b.fa_rot
    b_fa_cell = b.fa_cell
    delattr(b, "fa_pos")
    delattr(b, "fa_cell")
    delattr(b, "fa_rot")
    for i in range(len(b.sid) - 1):
        # Get graph and reset attributes
        g = deepcopy(Batch.get_example(b, i))
        g.fa_pos = b_fa_pos[0][b.ptr[i] : b.ptr[i + 1]]
        g.fa_rot = b_fa_rot[0][i, :, :]
        g.fa_cell = b_fa_cell[0][i, :, :]
        count = 0
        count_fix = 0
        for j in range(0, 360, 60):
            rotated_g, R = custom_rotate_graph(deepcopy(g), degrees=j)
            # FA summary
            fa_pos, U, sign, eigenval, C = summary_fa(g)
            r_fa_pos, r_U, r_sign, r_eigenval, r_C = summary_fa(rotated_g)
            print(fa_pos[:3], "\n", U, "\n", sign, "\n", eigenval, "\n", C)
            print(r_fa_pos[:3], "\n", r_U, "\n", r_sign, "\n", r_eigenval, "\n", r_C)
            # Frame averaging on rotated graph
            rotated_g = frame_averaging_2D(rotated_g, "det")
            # print(R)
            # print(rotated_g.fa_rot[0])
            # Pos difference
            pos_diff = torch.sum(torch.abs(g.fa_pos - rotated_g.fa_pos[0]))
            # print(g.fa_pos[:2])
            # print(rotated_g.fa_pos[0][:2])
            # Prints
            sign_diff = r_sign * sign
            r_new_U = r_U * sign_diff
            new_r_fa_pos = rotated_g.pos[:, :2] @ r_new_U
            new_pos_diff = torch.sum(torch.abs(fa_pos - new_r_fa_pos))
            if pos_diff < 1:
                print(j)
                count += 1
            if new_pos_diff < 1:
                print("fix: ", j)
                count_fix += 1
        if count == 360 / 40:
            print("ALL OF THEM")
        # rotated_graph, all_fa = frame_averaging_3D(rotated_graph, False)
