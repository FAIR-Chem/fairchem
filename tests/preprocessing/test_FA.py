"""
Test if distances are preserved with periodic boundary conditions
when applying rotations on the graph, including frame averaging rotations.
"""

import sys
import warnings
from copy import deepcopy
from pathlib import Path

import torch

from ocpmodels.common.transforms import RandomRotate
from ocpmodels.common.utils import get_pbc_distances, make_script_trainer
from ocpmodels.datasets.data_transforms import FrameAveraging
from ocpmodels.trainers import EnergyTrainer

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


def distance_preservation_test(g, method, distances):
    """Test if distances are preserved with periodic boundary conditions
    when applying rotations on the graph, including frame averaging rotations

    Args:
        g (data.Data): one graph
        method (str): name of method tested
        distances (tensor): initial (pbc) distances between atom pairs,
            which we want preserved in transformed graph.

    Prints True
    """

    if method in {"Rotated graph", "RandomRotate"}:
        pos = g.pos
        cell = g.cell
    elif hasattr(g, "fa_pos"):
        pos = g.fa_pos[0]
        cell = g.fa_cell[0]
    else:
        raise ValueError("Unsupported method type")

    out = get_pbc_distances(
        pos,
        g.edge_index,
        cell,
        g.cell_offsets,
        g.neighbors,
        return_distance_vec=True,
    )

    # assert torch.allclose(out["distances"], distances, atol=1e-03)
    print(
        f"Result for {method}: ",
        torch.allclose(out["distances"], distances, atol=1e-03),
    )


if __name__ == "__main__":

    config = {}
    config["graph_rewiring"] = "remove-tag-0"
    config["test_ri"] = True

    str_args = sys.argv[1:]
    if all("config" not in arg for arg in str_args):
        str_args.append("--is_debug")
        str_args.append("--config=sfarinet-is2re-10k")
        warnings.warn(
            "No model / mode is given; chosen as default" + f"Using: {str_args[-1]}"
        )

    trainer: EnergyTrainer = make_script_trainer(str_args=str_args, overrides=config)

    fa_transform = FrameAveraging("2D", "random")

    for batch in trainer.val_loader:
        break
    # Set up

    for i in range(len(batch[0].sid)):

        # FA pos
        g = deepcopy(batch[0][i])
        d = deepcopy(g.distances)
        g = fa_transform(g)
        distance_preservation_test(g, "FA", d)

        # Rotated graph
        g = deepcopy(batch[0][i])
        R = torch.tensor([[0.6, -0.8, 0], [0.8, 0.6, 0], [0, 0, 1]])
        g.pos = g.pos @ R.t()
        g.cell = g.cell @ R.t()
        distance_preservation_test(g, "Rotated graph", d)

        # Original method
        g = deepcopy(batch[0][i])
        transform = RandomRotate([-180, 180], [2])
        g, _, _ = transform(g)
        distance_preservation_test(g, "RandomRotate", d)

        # FA of rotated graph
        g = deepcopy(batch[0][i])
        R = torch.tensor([[0.6, -0.8, 0], [0.8, 0.6, 0], [0, 0, 1]])
        g.pos = g.pos @ R.t()
        g.cell = g.cell @ R.t()
        g = fa_transform(g)
        distance_preservation_test(g, "FA on rotated graph", d)

        print(" -- Next --")
