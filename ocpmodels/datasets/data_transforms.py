import torch

from ocpmodels.preprocessing.frame_averaging import (
    data_augmentation,
    frame_averaging_2D,
    frame_averaging_3D,
)
from ocpmodels.preprocessing.graph_rewiring import (
    one_supernode_per_atom_type,
    one_supernode_per_atom_type_dist,
    one_supernode_per_graph,
    remove_tag0_nodes,
)


class Transform:
    def __call__(self, data):
        raise NotImplementedError

    def __str__(self):
        name = self.__class__.__name__
        items = [
            f"{k}={v}"
            for k, v in self.__dict__.items()
            if not callable(v) and k != "inactive"
        ]
        s = f"{name}({', '.join(items)})"
        if self.inactive:
            s = f"[inactive] {s}"
        return s


class FrameAveraging(Transform):
    r"""Frame Averaging (FA) Transform for (PyG) Data objects (e.g. 3D atomic graphs).
    Computes new atomic positions (`fa_pos`) for all datapoints, as well as new unit
    cells (`fa_cell`) attributes for crystal structures, when applicable. The rotation
    matrix (`fa_rot`) used for the frame averaging is also stored.

    Args:
        frame_averaging (str): Transform method used.
            Can be 2D FA, 3D FA, Data Augmentation or no FA, respectively denoted by
            (`"2D"`, `"3D"`, `"DA"`, `""`)
        fa_method (str): the actual frame averaging technique used.
            "stochastic" refers to sampling one frame at random (at each epoch), "det"
            to chosing deterministically one frame, and "all" to using all frames. The
            prefix "se3-" refers to the SE(3) equivariant version of the method. ""
            means that no frame averaging is used. (`""`, `"stochastic"`, `"all"`,
            `"det"`, `"se3-stochastic"`, `"se3-all"`, `"se3-det"`)

    Returns:
        (data.Data): updated data object with new positions (+ unit cell) attributes
        and the rotation matrices used for the frame averaging transform.
    """

    def __init__(self, frame_averaging=None, fa_method=None):
        self.fa_method = (
            "random" if (fa_method is None or fa_method == "") else fa_method
        )
        self.frame_averaging = "" if frame_averaging is None else frame_averaging
        self.inactive = not self.frame_averaging
        assert self.frame_averaging in {
            "",
            "2D",
            "3D",
            "DA",
        }
        assert self.fa_method in {
            "",
            "random",
            "det",
            "all",
            "se3-random",
            "se3-det",
            "se3-all",
        }

        if self.frame_averaging:
            if self.frame_averaging == "2D":
                self.fa_func = frame_averaging_2D
            elif self.frame_averaging == "3D":
                self.fa_func = frame_averaging_3D
            elif self.frame_averaging == "DA":
                self.fa_func = data_augmentation
            else:
                raise ValueError(f"Unknown frame averaging: {self.frame_averaging}")

    def __call__(self, data):
        if self.inactive:
            return data
        elif self.frame_averaging == "DA":
            return self.fa_func(data, self.fa_method)
        else:
            data.fa_pos, data.fa_cell, data.fa_rot = self.fa_func(
                data.pos, data.cell if hasattr(data, "cell") else None, self.fa_method
            )
            return data


class GraphRewiring(Transform):
    def __init__(self, rewiring_type=None) -> None:
        self.rewiring_type = rewiring_type

        self.inactive = not self.rewiring_type

        if self.rewiring_type:
            if self.rewiring_type == "remove-tag-0":
                self.rewiring_func = remove_tag0_nodes
            elif self.rewiring_type == "one-supernode-per-graph":
                self.rewiring_func = one_supernode_per_graph
            elif self.rewiring_type == "one-supernode-per-atom-type":
                self.rewiring_func = one_supernode_per_atom_type
            elif self.rewiring_type == "one-supernode-per-atom-type-dist":
                self.rewiring_func = one_supernode_per_atom_type_dist
            else:
                raise ValueError(f"Unknown self.graph_rewiring {self.graph_rewiring}")

    def __call__(self, data):
        if self.inactive:
            return data

        data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
        data.natoms = torch.tensor([data.natoms])
        data.ptr = torch.tensor([0, data.natoms])

        return self.rewiring_func(data)


class Compose:
    # https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#Compose
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class AddAttributes:
    def __call__(self, data):
        if (
            not hasattr(data, "distances")
            and hasattr(data, "edge_index")
            and data.edge_index is not None
        ):
            data.distances = torch.sqrt(
                (
                    (data.pos[data.edge_index[0, :]] - data.pos[data.edge_index[1, :]])
                    ** 2
                ).sum(-1)
            ).float()
        return data


def get_transforms(trainer_config):
    transforms = [
        AddAttributes(),
        GraphRewiring(trainer_config.get("graph_rewiring")),
        FrameAveraging(trainer_config["frame_averaging"], trainer_config["fa_method"]),
    ]
    return Compose(transforms)
