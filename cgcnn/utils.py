import os
import shutil

import numpy as np
from sklearn import metrics

import torch
from torch_geometric.utils import remove_self_loops


def save_checkpoint(state, is_best, checkpoint_dir="checkpoints/"):
    filename = os.path.join(checkpoint_dir, "checkpoint.pth.tar")
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(
            filename, os.path.join(checkpoint_dir, "model_best.pth.tar")
        )


# https://stackoverflow.com/questions/38987/how-to-merge-two-dictionaries-in-a-single-expression
def update_config(original, update):
    """
    Recursively update a dict.
    Subdict's won't be overwritten but also updated.
    """
    for key, value in original.items():
        if key not in update:
            update[key] = value
        elif isinstance(value, dict):
            update_config(value, update[key])
    return update


class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data
