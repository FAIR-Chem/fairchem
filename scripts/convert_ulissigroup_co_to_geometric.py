import pickle
from itertools import product, repeat

import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

data_path = "data/data/2020_02_16_ulissigroup_co/initial/initial.pkl"
data_out_path = "data/data/2020_02_16_ulissigroup_co/initial/data.pt"


def collate(data_list):
    keys = data_list[0].keys
    data = data_list[0].__class__()

    for key in keys:
        data[key] = []
    slices = {key: [0] for key in keys}

    for item, key in product(data_list, keys):
        data[key].append(item[key])
        if torch.is_tensor(item[key]):
            s = slices[key][-1] + item[key].size(
                item.__cat_dim__(key, item[key])
            )
        elif isinstance(item[key], int) or isinstance(item[key], float):
            s = slices[key][-1] + 1
        else:
            raise ValueError("Unsupported attribute type")
        slices[key].append(s)

    if hasattr(data_list[0], "__num_nodes__"):
        data.__num_nodes__ = []
        for item in data_list:
            data.__num_nodes__.append(item.num_nodes)

    for key in keys:
        if torch.is_tensor(data_list[0][key]):
            data[key] = torch.cat(
                data[key], dim=data.__cat_dim__(key, data_list[0][key])
            )
        else:
            data[key] = torch.tensor(data[key])
        slices[key] = torch.tensor(slices[key], dtype=torch.long)

    return data, slices


if __name__ == "__main__":
    data = pickle.load(open(data_path, "rb"))

    # data[0][0] is 22 x 98; atom_feat
    # data[0][1] is 22 x 12 x 6; nbr_feat
    # data[0][2] is 22 x 12; nbr_idx
    # data[0][3] is 1; target binding energy

    # We have to convert it to be compatible with the pytorch geometric API
    # i.e. it should have x (same as atom_fea), edge_index, edge_attr,
    # y (same as target), and optionally pos (I think).

    data_list = []

    for i in tqdm(data):
        x = i[0]
        y = i[3]
        edge_index = [[], []]
        edge_attr = torch.FloatTensor(
            i[2].shape[0] * i[2].shape[1], i[1].shape[-1]
        )
        for j in range(i[2].shape[0]):
            for k in range(i[2].shape[1]):
                edge_index[0].append(j)
                edge_index[1].append(i[2][j, k])
                edge_attr[j * i[2].shape[1] + k] = i[1][j, k].clone()
        edge_index = torch.LongTensor(edge_index)
        data_list.append(
            Data(
                x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=None
            )
        )

    data, slices = collate(data_list)
    torch.save((data, slices), data_out_path)
