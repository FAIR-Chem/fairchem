import pickle
import torch
from torch_geometric.data import (InMemoryDataset, Data)
from tqdm import tqdm
from itertools import repeat, product
import numpy as np

class Convert():
    def __init__(self, in_path, out_path):
        super(Convert, self).__init__()
        self.in_path = in_path
        self.out_path = out_path
        

    def collate(self, data_list):
        keys = data_list[0].keys
        data = data_list[0].__class__()

        for key in keys:
            data[key] = []
        slices = {key: [0] for key in keys}

        for item, key in product(data_list, keys):
            data[key].append(item[key])
            if torch.is_tensor(item[key]):
                s = slices[key][-1] + item[key].size(
                    item.__cat_dim__(key, item[key]))
            elif isinstance(item[key], int) or isinstance(item[key], float):
                s = slices[key][-1] + 1
            else:
                raise ValueError('Unsupported attribute type')
            slices[key].append(s)

        if hasattr(data_list[0], '__num_nodes__'):
            data.__num_nodes__ = []
            for item in data_list:
                data.__num_nodes__.append(item.num_nodes)

        for key in keys:
            if torch.is_tensor(data_list[0][key]):
                data[key] = torch.cat(
                    data[key], dim=data.__cat_dim__(key, data_list[0][key]))
            else:
                data[key] = torch.tensor(data[key])
            slices[key] = torch.tensor(slices[key], dtype=torch.long)

        return data, slices
    
    def geometric(self):
        data = pickle.load(open(self.in_path, "rb"))

        # data[0][0] is 22 x 98; atom_feat
        # data[0][1] is 22 x 12 x 6; nbr_feat
        # data[0][2] is 22 x 12; nbr_idx
        # data[0][3] is 1; target binding energy

        # We have to convert it to be compatible with the pytorch geometric API
        # i.e. it should have x (same as atom_fea), edge_index, edge_attr,
        # y (same as target), and optionally pos (I think).

        data_list = []
        graph_atom_idx, connectivity_atom_idx = [], []
        graph_idx = 0
        base_idx = 0
        
        for idx, i in tqdm(enumerate(data)):
            x = i[0]
            dist = i[3]

            n_i = x.shape[0]
            new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
            graph_atom_idx.append(new_idx)
            base_idx += n_i

                #Creating Mask that only considers atoms with distance <= 2
            connectivity_idx = np.where(dist <= 2)[0]
            connectivity_base = np.zeros((n_i,1))
            connectivity_base[connectivity_idx] = 1
        #     connectivity_atom_idx.append(torch.FloatTensor(connectivity_base))

            graph_idx += 1

            y = i[-1].view(-1).float()
            edge_index = [[], []]
        #             edge_attr = torch.zeros(i[2].shape[0] * i[2].shape[1], i[1].shape[-1])
            edge_attr = []
            for j in range(i[2].shape[0]):
                for k in range(i[2].shape[1]):

#                     if i[2][j,k] != 99: # Filter out fake neighbors
                    edge_index[0].append(j)
                    edge_index[1].append(i[2][j, k])
    #                         edge_attr[j * i[2].shape[1] + k] = i[1][j, k].clone()
                    edge_attr.append(i[1][j, k].clone())

            edge_index = torch.LongTensor(edge_index)
            if not edge_attr:
                print(idx)
            edge_attr = torch.stack(edge_attr).float()
            connectivity = torch.FloatTensor(connectivity_base)   

            data_list.append(
                Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=y,
                    connectivity = connectivity,
                    graph_idx = torch.tensor(graph_idx).view(-1),
                    pos=None
                )
            )
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.out_path)
        return data, slices
        
