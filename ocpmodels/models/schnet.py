import torch
from torch_geometric.nn import SchNet
from torch_scatter import scatter

from ocpmodels.common.registry import registry


@registry.register_model("schnet")
class SchNetWrap(SchNet):
    def __init__(
        self,
        num_atoms,  # not used
        bond_feat_dim,  # not used
        num_targets,
        hidden_channels=128,
        num_filters=128,
        num_interactions=6,
        num_gaussians=50,
        cutoff=10.0,
    ):
        self.num_targets = num_targets

        super(SchNetWrap, self).__init__(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
        )

    def forward(self, data):
        z = data.atomic_numbers.long()
        pos = data.pos
        batch = data.batch

        # return super(SchNetWrap, self).forward(z, pos, batch)
        ############
        assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros_like(z) if batch is None else batch

        h = self.embedding(z)

        edge_index = data.edge_index
        edge_index[edge_index == -1] = 0
        row, col = edge_index

        edge_weight = pos[row] - pos[col]

        # correct for pbc
        cell = torch.repeat_interleave(data.cell, data.natoms * 12, dim=0)
        cell_offsets = data.cell_offsets
        offsets = (
            cell_offsets.float().view(-1, 1, 3).bmm(cell.float()).view(-1, 3)
        )
        edge_weight += offsets

        # compute distances
        edge_weight = edge_weight.norm(dim=-1)

        # remove zero distances
        nonzero_idx = torch.nonzero(edge_weight).flatten()
        edge_weight = edge_weight[nonzero_idx]
        edge_index = edge_index[:, nonzero_idx]

        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        out = scatter(h, batch, dim=0, reduce=self.readout)

        return out

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
