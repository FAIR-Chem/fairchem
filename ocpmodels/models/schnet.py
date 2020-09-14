import torch
from torch_geometric.nn import SchNet
from torch_scatter import scatter

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import get_pbc_distances


@registry.register_model("schnet")
class SchNetWrap(SchNet):
    def __init__(
        self,
        num_atoms,  # not used
        bond_feat_dim,  # not used
        num_targets,
        use_pbc=True,
        regress_forces=True,
        hidden_channels=128,
        num_filters=128,
        num_interactions=6,
        num_gaussians=50,
        cutoff=10.0,
        readout="add",
    ):
        self.num_targets = num_targets
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc

        super(SchNetWrap, self).__init__(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            readout=readout,
        )

    def forward(self, data):
        z = data.atomic_numbers.long()
        pos = data.pos
        if self.regress_forces:
            pos = pos.requires_grad_(True)
        batch = data.batch

        if self.use_pbc:
            assert z.dim() == 1 and z.dtype == torch.long

            out = get_pbc_distances(
                pos,
                data.edge_index,
                data.cell,
                data.cell_offsets,
                data.neighbors,
            )

            edge_index = out["edge_index"]
            edge_weight = out["distances"]
            edge_attr = self.distance_expansion(edge_weight)

            h = self.embedding(z)
            for interaction in self.interactions:
                h = h + interaction(h, edge_index, edge_weight, edge_attr)

            h = self.lin1(h)
            h = self.act(h)
            h = self.lin2(h)

            batch = torch.zeros_like(z) if batch is None else batch
            energy = scatter(h, batch, dim=0, reduce=self.readout)
        else:
            energy = super(SchNetWrap, self).forward(z, pos, batch)

        if self.regress_forces:
            forces = -1 * (
                torch.autograd.grad(
                    energy,
                    pos,
                    grad_outputs=torch.ones_like(energy),
                    create_graph=True,
                )[0]
            )
            return energy, forces
        else:
            return energy

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
