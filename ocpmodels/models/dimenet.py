from torch import nn
from torch_geometric.nn import DimeNet, radius_graph

from ocpmodels.common.registry import registry


@registry.register_model("dimenet")
class DimeNetWrap(DimeNet):
    def __init__(
        self,
        num_atoms,  # not used
        bond_feat_dim,  # not used
        num_targets,
        hidden_channels=128,
        num_blocks=6,
        num_bilinear=8,
        num_spherical=7,
        num_radial=6,
        cutoff=10.0,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_output_layers=3,
    ):
        self.num_targets = num_targets
        self.cutoff = cutoff

        super(DimeNetWrap, self).__init__(
            in_channels=95,
            hidden_channels=hidden_channels,
            out_channels=num_targets,
            num_blocks=num_blocks,
            num_bilinear=num_bilinear,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_output_layers=num_output_layers,
        )

        # self.atom_embedding = nn.Embedding(100, hidden_channels)
        # self.atom_embedding.reset_parameters()

    def forward(self, data):
        # x = self.atom_embedding(data.atomic_numbers.long())
        x = data.x
        # pos = data.pos
        pos = x[:, -3:]
        # edge_index = radius_graph(pos, r=self.cutoff, batch=data.batch)
        edge_index = data.edge_index

        return super(DimeNetWrap, self).forward(x, pos, edge_index, data.batch)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
