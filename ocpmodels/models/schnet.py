from torch_geometric.nn import SchNet

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

        return super(SchNetWrap, self).forward(z, pos, batch)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
