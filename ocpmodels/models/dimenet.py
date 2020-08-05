import torch
from torch import nn
from torch_geometric.nn import DimeNet, radius_graph
from torch_scatter import scatter

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import get_pbc_distances


@registry.register_model("dimenet")
class DimeNetWrap(DimeNet):
    def __init__(
        self,
        num_atoms,
        bond_feat_dim,  # not used
        num_targets,
        use_pbc=True,
        regress_forces=True,
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
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.cutoff = cutoff

        super(DimeNetWrap, self).__init__(
            in_channels=hidden_channels,
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

        self.embedding = nn.Embedding(100, hidden_channels)

    def forward(self, data):
        pos = data.pos
        if self.regress_forces:
            pos = pos.requires_grad_(True)
        batch = data.batch
        x = self.embedding(data.atomic_numbers.long())
        if self.use_pbc:
            edge_index, dist = get_pbc_distances(
                pos, data.edge_index, data.cell, data.cell_offsets, data.natoms
            )
            j, i = edge_index
        else:
            edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
            j, i = edge_index
            dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(
            edge_index, num_nodes=x.size(0)
        )

        # Calculate angles.
        pos_i = pos[idx_i].detach()
        pos_ji, pos_ki = (
            pos[idx_j].detach() - pos_i,
            pos[idx_k].detach() - pos_i,
        )
        a = (pos_ji * pos_ki).sum(dim=-1)
        b = torch.cross(pos_ji, pos_ki).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        # Embedding block.
        x = self.emb(x, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))

        # Interaction blocks.
        for interaction_block, output_block in zip(
            self.interaction_blocks, self.output_blocks[1:]
        ):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P += output_block(x, rbf, i)

        energy = P.sum(dim=0) if batch is None else scatter(P, batch, dim=0)

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
