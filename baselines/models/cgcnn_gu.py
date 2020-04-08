import torch
import torch.nn as nn
from baselines.common.registry import registry
from baselines.models.base import BaseModel
from baselines.modules.layers import CGCNNGuConv
from torch_geometric.nn import global_mean_pool


@registry.register_model("cgcnn_gu")
class CGCNNGu(BaseModel):
    """Implements a modified version of Crystal Graph ConvNet as outlined in
    https://pubs.acs.org/doi/abs/10.1021/acs.jpclett.0c00634
    """

    def __init__(
        self,
        num_atoms,
        bond_feat_dim,
        num_targets,
        atom_embedding_size=64,
        num_graph_conv_layers=6,
        fc_feat_size=128,
        num_fc_layers=4,
    ):
        super(CGCNNGu, self).__init__(num_atoms, bond_feat_dim, num_targets)
        self.embedding = nn.Linear(self.num_atoms, atom_embedding_size)

        self.convs = nn.ModuleList(
            [
                CGCNNGuConv(
                    node_dim=atom_embedding_size, edge_dim=self.bond_feat_dim
                )
                for _ in range(num_graph_conv_layers)
            ]
        )

        # Attention layers for https://pubs.acs.org/doi/abs/10.1021/acs.jpclett.0c00634
        # https://github.com/kaist-amsg/LS-CGCNN-ens/blob/master/cgcnn/cgcnn_bn_global_attn.py#L46
        self.attn_mlp = nn.Sequential(
            nn.Linear(atom_embedding_size, atom_embedding_size), nn.Tanh()
        )
        self.attn_sigmoid = nn.Sigmoid()

        self.conv_to_fc = nn.Sequential(
            nn.Linear(atom_embedding_size, fc_feat_size), nn.Softplus()
        )

        if num_fc_layers > 1:
            layers = []
            for _ in range(num_fc_layers - 1):
                layers.append(nn.Linear(fc_feat_size, fc_feat_size))
                layers.append(nn.Softplus())
            self.fcs = nn.Sequential(*layers)
        self.fc_out = nn.Linear(fc_feat_size, self.num_targets)

    def forward(self, data):
        node_feats = self.embedding(data.x)
        for f in self.convs:
            node_feats = f(node_feats, data.edge_index, data.edge_attr)

        # Attention as in https://pubs.acs.org/doi/abs/10.1021/acs.jpclett.0c00634
        # https://github.com/kaist-amsg/LS-CGCNN-ens/blob/master/cgcnn/cgcnn_bn_global_attn.py#L66
        v_ = global_mean_pool(node_feats, data.batch)
        c = self.attn_mlp(v_)
        a = self.attn_sigmoid(
            torch.sum(node_feats * c[data.batch], dim=1, keepdim=True)
        )
        node_feats = a * node_feats

        mol_feats = global_mean_pool(node_feats, data.batch)
        mol_feats = self.conv_to_fc(mol_feats)
        if hasattr(self, "fcs"):
            mol_feats = self.fcs(mol_feats)
        out = self.fc_out(mol_feats)
        return out
