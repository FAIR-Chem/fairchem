from math import pi as PI

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class CGCNNConv(MessagePassing):
    """Implements the message passing layer from
    `"Crystal Graph Convolutional Neural Networks for an
    Accurate and Interpretable Prediction of Material Properties"
    <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301>`.
    """

    def __init__(self, node_dim, edge_dim, cutoff=6.0, **kwargs):
        super(CGCNNConv, self).__init__(aggr="add")
        self.node_feat_size = node_dim
        self.edge_feat_size = edge_dim
        self.cutoff = cutoff

        self.fc_pre = nn.Sequential(
            nn.Linear(
                2 * self.node_feat_size + self.edge_feat_size,
                2 * self.node_feat_size,
            ),
            nn.BatchNorm1d(2 * self.node_feat_size),
        )

        self.fc_post = nn.Sequential(nn.BatchNorm1d(self.node_feat_size))

    def forward(self, x, edge_index, edge_weight, edge_attr):
        """
        Arguments:
            x has shape [num_nodes, node_feat_size]
            edge_index has shape [2, num_edges]
            edge_attr is [num_edges, edge_feat_size]
        """
        # This weighting is not in the original cgcnn implementation, but helps
        # performance quite a bit. Idea borrowed from SchNet.
        edge_weight = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)

        return self.propagate(
            edge_index, x=x, edge_weight=edge_weight, edge_attr=edge_attr
        )

    def message(self, x_i, x_j, edge_weight, edge_attr):
        """
        Arguments:
            x_i has shape [num_edges, node_feat_size]
            x_j has shape [num_edges, node_feat_size]
            edge_attr has shape [num_edges, edge_feat_size]

        Returns:
            tensor of shape [num_edges, node_feat_size]
        """
        z = self.fc_pre(
            torch.cat([x_i, x_j, edge_attr], dim=1)
        ) * edge_weight.view(-1, 1)
        z1, z2 = z.chunk(2, dim=1)
        z1 = nn.Sigmoid()(z1)
        z2 = nn.Softplus()(z2)
        return z1 * z2

    def update(self, aggr_out, x):
        """
        Arguments:
            aggr_out has shape [num_nodes, node_feat_size]
                This is the result of aggregating features output by the
                `message` function from neighboring nodes. The aggregation
                function is specified in the constructor (`add` for CGCNN).
            x has shape [num_nodes, node_feat_size]

        Returns:
            tensor of shape [num_nodes, node_feat_size]
        """
        aggr_out = nn.Softplus()(x + self.fc_post(aggr_out))
        return aggr_out


class CGCNNGuConv(MessagePassing):
    """Implements the message passing layer from
    `"Crystal Graph Convolutional Neural Networks for an
    Accurate and Interpretable Prediction of Material Properties"
    <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301>`
    with modifications from https://pubs.acs.org/doi/abs/10.1021/acs.jpclett.0c00634.
    """

    def __init__(self, node_dim, edge_dim, **kwargs):
        super(CGCNNGuConv, self).__init__(aggr="add")
        self.node_feat_size = node_dim
        self.edge_feat_size = edge_dim

        self.fc_pre = nn.Sequential(
            nn.Linear(
                2 * self.node_feat_size + self.edge_feat_size,
                2 * self.node_feat_size,
            ),
            nn.BatchNorm1d(2 * self.node_feat_size),
        )

        self.fc_post = nn.Sequential(nn.BatchNorm1d(self.node_feat_size))

    def forward(self, x, edge_index, edge_attr):
        """
        Arguments:
            x has shape [num_nodes, node_feat_size]
            edge_index has shape [2, num_edges]
            edge_attr is [num_edges, edge_feat_size]
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        """
        Arguments:
            x_i has shape [num_edges, node_feat_size]
            x_j has shape [num_edges, node_feat_size]
            edge_attr has shape [num_edges, edge_feat_size]

        Returns:
            tensor of shape [num_edges, node_feat_size]
        """
        z = self.fc_pre(torch.cat([x_i, x_j, edge_attr], dim=1))
        z1, z2 = z.chunk(2, dim=1)
        z1 = nn.Sigmoid()(z1)
        z2 = nn.Tanh()(z2)
        return z1 * z2

    def update(self, aggr_out, x):
        """
        Arguments:
            aggr_out has shape [num_nodes, node_feat_size]
                This is the result of aggregating features output by the
                `message` function from neighboring nodes. The aggregation
                function is specified in the constructor (`add` for CGCNN).
            x has shape [num_nodes, node_feat_size]

        Returns:
            tensor of shape [num_nodes, node_feat_size]
        """
        aggr_out = nn.Tanh()(x + self.fc_post(aggr_out))
        return aggr_out


class AttentionConv(MessagePassing):
    """Implements the graph attentional operator from
    `"Path-Augmented Graph Transformer Network" <https://arxiv.org/abs/1905.12712>`.

    Also related to `"Graph Transformer" <https://openreview.net/forum?id=HJei-2RcK7>`
    and `"Graph Attention Networks" <https://arxiv.org/abs/1710.10903>`.
    """

    def __init__(
        self,
        node_dim,
        edge_dim,
        hidden_dim=64,
        num_heads=3,
        concat=True,
        negative_slope=0.2,
        dropout=0,
        **kwargs
    ):
        super(AttentionConv, self).__init__(aggr="add")

        self.node_feat_size = node_dim
        self.edge_feat_size = edge_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.concat = concat
        self.negative_slope = negative_slope

        self.att = nn.Sequential(
            nn.Linear(
                2 * self.node_feat_size + self.edge_feat_size,
                self.hidden_dim * self.num_heads,
            ),
            nn.LeakyReLU(negative_slope=self.negative_slope),
            nn.Linear(self.hidden_dim * self.num_heads, self.num_heads),
        )
        self.dropout = nn.Dropout(p=dropout)
        self.value = nn.Linear(
            self.node_feat_size + self.edge_feat_size,
            self.hidden_dim * self.num_heads,
        )

        if self.concat is True:
            self.fc_post = nn.Linear(
                self.num_heads * self.hidden_dim, self.node_feat_size
            )
        else:
            self.fc_post = nn.Linear(self.hidden_dim, self.node_feat_size)

    def forward(self, x, edge_index, edge_attr):
        """
        Arguments:
            x has shape [num_nodes, node_feat_size]
            edge_index has shape [2, num_edges]
            edge_attr is [num_edges, edge_feat_size]
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr, edge_index_i, size_i):
        """
        Arguments:
            x_i has shape [num_edges, node_feat_size]
            x_j has shape [num_edges, node_feat_size]
            edge_attr has shape [num_edges, edge_feat_size]

        Returns:
            tensor of shape [num_edges, num_heads, hidden_dim]
        """
        alpha = self.att(torch.cat([x_i, x_j, edge_attr], dim=-1))
        alpha = softmax(alpha, edge_index_i, size_i)
        alpha = self.dropout(alpha).view(-1, self.num_heads, 1)

        value = self.value(torch.cat([x_j, edge_attr], dim=-1)).view(
            -1, self.num_heads, self.hidden_dim
        )
        return value * alpha

    def update(self, aggr_out, x):
        """
        Arguments:
            aggr_out has shape [num_nodes, num_heads, hidden_dim]
                This is the result of aggregating features output by the
                `message` function from neighboring nodes. The aggregation
                function is specified in the constructor (`add` in this case).
            x has shape [num_nodes, node_feat_size]

        Returns:
            tensor of shape [num_nodes, node_feat_size]
        """
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.num_heads * self.hidden_dim)
        else:
            aggr_out = aggr_out.mean(dim=1)

        aggr_out = nn.ReLU()(x + self.fc_post(aggr_out))

        return aggr_out


class DOGSSConv(MessagePassing):
    def __init__(self, node_dim, edge_dim, **kwargs):
        super(DOGSSConv, self).__init__(aggr="add")
        self.node_feat_size = node_dim
        self.edge_feat_size = edge_dim

        self.fc_pre_node = nn.Sequential(
            nn.Linear(
                2 * self.node_feat_size + self.edge_feat_size,
                2 * self.node_feat_size,
            ),
            nn.BatchNorm1d(2 * self.node_feat_size),
        )

        self.fc_pre_edge = nn.Sequential(
            nn.Linear(
                2 * self.node_feat_size + self.edge_feat_size,
                2 * self.edge_feat_size,
            ),
            nn.BatchNorm1d(2 * self.edge_feat_size),
        )
        self.fc_node = nn.Sequential(nn.BatchNorm1d(self.node_feat_size))
        self.fc_edge = nn.Sequential(nn.BatchNorm1d(self.edge_feat_size))

    def forward(self, x, edge_index, edge_attr):
        """
        Arguments:
            x has shape [num_nodes, node_feat_size]
            edge_index has shape [2, num_edges]
            edge_attr is [num_edges, edge_feat_size]
        """

        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        """
        Arguments:
            x_i has shape [num_edges, node_feat_size]
            x_j has shape [num_edges, node_feat_size]
            edge_attr has shape [num_edges, edge_feat_size]

        Returns:
            tensor of shape [num_edges, node_feat_size]
        """
        z = torch.cat([x_i, x_j, edge_attr], dim=1)

        z_node = self.fc_pre_node(z)
        z1_node, z2_node = z_node.chunk(2, dim=1)
        z1_node = nn.Sigmoid()(z1_node)
        z2_node = nn.Softplus()(z2_node)

        z_edge = self.fc_pre_edge(z)
        z1_edge, z2_edge = z_edge.chunk(2, dim=1)
        z1_edge = nn.Sigmoid()(z1_edge)
        z2_edge = nn.Softplus()(z2_edge)

        msg_node = z1_node * z2_node
        msg_edge = z1_edge * z2_edge

        # Edge messages are not being passed to aggregation.
        self.edge_attr = edge_attr
        self.msg_edge = msg_edge
        return msg_node

    def update(self, aggr_out, x):
        """
        Arguments:
            aggr_out has shape [num_nodes, node_feat_size]
                This is the result of aggregating features output by the
                `message` function from neighboring nodes. The aggregation
                function is specified in the constructor (`add` for CGCNN).
            x has shape [num_nodes, node_feat_size]

        Returns:
            tensor of shape [num_nodes, node_feat_size]
            tensor of shape [num_edgs, edge_feat_size]
        """
        aggr_node = nn.Softplus()(x + self.fc_node(aggr_out))
        aggr_edge = nn.Softplus()(self.edge_attr + self.fc_edge(self.msg_edge))
        return aggr_node, aggr_edge
