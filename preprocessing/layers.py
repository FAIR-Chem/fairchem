import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

   
class DOGSSConv(MessagePassing):
    """Implements the message passing layer from
    `"Crystal Graph Convolutional Neural Networks for an
    Accurate and Interpretable Prediction of Material Properties"
    <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301>`.
    """

    def __init__(self, node_dim, edge_dim, **kwargs):
        super(DOGSSConv, self).__init__(aggr="add")
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        self.fc_pre = nn.Sequential(
            nn.Linear(2 * self.node_dim + self.edge_dim, 2 * self.node_dim),
            nn.BatchNorm1d(2 * self.node_dim),
        )

        self.fc_post = nn.Sequential(nn.BatchNorm1d(self.node_dim))

    def forward(self, x, edge_index, edge_attr):
        """
        Arguments:
            x has shape [num_nodes, node_dim]
            edge_index has shape [2, num_edges]
            edge_attr is [num_edges, edge_dim]
        """
#         print(edge_index.is_cuda, x.is_cuda, edge_attr.is_cuda)
        if not edge_index.is_cuda:
            edge_index = edge_index.cuda()
        
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        """
        Arguments:
            x_i has shape [num_edges, node_dim]
            x_j has shape [num_edges, node_dim]
            edge_attr has shape [num_edges, edge_dim]

        Returns:
            tensor of shape [num_edges, node_dim]
        """
        z = self.fc_pre(torch.cat([x_i, x_j, edge_attr], dim=1))
        z1, z2 = z.chunk(2, dim=1)
        z1 = nn.Sigmoid()(z1)
        z2 = nn.Softplus()(z2)
        return z1 * z2

    def update(self, aggr_out, x):
        """
        Arguments:
            aggr_out has shape [num_nodes, node_dim]
                This is the result of aggregating features output by the
                `message` function from neighboring nodes. The aggregation
                function is specified in the constructor (`add` for CGCNN).
            x has shape [num_nodes, node_dim]

        Returns:
            tensor of shape [num_nodes, node_dim]
        """
        aggr_out = nn.Softplus()(x + self.fc_post(aggr_out))
        return aggr_out
