"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Union
from argparse import Namespace
import numpy
import scipy.stats as stats
import torch
from torch import nn as nn
from torch.nn import LayerNorm, functional as F
from torch_geometric.nn import radius_graph
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter

from ocpmodels.models.base import BaseModel
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc,
)

@registry.register_model("GraphTransformer")
class GraphTransformer(BaseModel):
    """Using graph transformer architecture from GROVER model from the
    `"Self-Supervised Graph Transformer on Large-Scale Molecular Data" <https://arxiv.org/abs/2007.02835>`_.

    [replace with gtransformer specific info]
    Each layer uses interaction block of the form:

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \odot
        h_{\mathbf{\Theta}} ( \exp(-\gamma(\mathbf{e}_{j,i} - \mathbf{\mu}))),

    Args:
        num_atoms (int): Unused argument
        bond_feat_dim (int): Unused argument
        num_targets (int): Number of targets to predict.
        use_pbc (bool, optional): If set to :obj:`True`, account for periodic boundary conditions.
            (default: :obj:`True`)
        regress_forces (bool, optional): If set to :obj:`True`, predict forces by differentiating
            energy with respect to positions.
            (default: :obj:`True`)
        otf_graph (bool, optional): If set to :obj:`True`, compute graph edges on the fly.
            (default: :obj:`False`)
        hidden_channels (int, optional): Number of hidden channels.
            (default: :obj:`128`)
        num_filters (int, optional): Number of filters to use.
            (default: :obj:`128`)
        num_interactions (int, optional): Number of interaction blocks
            (default: :obj:`6`)
        num_gaussians (int, optional): The number of gaussians :math:`\mu`.
            (default: :obj:`50`)
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        readout (string, optional): Whether to apply :obj:`"add"` or
            :obj:`"mean"` global aggregation. (default: :obj:`"add"`)
    """
## FROM SCHNET
    # TODO: remove unnecessary arguments
    def __init__(
        self,
        args, # Can refactor code to only do explicit args instead of namespace in create_ffn
        num_atoms,  # not used (i dont think)
        bond_feat_dim,  # not used for schnet, but needed for gtrans
        num_targets,
        use_pbc=True,
        regress_forces=True,
        otf_graph=False,
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
        self.cutoff = cutoff
        self.otf_graph = otf_graph

        # Settings taken from default single-GPU GROVER training settings (besides PReLU)
        self.encoders = GTransEncoder(hidden_size=100,
                                     edge_fdim=bond_feat_dim,
                                     node_fdim=4, # node features are of dimension 4: atomic number, pos_x, pos_y, pos_z
                                     dropout=0.1,
                                     activation="ReLU",
                                     num_mt_block=1,
                                     num_attn_head=4,
                                     atom_emb_output=False,  # options: True, False, None, "atom", "bond", "both"
                                     bias=False,
                                     cuda=True,
                                     res_connection=False)

        self.mol_atom_from_atom_ffn = self.create_ffn(args)
        self.mol_atom_from_bond_ffn = self.create_ffn(args)

    # TODO: change inputs to only what's explicitly needed (instead of namespace)
    # From GROVER-finetune (FFNs for GTransformer output embeddings)
    def create_ffn(self, args: Namespace):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        # Note: args.features_dim is set according the real loaded features data
        if args.features_only:
            first_linear_dim = args.features_size + args.features_dim
        else:
            if args.self_attention:
                first_linear_dim = args.hidden_size * args.attn_out
                # GROVERTODO: Ad-hoc!
                # if args.use_input_features:
                first_linear_dim += args.features_dim
            else:
                first_linear_dim = args.hidden_size + args.features_dim

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)
        # GROVERTODO: ffn_hidden_size
        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, args.output_size),
            ])

        # Create FFN model
        return nn.Sequential(*ffn)

    # TODO: look at OCP loss function and incorporate this new GTrans loss
    # From GROVER-finetune, will have to look into OCP loss function and edit it to incorporate both atom and bond loss
    def get_loss_func(args):
        def loss_func(preds, targets,
                      dt=args.dataset_type,
                      dist_coff=args.dist_coff):

            if dt == 'classification':
                pred_loss = nn.BCEWithLogitsLoss(reduction='none')
            elif dt == 'regression':
                pred_loss = nn.MSELoss(reduction='none')
            else:
                raise ValueError(f'Dataset type "{args.dataset_type}" not supported.')

            # print(type(preds))
            # GROVERTODO: Here, should we need to involve the model status? Using len(preds) is just a hack.
            if type(preds) is not tuple:
                # in eval mode.
                return pred_loss(preds, targets)

            # in train mode.
            dist_loss = nn.MSELoss(reduction='none')
            # dist_loss = nn.CosineSimilarity(dim=0)
            # print(pred_loss)

            dist = dist_loss(preds[0], preds[1])
            pred_loss1 = pred_loss(preds[0], targets)
            pred_loss2 = pred_loss(preds[1], targets)
            return pred_loss1 + pred_loss2 + dist_coff * dist

        return loss_func

## Combination of SchNet and DimeNet++ (dimenet does not include edge_attr)
    @conditional_grad(torch.enable_grad())
    def _forward(self, data):
        atomic_numbers = data.atomic_numbers.long()
        pos = data.pos
        batch = data.batch

        if self.otf_graph:
            edge_index, cell_offsets, neighbors = radius_graph_pbc(
                data, self.cutoff, 50, data.pos.device
            )
            data.edge_index = edge_index
            data.cell_offsets = cell_offsets
            data.neighbors = neighbors

        if self.use_pbc:
            out = get_pbc_distances(
                pos,
                data.edge_index,
                data.cell,
                data.cell_offsets,
                data.neighbors,
            )

            edge_index = out["edge_index"]
            dist = out["distances"]
        else:
            edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
            j, i = edge_index
            dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        data.edge_attr = self.distance_expansion(dist)

        # Convert to format for GROVER
        batch = convert_input(data)

        # From GROVER
        output = self.encoders(batch)

        mol_atom_from_bond_output = output[1]
        mol_atom_from_atom_output = output[0]

        # TODO: figure out how OCP checks training vs evaluation modes, adapt code here
        # From GROVER-finetune, will have to adapt to tell if OCP is in training or evaluation mode
        # During training it looks like model should have both atom and bond outputs, do we need to change OCP code?
        if self.training:
            atom_ffn_output = self.mol_atom_from_atom_ffn(mol_atom_from_atom_output)
            bond_ffn_output = self.mol_atom_from_bond_ffn(mol_atom_from_bond_output)
            return atom_ffn_output, bond_ffn_output
        else:
            atom_ffn_output = self.mol_atom_from_atom_ffn(mol_atom_from_atom_output)
            bond_ffn_output = self.mol_atom_from_bond_ffn(mol_atom_from_bond_output)
            # Our task is not classification, it's regression on energy/forces, commented this out
            # if self.classification:
            #     atom_ffn_output = self.sigmoid(atom_ffn_output)
            #     bond_ffn_output = self.sigmoid(bond_ffn_output)
            output = (atom_ffn_output + bond_ffn_output) / 2

        # How to parse output from GROVER:
        # if self.embedding_output_type == 'atom':
        #     return {"atom_from_atom": output[0], "atom_from_bond": output[1],
        #             "bond_from_atom": None, "bond_from_bond": None}  # atom_from_atom, atom_from_bond
        # elif self.embedding_output_type == 'bond':
        #     return {"atom_from_atom": None, "atom_from_bond": None,
        #             "bond_from_atom": output[0], "bond_from_bond": output[1]}  # bond_from_atom, bond_from_bond
        # elif self.embedding_output_type == "both":
        #     return {"atom_from_atom": output[0][0], "bond_from_atom": output[0][1],
        #             "atom_from_bond": output[1][0], "bond_from_bond": output[1][1]}

        # TODO: check to see if the 'output' above will actually be trained to converge to energy (ocp training/loss)
        energy = output
        energy = 0 # Need to get energy over the whole system from these 4 embeddings (pooling maybe?)

        # FROM DIMENET (will need to replace with GROVER specific
        # # Embedding block.
        # x = self.emb(data.atomic_numbers.long(), rbf, i, j)
        # P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))
        #
        # # Interaction blocks.
        # for interaction_block, output_block in zip(
        #         self.interaction_blocks, self.output_blocks[1:]
        # ):
        #     x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
        #     P += output_block(x, rbf, i, num_nodes=pos.size(0))
        #
        # energy = P.sum(dim=0) if batch is None else scatter(P, batch, dim=0)

        return energy

    # How does this function work? Do we need to change it
    def forward(self, data):
        if self.regress_forces:
            data.pos.requires_grad_(True)
        energy = self._forward(data)

        if self.regress_forces:
            forces = -1 * (
                torch.autograd.grad(
                    energy,
                    data.pos,
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

# Helper function to onvert from PyTorch Geometric input to GROVER input:
# Helper function to convert from PyTorch Geometric input to GROVER input:
def convert_input(data):
    """
        :param data: data as PyTorch geometric object
        :param f_atoms: the atom features, num_atoms * atom_dim
        :param f_bonds: the bond features, num_bonds * bond_dim
        :param a2b: mapping from atom index to incoming bond indices.
        :param a2a: mapping from atom index to its neighbors. num_atoms * max_num_bonds
        :param b2a: mapping from bond index to the index of the atom the bond is coming from.
        :param b2revb: mapping from bond index to the index of the reverse bond.
        :return: batch = (f_atoms, f_bonds, a2b, a2a, b2a, b2revb)
    """
    # Per atom features: (atomic_number, pos_x, pos_y, pos_z)
    f_atoms = torch.stack((data.atomic_numbers.long(), data.pos[:, 0], data.pos[:, 1], data.pos[:, 2]), 1)
    # Per edge features (calculated by atomic distances in model forward pass)
    f_bonds = data.edge_attr

    a2a = [[] for j in range(data.natoms)]  # List of lists - Dynamically append neighbors for a given atom
    a2b = [[] for j in range(data.natoms)]  # List of lists - Dynamically append edges for a given atom
    b2a = torch.zeros((data.edge_index.shape[1],))  # (num_edges, ) - One originating atom per edge
    b2revb = torch.zeros((data.edge_index.shape[1],))  # (num_edges, ) - One reverse bond per bond
    rev_edges = {}  # Dict of lists for each (from_atom, to_atom) pair, saving edge numbers

    for i in range(data.edge_index.shape[1]):
        from_atom = int(data.edge_index[0][i])
        to_atom = int(data.edge_index[1][i])

        a2a[from_atom].append(to_atom)  # Mark b as neighbor of a
        a2b[from_atom].append(i)  # Mark bond i as outgoing bond from atom a
        b2a[i] = from_atom  # Mark a as atom where bond i is originating
        key = frozenset({to_atom, from_atom})
        if (key not in rev_edges):  # If the edge from these two atoms has not been seen yet
            rev_edges[key] = []  # Declare it as a list (so we can keep track of the edge numbers)
        rev_edges[key].append(i)  # Append the edge number to the list

    # Iterate through and set b2revb
    for atoms, edges in rev_edges.items():
        b2revb[edges[0]] = edges[1]
        b2revb[edges[1]] = edges[0]

    # Convert list of lists for a2a and a2b into tensor: (num_nodes, max_edges)
    # Option 1: Trims length to max number of edges seen in the data (<= 50)
    a2a_pad = len(max(a2a, key=len))
    a2b_pad = len(max(a2b, key=len))

    # Option 2: Sets length to max number of possible edges (should be 50 but in test ipynb it's 55)
    # a2a_pad = 50
    # a2b_pad = 50

    # -1 is not a valid atom or edge index so we pad with this
    a2a = torch.tensor([i + [-1] * (a2a_pad - len(i)) for i in a2a])
    a2b = torch.tensor([i + [-1] * (a2b_pad - len(i)) for i in a2b])

    batch = (f_atoms, f_bonds, a2b, a2a, b2a, b2revb)
    return batch

## FROM GROVER
class SelfAttention(nn.Module):
    """
       Self SelfAttention Layer
       Given $X\in \mathbb{R}^{n \times in_feature}$, the attention is calculated by: $a=Softmax(W_2tanh(W_1X))$, where
       $W_1 \in \mathbb{R}^{hidden \times in_feature}$, $W_2 \in \mathbb{R}^{out_feature \times hidden}$.
       The final output is: $out=aX$, which is unrelated with input $n$.
    """

    def __init__(self, *, hidden, in_feature, out_feature):
        """
        The init function.
        :param hidden: the hidden dimension, can be viewed as the number of experts.
        :param in_feature: the input feature dimension.
        :param out_feature: the output feature dimension.
        """
        super(SelfAttention, self).__init__()
        self.w1 = torch.nn.Parameter(torch.FloatTensor(hidden, in_feature))
        self.w2 = torch.nn.Parameter(torch.FloatTensor(out_feature, hidden))
        self.reset_parameters()

    def reset_parameters(self):
        """
        Use xavier_normal method to initialize parameters.
        """
        nn.init.xavier_normal_(self.w1)
        nn.init.xavier_normal_(self.w2)

    def forward(self, X):
        """
        The forward function.
        :param X: The input feature map. $X \in \mathbb{R}^{n \times in_feature}$.
        :return: The final embeddings and attention matrix.
        """
        x = torch.tanh(torch.matmul(self.w1, X.transpose(1, 0)))
        x = torch.matmul(self.w2, x)
        attn = torch.nn.functional.softmax(x, dim=-1)
        x = torch.matmul(attn, X)
        return x, attn

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product SelfAttention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        """
        :param query:
        :param key:
        :param value:
        :param mask:
        :param dropout:
        :return:
        """
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / numpy.math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    """
    The multi-head attention module. Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1, bias=False):
        """
        :param h:
        :param d_model:
        :param dropout:
        :param bias:
        """
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h  # number of heads

        self.linear_layers = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(3)])  # why 3: query, key, value
        self.output_linear = nn.Linear(d_model, d_model, bias)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        :param query:
        :param key:
        :param value:
        :param mask:
        :return:
        """
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, _ = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)

class Head(nn.Module):
    """
    One head for multi-headed attention.
    :return: (query, key, value)
    """

    def __init__(self, args, hidden_size, atom_messages=False):
        """
        Initialization.
        :param args: The argument.
        :param hidden_size: the dimension of hidden layer in Head.
        :param atom_messages: the MPNEncoder type.
        """
        super(Head, self).__init__()
        atom_fdim = hidden_size
        bond_fdim = hidden_size
        hidden_size = hidden_size
        self.atom_messages = atom_messages
        if self.atom_messages:
            init_message_dim = atom_fdim
            attached_fea_dim = bond_fdim
        else:
            init_message_dim = bond_fdim
            attached_fea_dim = atom_fdim

        # Here we use the message passing network as query, key and value.
        self.mpn_q = MPNEncoder(args=args,
                                atom_messages=atom_messages,
                                init_message_dim=init_message_dim,
                                attached_fea_fdim=attached_fea_dim,
                                hidden_size=hidden_size,
                                bias=args.bias,
                                depth=args.depth,
                                dropout=args.dropout,
                                undirected=args.undirected,
                                dense=args.dense,
                                aggregate_to_atom=False,
                                attach_fea=False,
                                input_layer="none",
                                dynamic_depth="truncnorm")
        self.mpn_k = MPNEncoder(args=args,
                                atom_messages=atom_messages,
                                init_message_dim=init_message_dim,
                                attached_fea_fdim=attached_fea_dim,
                                hidden_size=hidden_size,
                                bias=args.bias,
                                depth=args.depth,
                                dropout=args.dropout,
                                undirected=args.undirected,
                                dense=args.dense,
                                aggregate_to_atom=False,
                                attach_fea=False,
                                input_layer="none",
                                dynamic_depth="truncnorm")
        self.mpn_v = MPNEncoder(args=args,
                                atom_messages=atom_messages,
                                init_message_dim=init_message_dim,
                                attached_fea_fdim=attached_fea_dim,
                                hidden_size=hidden_size,
                                bias=args.bias,
                                depth=args.depth,
                                dropout=args.dropout,
                                undirected=args.undirected,
                                dense=args.dense,
                                aggregate_to_atom=False,
                                attach_fea=False,
                                input_layer="none",
                                dynamic_depth="truncnorm")

    def forward(self, f_atoms, f_bonds, a2b, a2a, b2a, b2revb):
        """
        The forward function.
        :param f_atoms: the atom features, num_atoms * atom_dim
        :param f_bonds: the bond features, num_bonds * bond_dim
        :param a2b: mapping from atom index to incoming bond indices.
        :param a2a: mapping from atom index to its neighbors. num_atoms * max_num_bonds
        :param b2a: mapping from bond index to the index of the atom the bond is coming from.
        :param b2revb: mapping from bond index to the index of the reverse bond.
        :return:
        """
        if self.atom_messages:
            init_messages = f_atoms
            init_attached_features = f_bonds
            a2nei = a2a
            a2attached = a2b
            b2a = b2a
            b2revb = b2revb
        else:
            init_messages = f_bonds
            init_attached_features = f_atoms
            a2nei = a2b
            a2attached = a2a
            b2a = b2a
            b2revb = b2revb

        q = self.mpn_q(init_messages=init_messages,
                       init_attached_features=init_attached_features,
                       a2nei=a2nei,
                       a2attached=a2attached,
                       b2a=b2a,
                       b2revb=b2revb)
        k = self.mpn_k(init_messages=init_messages,
                       init_attached_features=init_attached_features,
                       a2nei=a2nei,
                       a2attached=a2attached,
                       b2a=b2a,
                       b2revb=b2revb)
        v = self.mpn_v(init_messages=init_messages,
                       init_attached_features=init_attached_features,
                       a2nei=a2nei,
                       a2attached=a2attached,
                       b2a=b2a,
                       b2revb=b2revb)
        return q, k, v

class MTBlock(nn.Module):
    """
    The Multi-headed attention block.
    """

    def __init__(self,
                 args,
                 num_attn_head,
                 input_dim,
                 hidden_size,
                 activation="ReLU",
                 dropout=0.0,
                 bias=True,
                 atom_messages=False,
                 cuda=True,
                 res_connection=False):
        """
        :param args: the arguments.
        :param num_attn_head: the number of attention head.
        :param input_dim: the input dimension.
        :param hidden_size: the hidden size of the model.
        :param activation: the activation function.
        :param dropout: the dropout ratio
        :param bias: if true: all linear layer contains bias term.
        :param atom_messages: the MPNEncoder type
        :param cuda: if true, the model run with GPU.
        :param res_connection: enables the skip-connection in MTBlock.
        """
        super(MTBlock, self).__init__()
        # self.args = args
        self.atom_messages = atom_messages
        self.hidden_size = hidden_size
        self.heads = nn.ModuleList()
        self.input_dim = input_dim
        self.cuda = cuda
        self.res_connection = res_connection
        self.act_func = get_activation_function(activation)
        self.dropout_layer = nn.Dropout(p=dropout)
        # Note: elementwise_affine has to be consistent with the pre-training phase
        self.layernorm = nn.LayerNorm(self.hidden_size, elementwise_affine=True)

        self.W_i = nn.Linear(self.input_dim, self.hidden_size, bias=bias)
        self.attn = MultiHeadedAttention(h=num_attn_head,
                                         d_model=self.hidden_size,
                                         bias=bias,
                                         dropout=dropout)
        self.W_o = nn.Linear(self.hidden_size * num_attn_head, self.hidden_size, bias=bias)
        self.sublayer = SublayerConnection(self.hidden_size, dropout)
        for _ in range(num_attn_head):
            self.heads.append(Head(args, hidden_size=hidden_size, atom_messages=atom_messages))

    def forward(self, batch, features_batch=None):
        """
        :param batch: the graph batch generated by GroverCollator.
        :param features_batch: the additional features of molecules. (deprecated)
        :return:
        """
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a = batch

        if self.atom_messages:
            # Only add linear transformation in the input feature.
            if f_atoms.shape[1] != self.hidden_size:
                f_atoms = self.W_i(f_atoms)
                f_atoms = self.dropout_layer(self.layernorm(self.act_func(f_atoms)))

        else:  # bond messages
            if f_bonds.shape[1] != self.hidden_size:
                f_bonds = self.W_i(f_bonds)
                f_bonds = self.dropout_layer(self.layernorm(self.act_func(f_bonds)))

        queries = []
        keys = []
        values = []
        for head in self.heads:
            q, k, v = head(f_atoms, f_bonds, a2b, a2a, b2a, b2revb)
            queries.append(q.unsqueeze(1))
            keys.append(k.unsqueeze(1))
            values.append(v.unsqueeze(1))
        queries = torch.cat(queries, dim=1)
        keys = torch.cat(keys, dim=1)
        values = torch.cat(values, dim=1)

        x_out = self.attn(queries, keys, values)  # multi-headed attention
        x_out = x_out.view(x_out.shape[0], -1)
        x_out = self.W_o(x_out)

        x_in = None
        # support no residual connection in MTBlock.
        if self.res_connection:
            if self.atom_messages:
                x_in = f_atoms
            else:
                x_in = f_bonds

        if self.atom_messages:
            f_atoms = self.sublayer(x_in, x_out)
        else:
            f_bonds = self.sublayer(x_in, x_out)

        batch = f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a
        features_batch = features_batch
        return batch, features_batch

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        """Initialization.
        :param size: the input dimension.
        :param dropout: the dropout ratio.
        """
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size, elementwise_affine=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, outputs):
        """Apply residual connection to any sublayer with the same size."""
        # return x + self.dropout(self.norm(x))
        if inputs is None:
            return self.dropout(self.norm(outputs))
        return inputs + self.dropout(self.norm(outputs))

class GTransEncoder(nn.Module):
    def __init__(self,
                 args,
                 hidden_size,
                 edge_fdim,
                 node_fdim,
                 dropout=0.0,
                 activation="ReLU",
                 num_mt_block=1,
                 num_attn_head=4,
                 atom_emb_output: Union[bool, str] = False,  # options: True, False, None, "atom", "bond", "both"
                 bias=False,
                 cuda=True,
                 res_connection=False):
        """
        :param args: the arguments.
        :param hidden_size: the hidden size of the model.
        :param edge_fdim: the dimension of additional feature for edge/bond.
        :param node_fdim: the dimension of additional feature for node/atom.
        :param dropout: the dropout ratio
        :param activation: the activation function
        :param num_mt_block: the number of mt block.
        :param num_attn_head: the number of attention head.
        :param atom_emb_output:  enable the output aggregation after message passing.
                                              atom_messages:      True                      False
        -False: no aggregating to atom. output size:     (num_atoms, hidden_size)    (num_bonds, hidden_size)
        -True:  aggregating to atom.    output size:     (num_atoms, hidden_size)    (num_atoms, hidden_size)
        -None:                         same as False
        -"atom":                       same as True
        -"bond": aggragating to bond.   output size:     (num_bonds, hidden_size)    (num_bonds, hidden_size)
        -"both": aggregating to atom&bond. output size:  (num_atoms, hidden_size)    (num_bonds, hidden_size)
                                                         (num_bonds, hidden_size)    (num_atoms, hidden_size)
        :param bias: enable bias term in all linear layers.
        :param cuda: run with cuda.
        :param res_connection: enables the skip-connection in MTBlock.
        """
        super(GTransEncoder, self).__init__()

        # For the compatibility issue.
        if atom_emb_output is False:
            atom_emb_output = None
        if atom_emb_output is True:
            atom_emb_output = 'atom'

        self.hidden_size = hidden_size
        self.dropout = dropout
        self.activation = activation
        self.cuda = cuda
        self.bias = bias
        self.res_connection = res_connection
        self.edge_blocks = nn.ModuleList()
        self.node_blocks = nn.ModuleList()

        edge_input_dim = edge_fdim
        node_input_dim = node_fdim
        edge_input_dim_i = edge_input_dim
        node_input_dim_i = node_input_dim

        for i in range(num_mt_block):
            if i != 0:
                edge_input_dim_i = self.hidden_size
                node_input_dim_i = self.hidden_size
            self.edge_blocks.append(MTBlock(args=args,
                                            num_attn_head=num_attn_head,
                                            input_dim=edge_input_dim_i,
                                            hidden_size=self.hidden_size,
                                            activation=activation,
                                            dropout=dropout,
                                            bias=self.bias,
                                            atom_messages=False,
                                            cuda=cuda))
            self.node_blocks.append(MTBlock(args=args,
                                            num_attn_head=num_attn_head,
                                            input_dim=node_input_dim_i,
                                            hidden_size=self.hidden_size,
                                            activation=activation,
                                            dropout=dropout,
                                            bias=self.bias,
                                            atom_messages=True,
                                            cuda=cuda))

        self.atom_emb_output = atom_emb_output

        self.ffn_atom_from_atom = PositionwiseFeedForward(self.hidden_size + node_fdim,
                                                          self.hidden_size * 4,
                                                          activation=self.activation,
                                                          dropout=self.dropout,
                                                          d_out=self.hidden_size)

        self.ffn_atom_from_bond = PositionwiseFeedForward(self.hidden_size + node_fdim,
                                                          self.hidden_size * 4,
                                                          activation=self.activation,
                                                          dropout=self.dropout,
                                                          d_out=self.hidden_size)

        self.ffn_bond_from_atom = PositionwiseFeedForward(self.hidden_size + edge_fdim,
                                                          self.hidden_size * 4,
                                                          activation=self.activation,
                                                          dropout=self.dropout,
                                                          d_out=self.hidden_size)

        self.ffn_bond_from_bond = PositionwiseFeedForward(self.hidden_size + edge_fdim,
                                                          self.hidden_size * 4,
                                                          activation=self.activation,
                                                          dropout=self.dropout,
                                                          d_out=self.hidden_size)

        self.atom_from_atom_sublayer = SublayerConnection(size=self.hidden_size, dropout=self.dropout)
        self.atom_from_bond_sublayer = SublayerConnection(size=self.hidden_size, dropout=self.dropout)
        self.bond_from_atom_sublayer = SublayerConnection(size=self.hidden_size, dropout=self.dropout)
        self.bond_from_bond_sublayer = SublayerConnection(size=self.hidden_size, dropout=self.dropout)

        self.act_func_node = get_activation_function(self.activation)
        self.act_func_edge = get_activation_function(self.activation)

        self.dropout_layer = nn.Dropout(p=args.dropout)

    def pointwise_feed_forward_to_atom_embedding(self, emb_output, atom_fea, index, ffn_layer):
        """
        The point-wise feed forward and long-range residual connection for atom view.
        aggregate to atom.
        :param emb_output: the output embedding from the previous multi-head attentions.
        :param atom_fea: the atom/node feature embedding.
        :param index: the index of neighborhood relations.
        :param ffn_layer: the feed forward layer
        :return:
        """
        aggr_output = select_neighbor_and_aggregate(emb_output, index)
        aggr_outputx = torch.cat([atom_fea, aggr_output], dim=1)
        return ffn_layer(aggr_outputx), aggr_output

    def pointwise_feed_forward_to_bond_embedding(self, emb_output, bond_fea, a2nei, b2revb, ffn_layer):
        """
        The point-wise feed forward and long-range residual connection for bond view.
        aggregate to bond.
        :param emb_output: the output embedding from the previous multi-head attentions.
        :param bond_fea: the bond/edge feature embedding.
        :param index: the index of neighborhood relations.
        :param ffn_layer: the feed forward layer
        :return:
        """
        aggr_output = select_neighbor_and_aggregate(emb_output, a2nei)
        # remove rev bond / atom --- need for bond view
        aggr_output = self.remove_rev_bond_message(emb_output, aggr_output, b2revb)
        aggr_outputx = torch.cat([bond_fea, aggr_output], dim=1)
        return ffn_layer(aggr_outputx), aggr_output

    @staticmethod
    def remove_rev_bond_message(orginal_message, aggr_message, b2revb):
        """
        :param orginal_message:
        :param aggr_message:
        :param b2revb:
        :return:
        """
        rev_message = orginal_message[b2revb]
        return aggr_message - rev_message

    def atom_bond_transform(self,
                            to_atom=True,  # False: to bond
                            atomwise_input=None,
                            bondwise_input=None,
                            original_f_atoms=None,
                            original_f_bonds=None,
                            a2a=None,
                            a2b=None,
                            b2a=None,
                            b2revb=None
                            ):
        """
        Transfer the output of atom/bond multi-head attention to the final atom/bond output.
        :param to_atom: if true, the output is atom emebedding, otherwise, the output is bond embedding.
        :param atomwise_input: the input embedding of atom/node.
        :param bondwise_input: the input embedding of bond/edge.
        :param original_f_atoms: the initial atom features.
        :param original_f_bonds: the initial bond features.
        :param a2a: mapping from atom index to its neighbors. num_atoms * max_num_bonds
        :param a2b: mapping from atom index to incoming bond indices.
        :param b2a: mapping from bond index to the index of the atom the bond is coming from.
        :param b2revb: mapping from bond index to the index of the reverse bond.
        :return:
        """

        if to_atom:
            # atom input to atom output
            atomwise_input, _ = self.pointwise_feed_forward_to_atom_embedding(atomwise_input, original_f_atoms, a2a,
                                                                              self.ffn_atom_from_atom)
            atom_in_atom_out = self.atom_from_atom_sublayer(None, atomwise_input)
            # bond to atom
            bondwise_input, _ = self.pointwise_feed_forward_to_atom_embedding(bondwise_input, original_f_atoms, a2b,
                                                                              self.ffn_atom_from_bond)
            bond_in_atom_out = self.atom_from_bond_sublayer(None, bondwise_input)
            return atom_in_atom_out, bond_in_atom_out
        else:  # to bond embeddings

            # atom input to bond output
            atom_list_for_bond = torch.cat([b2a.unsqueeze(dim=1), a2a[b2a]], dim=1)
            atomwise_input, _ = self.pointwise_feed_forward_to_bond_embedding(atomwise_input, original_f_bonds,
                                                                              atom_list_for_bond,
                                                                              b2a[b2revb], self.ffn_bond_from_atom)
            atom_in_bond_out = self.bond_from_atom_sublayer(None, atomwise_input)
            # bond input to bond output
            bond_list_for_bond = a2b[b2a]
            bondwise_input, _ = self.pointwise_feed_forward_to_bond_embedding(bondwise_input, original_f_bonds,
                                                                              bond_list_for_bond,
                                                                              b2revb, self.ffn_bond_from_bond)
            bond_in_bond_out = self.bond_from_bond_sublayer(None, bondwise_input)
            return atom_in_bond_out, bond_in_bond_out

    def forward(self, batch, features_batch = None):
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a = batch
        if self.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.cuda(), f_bonds.cuda(), a2b.cuda(), b2a.cuda(), b2revb.cuda()
            a2a = a2a.cuda()

        node_batch = f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a
        edge_batch = f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a

        # opt pointwise_feed_forward
        original_f_atoms, original_f_bonds = f_atoms, f_bonds

        # Note: features_batch is not used here.
        for nb in self.node_blocks:  # atom messages. Multi-headed attention
            node_batch, features_batch = nb(node_batch, features_batch)
        for eb in self.edge_blocks:  # bond messages. Multi-headed attention
            edge_batch, features_batch = eb(edge_batch, features_batch)

        atom_output, _, _, _, _, _, _, _ = node_batch  # atom hidden states
        _, bond_output, _, _, _, _, _, _ = edge_batch  # bond hidden states

        if self.atom_emb_output is None:
            # output the embedding from multi-head attention directly.
            return atom_output, bond_output

        if self.atom_emb_output == 'atom':
            return self.atom_bond_transform(to_atom=True,  # False: to bond
                                            atomwise_input=atom_output,
                                            bondwise_input=bond_output,
                                            original_f_atoms=original_f_atoms,
                                            original_f_bonds=original_f_bonds,
                                            a2a=a2a,
                                            a2b=a2b,
                                            b2a=b2a,
                                            b2revb=b2revb)
        elif self.atom_emb_output == 'bond':
            return self.atom_bond_transform(to_atom=False,  # False: to bond
                                            atomwise_input=atom_output,
                                            bondwise_input=bond_output,
                                            original_f_atoms=original_f_atoms,
                                            original_f_bonds=original_f_bonds,
                                            a2a=a2a,
                                            a2b=a2b,
                                            b2a=b2a,
                                            b2revb=b2revb)
        else:  # 'both'
            atom_embeddings = self.atom_bond_transform(to_atom=True,  # False: to bond
                                                       atomwise_input=atom_output,
                                                       bondwise_input=bond_output,
                                                       original_f_atoms=original_f_atoms,
                                                       original_f_bonds=original_f_bonds,
                                                       a2a=a2a,
                                                       a2b=a2b,
                                                       b2a=b2a,
                                                       b2revb=b2revb)

            bond_embeddings = self.atom_bond_transform(to_atom=False,  # False: to bond
                                                       atomwise_input=atom_output,
                                                       bondwise_input=bond_output,
                                                       original_f_atoms=original_f_atoms,
                                                       original_f_bonds=original_f_bonds,
                                                       a2a=a2a,
                                                       a2b=a2b,
                                                       b2a=b2a,
                                                       b2revb=b2revb)
            # Notice: need to be consistent with output format of DualMPNN encoder
            return ((atom_embeddings[0], bond_embeddings[0]),
                    (atom_embeddings[1], bond_embeddings[1]))

# Taken from GROVER nn_utils
def get_activation_function(activation: str) -> nn.Module:
    """
    Gets an activation function module given the name of the activation.

    :param activation: The name of the activation function.
    :return: The activation function module.
    """
    if activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU(0.1)
    elif activation == 'PReLU':
        return nn.PReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'SELU':
        return nn.SELU()
    elif activation == 'ELU':
        return nn.ELU()
    elif activation == "Linear":
        return lambda x: x
    else:
        raise ValueError(f'Activation "{activation}" not supported.')

def index_select_nd(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Selects the message features from source corresponding to the atom or bond indices in index.

    :param source: A tensor of shape (num_bonds, hidden_size) containing message features.
    :param index: A tensor of shape (num_atoms/num_bonds, max_num_bonds) containing the atom or bond
    indices to select from source.
    :return: A tensor of shape (num_atoms/num_bonds, max_num_bonds, hidden_size) containing the message
    features corresponding to the atoms/bonds specified in index.
    """
    index_size = index.size()  # (num_atoms/num_bonds, max_num_bonds)
    suffix_dim = source.size()[1:]  # (hidden_size,)
    final_size = index_size + suffix_dim  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    target = source.index_select(dim=0, index=index.view(-1))  # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = target.view(final_size)  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    return target

def select_neighbor_and_aggregate(feature, index):
    """
    The basic operation in message passing.
    Caution: the index_selec_ND would cause the reproducibility issue when performing the training on CUDA.
    See: https://pytorch.org/docs/stable/notes/randomness.html
    :param feature: the candidate feature for aggregate. (n_nodes, hidden)
    :param index: the selected index (neighbor indexes).
    :return:
    """
    neighbor = index_select_nd(feature, index)
    return neighbor.sum(dim=1)

# From GROVER layers.py
class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, d_ff, activation="PReLU", dropout=0.1, d_out=None):
        """Initialization.

        :param d_model: the input dimension.
        :param d_ff: the hidden dimension.
        :param activation: the activation function.
        :param dropout: the dropout rate.
        :param d_out: the output dimension, the default value is equal to d_model.
        """
        super(PositionwiseFeedForward, self).__init__()
        if d_out is None:
            d_out = d_model
        # By default, bias is on.
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)
        self.act_func = get_activation_function(activation)

    def forward(self, x):
        """
        The forward function
        :param x: input tensor.
        :return:
        """
        return self.W_2(self.dropout(self.act_func(self.W_1(x))))

class MPNEncoder(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self, args: Namespace,
                 atom_messages: bool,
                 init_message_dim: int,
                 attached_fea_fdim: int,
                 hidden_size: int,
                 bias: bool,
                 depth: int,
                 dropout: float,
                 undirected: bool,
                 dense: bool,
                 aggregate_to_atom: bool,
                 attach_fea: bool,
                 input_layer="fc",
                 dynamic_depth='none'
                 ):
        """
        Initializes the MPNEncoder.
        :param args: the arguments.
        :param atom_messages: enables atom_messages or not.
        :param init_message_dim:  the initial input message dimension.
        :param attached_fea_fdim:  the attached feature dimension.
        :param hidden_size: the output message dimension during message passing.
        :param bias: the bias in the message passing.
        :param depth: the message passing depth.
        :param dropout: the dropout rate.
        :param undirected: the message passing is undirected or not.
        :param dense: enables the dense connections.
        :param attach_fea: enables the feature attachment during the message passing process.
        :param dynamic_depth: enables the dynamic depth. Possible choices: "none", "uniform" and "truncnorm"
        """
        super(MPNEncoder, self).__init__()
        self.init_message_dim = init_message_dim
        self.attached_fea_fdim = attached_fea_fdim
        self.hidden_size = hidden_size
        self.bias = bias
        self.depth = depth
        self.dropout = dropout
        self.input_layer = input_layer
        self.layers_per_message = 1
        self.undirected = undirected
        self.atom_messages = atom_messages
        self.dense = dense
        self.aggreate_to_atom = aggregate_to_atom
        self.attached_fea = attach_fea
        self.dynamic_depth = dynamic_depth

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Input
        if self.input_layer == "fc":
            input_dim = self.init_message_dim
            self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        if self.attached_fea:
            w_h_input_size = self.hidden_size + self.attached_fea_fdim
        else:
            w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)

    def forward(self,
                init_messages,
                init_attached_features,
                a2nei,
                a2attached,
                b2a=None,
                b2revb=None,
                adjs=None
                ) -> torch.FloatTensor:
        """
        The forward function.
        :param init_messages:  initial massages, can be atom features or bond features.
        :param init_attached_features: initial attached_features.
        :param a2nei: the relation of item to its neighbors. For the atom message passing, a2nei = a2a. For bond
        messages a2nei = a2b
        :param a2attached: the relation of item to the attached features during message passing. For the atom message
        passing, a2attached = a2b. For the bond message passing a2attached = a2a
        :param b2a: remove the reversed bond in bond message passing
        :param b2revb: remove the revered atom in bond message passing
        :return: if aggreate_to_atom or self.atom_messages, return num_atoms x hidden.
        Otherwise, return num_bonds x hidden
        """

        # Input
        if self.input_layer == 'fc':
            input = self.W_i(init_messages)  # num_bonds x hidden_size # f_bond
            message = self.act_func(input)  # num_bonds x hidden_size
        elif self.input_layer == 'none':
            input = init_messages
            message = input

        attached_fea = init_attached_features  # f_atom / f_bond

        # dynamic depth
        # uniform sampling from depth - 1 to depth + 1
        # only works in training.
        if self.training and self.dynamic_depth != "none":
            if self.dynamic_depth == "uniform":
                # uniform sampling
                ndepth = numpy.random.randint(self.depth - 3, self.depth + 3)
            else:
                # truncnorm
                mu = self.depth
                sigma = 1
                lower = mu - 3 * sigma
                upper = mu + 3 * sigma
                X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
                ndepth = int(X.rvs(1))
        else:
            ndepth = self.depth

        # Message passing
        for _ in range(ndepth - 1):
            if self.undirected:
                # two directions should be the same
                message = (message + message[b2revb]) / 2

            nei_message = select_neighbor_and_aggregate(message, a2nei)
            a_message = nei_message
            if self.attached_fea:
                attached_nei_fea = select_neighbor_and_aggregate(attached_fea, a2attached)
                a_message = torch.cat((nei_message, attached_nei_fea), dim=1)

            if not self.atom_messages:
                rev_message = message[b2revb]
                if self.attached_fea:
                    atom_rev_message = attached_fea[b2a[b2revb]]
                    rev_message = torch.cat((rev_message, atom_rev_message), dim=1)
                # Except reverse bond its-self(w) ! \sum_{k\in N(u) \ w}
                message = a_message[b2a] - rev_message  # num_bonds x hidden
            else:
                message = a_message

            message = self.W_h(message)

            # BUG here, by default MPNEncoder use the dense connection in the message passing step.
            # The correct form should if not self.dense
            if self.dense:
                message = self.act_func(message)  # num_bonds x hidden_size
            else:
                message = self.act_func(input + message)
            message = self.dropout_layer(message)  # num_bonds x hidden

        output = message

        return output  # num_atoms x hidden