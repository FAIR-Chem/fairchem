"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch

from .base_layers import Dense
from ocpmodels.modules.phys_embeddings import PhysEmbedding


class AtomEmbedding(torch.nn.Module):
    """
    Initial atom embeddings based on the atom type

    Arguments
    ---------
    emb_size: int
        Atom embeddings size
    """

    def __init__(
        self,
        emb_size,
        num_elements,
        tag_hidden_channels=0,
        pg_hidden_channels=0,
        phys_hidden_channels=0,
        phys_embeds=False,
    ):
        super().__init__()
        self.emb_size = emb_size

        self.use_tag = tag_hidden_channels > 0
        self.use_pg = pg_hidden_channels > 0
        self.use_mlp_phys = phys_hidden_channels > 0 and phys_embeds

        # Phys embeddings
        self.phys_emb = PhysEmbedding(
            props=phys_embeds, props_grad=phys_hidden_channels > 0, pg=self.use_pg
        )
        # With MLP
        if self.use_mlp_phys:
            self.phys_lin = torch.nn.Linear(
                self.phys_emb.n_properties, phys_hidden_channels
            )
        else:
            phys_hidden_channels = self.phys_emb.n_properties

        # Period + group embeddings
        if self.use_pg:
            self.period_embedding = torch.nn.Embedding(
                self.phys_emb.period_size, pg_hidden_channels
            )
            self.group_embedding = torch.nn.Embedding(
                self.phys_emb.group_size, pg_hidden_channels
            )

        # Tag embedding
        if tag_hidden_channels:
            self.tag_embedding = torch.nn.Embedding(3, tag_hidden_channels)

        self.base_embedding = torch.nn.Embedding(
            num_elements + 1,  # account for z \in [1:..]
            emb_size
            - tag_hidden_channels
            - phys_hidden_channels
            - 2 * pg_hidden_channels,
        )

        # init by uniform distribution

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.base_embedding.weight, a=-np.sqrt(3), b=np.sqrt(3))
        if self.use_mlp_phys:
            torch.nn.init.xavier_uniform_(self.phys_lin.weight)
        if self.use_tag:
            self.tag_embedding.reset_parameters()
        if self.use_pg:
            self.period_embedding.reset_parameters()
            self.group_embedding.reset_parameters()

    def forward(self, z, tag=None):
        """
        Returns
        -------
        h: torch.Tensor, shape=(nAtoms, emb_size)
            Atom embeddings.
        """
        # Create atom embeddings based on its characteristic number
        h = self.base_embedding(z)

        if self.phys_emb.device != h.device:
            self.phys_emb = self.phys_emb.to(h.device)

        # Concat tag embedding
        if self.use_tag:
            h_tag = self.tag_embedding(tag)
            h = torch.cat((h, h_tag), dim=1)

        # Concat physics embeddings
        if self.phys_emb.n_properties > 0:
            h_phys = self.phys_emb.properties[z]
            if self.use_mlp_phys:
                h_phys = self.phys_lin(h_phys)
            h = torch.cat((h, h_phys), dim=1)

        # Concat period & group embedding
        if self.use_pg:
            h_period = self.period_embedding(self.phys_emb.period[z])
            h_group = self.group_embedding(self.phys_emb.group[z])
            h = torch.cat((h, h_period, h_group), dim=1)

        return h


class EdgeEmbedding(torch.nn.Module):
    """
    Edge embedding based on the concatenation of atom embeddings
    and a subsequent dense layer.

    Arguments
    ---------
    atom_features: int
        Embedding size of the atom embedding.
    edge_features: int
        Embedding size of the input edge embedding.
    out_features: int
        Embedding size after the dense layer.
    activation: str
        Activation function used in the dense layer.
    """

    def __init__(
        self,
        atom_features,
        edge_features,
        out_features,
        activation=None,
    ):
        super().__init__()
        in_features = 2 * atom_features + edge_features
        self.dense = Dense(in_features, out_features, activation=activation, bias=False)

    def forward(
        self,
        h,
        m,
        edge_index,
    ):
        """
        Arguments
        ---------
        h: torch.Tensor, shape (num_atoms, atom_features)
            Atom embeddings.
        m: torch.Tensor, shape (num_edges, edge_features)
            Radial basis in embedding block,
            edge embedding in interaction block.

        Returns
        -------
            m_st: torch.Tensor, shape=(nEdges, emb_size)
                Edge embeddings.
        """
        h_s = h[edge_index[0]]  # shape=(nEdges, emb_size)
        h_t = h[edge_index[1]]  # shape=(nEdges, emb_size)

        m_st = torch.cat([h_s, h_t, m], dim=-1)  # (nEdges, 2*emb_size+nFeatures)
        m_st = self.dense(m_st)  # (nEdges, emb_size)
        return m_st
