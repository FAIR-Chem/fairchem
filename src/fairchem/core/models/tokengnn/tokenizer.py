import math 

import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy.sparse as sps

from torch_geometric.nn.models.schnet import GaussianSmearing
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix

from .utils import make_mlp

class GraphFeatureTokenizer(nn.Module):
    """
    Compute node and edge features for each node and edge in the graph.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        ff_dim: int = 1024,
        dropout: float = 0.,
        num_elements: int = 100,
        rbf_radius: float = 12.,
        num_gaussians: int = 50,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        self.anum_encoder = nn.Embedding(
            num_embeddings=num_elements,
            embedding_dim=embed_dim,
            padding_idx=0
        )
        
        self.gaussian_smearing = GaussianSmearing(
            start=0, 
            stop=rbf_radius, 
            num_gaussians=num_gaussians
        )

        self.rbf_encoder = make_mlp(
            input_dim=num_gaussians,
            hidden_dim=ff_dim,
            output_dim=embed_dim,
            dropout=dropout,
        )

        self.direction_encoder = make_mlp(
            input_dim=3,
            hidden_dim=ff_dim,
            output_dim=embed_dim,
            dropout=dropout,
        )

        self.type_embedding = nn.Parameter(
            torch.randn((2, 1, embed_dim))
        )

        self.inv_sqrt2 = 1 / math.sqrt(2)
        self.inv_sqrt3 = 1 / math.sqrt(3)

    def forward(
        self,
        pos: torch.Tensor, 
        atomic_numbers: torch.Tensor, 
        edge_index: torch.Tensor,
    ):
        # initialize tokens
        tokens = torch.zeros((pos.size(0) + edge_index.size(1), self.embed_dim), device=pos.device)

        # encode nodes (atomic numbers)
        atom_emb = self.anum_encoder(atomic_numbers)
        tokens[:pos.size(0)] = self.inv_sqrt2 * (atom_emb + self.type_embedding[0]) 

        # encode edges (distances and displacements)
        vec = pos[edge_index[1]] - pos[edge_index[0]]
        vec_hat = F.normalize(vec, dim = -1)
        dist = torch.linalg.norm(vec, dim=-1)
        rbf = self.gaussian_smearing(dist)
        tokens[pos.size(0):] = self.inv_sqrt3 * (
            self.rbf_encoder(rbf) +
            self.direction_encoder(vec_hat) +  
            self.type_embedding[1]
        )

        # construct new edges
        edge_token_pos = torch.arange(pos.size(0), pos.size(0) + edge_index.size(1), device=edge_index.device)
        n2e = torch.stack([edge_index[0], edge_token_pos], dim = 0)
        e2n = torch.stack([edge_token_pos, edge_index[1]], dim = 0)
        n2e_sparse = to_scipy_sparse_matrix(n2e).tocsr()
        e2n_sparse = to_scipy_sparse_matrix(e2n).tocsr()
        # d2s = from_scipy_sparse_matrix(e2n_sparse @ n2e_sparse)[0].to(edge_index.device)
        dnd = from_scipy_sparse_matrix(e2n_sparse @ e2n_sparse.T)[0].to(edge_index.device)
        sns = from_scipy_sparse_matrix(n2e_sparse.T @ n2e_sparse)[0].to(edge_index.device)
        # s2d = from_scipy_sparse_matrix(n2e_sparse.T @ e2n_sparse.T)[0].to(edge_index.device)

        # concatenate and create mask
        edges = torch.cat([edge_index, n2e, e2n, dnd, sns], dim=1) # snd, dns
        edge_types = torch.cat([
            torch.arange(3, device = edge_index.device).repeat_interleave(edge_index.size(1)),
            torch.tensor([3], device = edge_index.device).repeat(dnd.size(1)),
            torch.tensor([4], device = edge_index.device).repeat(sns.size(1)),
        ], dim = 0)
        masks = F.one_hot(
            edge_types,
            5,
        ).bool().T # (5, edges)
        
        return tokens, edges, masks