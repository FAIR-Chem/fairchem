import math 
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.models.schnet import GaussianSmearing

from .utils import ResMLP

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

        self.rbf_encoder = ResMLP(
            input_dim=num_gaussians,
            hidden_dim=ff_dim,
            output_dim=embed_dim,
            dropout=dropout,
        )

        self.direction_encoder = ResMLP(
            input_dim=3,
            hidden_dim=ff_dim,
            output_dim=embed_dim,
            dropout=dropout,
        )

        self.type_embedding = nn.Parameter(
            torch.randn((2, 1, embed_dim))
        )

    def forward(
        self,
        batch: torch.Tensor,
        pos: torch.Tensor, 
        natoms: torch.Tensor,
        atomic_numbers: torch.Tensor, 
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # common constants and reshaping
        batch_size = len(natoms)
        natoms = natoms[:, None]

        # compute number of edges per batch
        batch_idx, counts = torch.unique_consecutive(batch[edge_index[0]], return_counts = True)
        edge_num = torch.zeros(batch.max() + 1, dtype=torch.long, device=edge_index.device)
        edge_num[batch_idx] = counts
        edge_num = edge_num[:, None]

        # tokenizing the sequence
        nmax = (natoms + edge_num).max()
        token_pos = torch.arange(nmax, device=natoms.device)[None, :].repeat(batch_size, 1)
        padded_node_mask = token_pos < natoms
        padded_edge_mask = (token_pos >= natoms) & (token_pos < edge_num + natoms)
        padded_mask = padded_node_mask | padded_edge_mask
        # initialize padded features and index
        padded_features = torch.zeros((batch_size, nmax, self.embed_dim), device=pos.device)
        padded_index = torch.zeros((batch_size, nmax, 2), device=pos.device, dtype=torch.long)
        padded_index[padded_node_mask] = torch.arange(natoms.sum(), device=natoms.device)[:, None]
        padded_index[padded_edge_mask] = edge_index.T

        # encode nodes (atomic numbers)
        atom_emb = self.anum_encoder(atomic_numbers)
        padded_features[padded_node_mask] = (atom_emb + self.type_embedding[0])

        # encode edges (distances and displacements)
        vec = pos[edge_index[1]] - pos[edge_index[0]]
        vec_hat = F.normalize(vec, dim = -1)
        dist = torch.linalg.norm(vec, dim=-1)
        rbf = self.gaussian_smearing(dist)
        padded_features[padded_edge_mask] = (
            self.rbf_encoder(rbf) +
            self.direction_encoder(vec_hat) +  
            self.type_embedding[1]
        )

        # ensure univariant
        padded_features = padded_features / math.sqrt(3)
        
        return padded_features, padded_mask, padded_node_mask, padded_edge_mask, padded_index