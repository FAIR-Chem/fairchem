import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RadialBasisFunction

class FeatureBuilder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_element: int = 100,
        rbf_radius: float = 12.,
        num_gaussians: int = 50,
    ):
        super().__init__()

        self.src_anum = nn.Embedding(
            num_embeddings=num_element,
            embedding_dim=embed_dim,
            padding_idx=0
        )

        self.dst_anum = nn.Embedding(
            num_embeddings=num_element,
            embedding_dim=embed_dim,
            padding_idx=0
        )

        self.rbf = RadialBasisFunction(
            rbf_radius=rbf_radius,
            num_gaussians=num_gaussians,
            embed_dim=embed_dim
        )

    def forward(
        self,
        pos,
        natoms,
        atomic_numbers,
    ):
        # common constants and reshaping
        batch_size = len(natoms)
        natoms = natoms[:, None]

        # tokenizing the sequence
        nmax = natoms.max()
        token_pos = torch.arange(nmax, device=natoms.device)[None, :].repeat(batch_size, 1)
        padded_mask = token_pos < natoms
        padded_pos = torch.zeros((batch_size, nmax, 3), device=pos.device)
        padded_pos[padded_mask] = pos
        padded_anum = torch.zeros((batch_size, nmax), device=atomic_numbers.device, dtype=torch.long)
        padded_anum[padded_mask] = atomic_numbers

        # construct matrices
        padded_pos = padded_pos.transpose(0, 1)
        padded_anum = padded_anum.transpose(0, 1)
        mask = padded_mask.transpose(0, 1)[:, None] & padded_mask.transpose(0, 1)[None, :]

        # construct vec and dist
        vec = (padded_pos[None, :] - padded_pos[:, None])
        dist = torch.linalg.norm(vec, dim=3)
        vec_hat = F.normalize(vec, dim=3)

        # construct features
        padded_features = self.rbf(dist) + self.src_anum(padded_anum[:, None]) + self.dst_anum(padded_anum[None, :])
        padded_features = padded_features / math.sqrt(3)

        return padded_features, mask, padded_mask, dist, vec_hat