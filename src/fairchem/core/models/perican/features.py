import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureBuilder(nn.Module):
    def __init__(
        self
    ):
        super().__init__()

    def forward(
        self,
        pos,
        ntaoms,
        atomic_number,
    ):
        # common constants and reshaping
        batch_size = len(natoms)
        natoms = natoms[:, None]

        # tokenizing the sequence
        nmax = natoms.max()
        token_pos = torch.arange(nmax, device=natoms.device)[None, :].repeat(batch_size, 1)
        padded_mask = token_pos < natoms