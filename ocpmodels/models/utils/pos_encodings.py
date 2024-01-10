import math

import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_count: int = 210):
        super().__init__()
        position = torch.arange(max_count).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_count, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.squeeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [num_atoms, batch_size, embedding_dim]
            x: Tensor, shape [num_atoms]
        """
        return self.pe[x]
