"""
modified from https://github.com/jw9730/tokengt
"""

import torch
import torch.nn as nn
from fairseq import utils


class FeedForward(nn.Module):
    def __init__(
            self,
            embedding_dim,
            ffn_embedding_dim,
            activation_fn,
            dropout,
    ):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, ffn_embedding_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.fc2 = nn.Linear(ffn_embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
