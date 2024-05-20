import torch
import torch.nn as nn
import torch.nn.functional as F

def make_mlp(
    input_dim: int = 512,
    hidden_dim: int = 1024,
    output_dim: int = 512,
    dropout: float = 0., 
):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Dropout(dropout),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )