import math
import torch
import torch.nn as nn

from typing import Optional

class ResMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0., 
        init_gain: float = 1.,
    ):
        super().__init__()
        assert num_layers >= 2

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()

        self.linears = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) 
            for _ in range(num_layers - 2)
        ])

        self.input = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.reset_parameters(init_gain)

    def reset_parameters(self, init_gain=1.):
        for linear in [self.input, self.output, *self.linears]:
            nn.init.uniform_(
                linear.weight,
                - init_gain * math.sqrt(6 / linear.weight.size(1)),
                init_gain * math.sqrt(6 / linear.weight.size(1))
            )
            nn.init.zeros_(linear.bias)

    def forward(self, x: torch.Tensor, gate: Optional[torch.Tensor] = None):

        x = self.input(x)
        x = self.activation(x)
        x = self.dropout(x)

        for linear in self.linears:
            z = linear(x)
            z = self.activation(z)
            z = self.dropout(z)
            x = x + z

        if gate is not None:
            x = x * gate

        x = self.output(x)

        return x