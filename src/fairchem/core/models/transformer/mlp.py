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
        self.reset_parameters()

    def reset_parameters(self):
        # initialize modules using Kaiming initialization
        # taking variance accumulation from residual into account
        nn.init.uniform_(
                self.input.weight,
                - math.sqrt(6 / self.input.weight.size(1)),
                math.sqrt(6 / self.input.weight.size(1))
            )
        nn.init.zeros_(self.input.bias)

        for i, linear in enumerate(self.linears):
            nn.init.uniform_(
                linear.weight,
                - math.sqrt(6 / ((i+1) * linear.weight.size(1))),
                math.sqrt(6 / ((i+1) * linear.weight.size(1)))
            )
            nn.init.zeros_(linear.bias)

        nn.init.uniform_(
            self.output.weight,
            - math.sqrt(3 / ((len(self.linears) + 1) * self.output.weight.size(1))),
            math.sqrt(3 / ((len(self.linears) + 1) * self.output.weight.size(1)))
        )
        nn.init.zeros_(self.output.bias)

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