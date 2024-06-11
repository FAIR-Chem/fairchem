import math
import torch
import torch.nn as nn

class GaussianSmearing(nn.Module):
    def __init__(
        self,
        rbf_radius: float = 1,
        num_gaussians: int = 50,
    ):
        super().__init__()

        self.register_buffer("offset", torch.linspace(0, rbf_radius, num_gaussians))
        self.register_buffer("inv_var", torch.tensor((num_gaussians / rbf_radius)**2))

    def forward(self, x: torch.Tensor):
        # compute rbfs and embeddings
        shape = x.shape
        x = x.view(-1, 1) - self.offset.view(1, -1)
        x = - 0.5 * x.square() * self.inv_var
        x = torch.exp(x)
        x = x.view(*shape, -1)
        return x