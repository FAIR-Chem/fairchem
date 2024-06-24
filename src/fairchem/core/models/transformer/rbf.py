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
        self.coeff = - 0.5 * (((num_gaussians - 1) / rbf_radius) ** 2)

    def forward(self, x: torch.Tensor):
        # compute gaussian smearing values
        shape = x.shape
        x = x.view(-1, 1) - self.offset.view(1, -1)
        x = x.square_().mul_(self.coeff).exp_()
        x = x.view(*shape, -1)
        return x