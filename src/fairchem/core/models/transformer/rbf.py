import math
import torch
import torch.nn as nn

class GaussianSmearing(nn.Module):
    def __init__(
        self,
        rbf_radius: float = 1,
        num_gaussians: int = 50,
        std: float = 0.02,
    ):
        super().__init__()

        self.register_buffer("offset", torch.linspace(0, rbf_radius, num_gaussians))
        self.register_buffer("inv_var", torch.tensor(1 / (std ** 2)))

    def forward(self, x: torch.Tensor):
        # compute rbfs and embeddings
        shape = x.shape
        x = x.view(-1, 1) - self.offset.view(1, -1)
        x = - 0.5 * x.square() * self.inv_var
        x = torch.exp(x)
        x = x.view(*shape, -1)
        return x

class RadialBasisFunction(nn.Module):
    def __init__(
        self,
        rbf_radius: float = 1,
        num_gaussians: int = 50,
        embed_dim: int = 128,
    ):
        super().__init__()

        self.smearing = GaussianSmearing(
            rbf_radius=rbf_radius,
            num_gaussians=num_gaussians,
            std=rbf_radius/num_gaussians,
        )
        
        self.linear = nn.Linear(num_gaussians, embed_dim, bias=False)
        self.reset_parameters()
    
    def reset_parameters(self):
        # initialize such that the Linear layer outputs unit variance
        # Gaussaian smearing results in a sum of 2.50663 (eliptic function)
        nn.init.uniform_(
            self.linear.weight,
            - math.sqrt(3 / 2.50663),
            math.sqrt(3 / 2.50663)
        )

    def forward(self, x: torch.Tensor):
        rbf = self.smearing(x)
        return self.linear(rbf), rbf