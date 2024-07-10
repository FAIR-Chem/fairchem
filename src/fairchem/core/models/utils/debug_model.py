import torch
import torch.nn as nn

from fairchem.core.common.registry import registry
from fairchem.core.models.base import BaseModel

from fairchem.core.common.utils import conditional_grad


@registry.register_model("debug")
class Debug(BaseModel):

    def __init__(
        self,
        num_atoms: int,  # not used
        bond_feat_dim: int,  # not used
        num_targets: int,  # not used
        otf_graph: bool,
    ) -> None:
        super().__init__()

        self.regress_forces = True
        self.otf_graph = otf_graph

        self.forces_linear = nn.Linear(3, 3)
        self.energy_coef = nn.Parameter(torch.tensor(1.0))

    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        outputs = {
            "forces": self.forces_linear(data["pos"].float()),
            "energy": self.energy_coef * data["energy"].float(),
        }
        return outputs
