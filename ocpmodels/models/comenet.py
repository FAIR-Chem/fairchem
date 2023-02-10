from dig.threedgraph.method import ComENet as DIGComENet
from ocpmodels.models.base_model import BaseModel
import torch
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad
from copy import deepcopy


@registry.register_model("comenet")
class ComENet(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.regress_forces = False
        self.cutoff = kwargs.get("cutoff", 5.0)
        self.num_layers = kwargs.get("num_layers", 4)
        self.hidden_channels = kwargs.get("hidden_channels", 128)
        self.out_channels = kwargs.get("out_channels", 1)
        self.num_spherical = kwargs.get("num_spherical", 3)
        self.num_radial = kwargs.get("num_radial", 6)
        self.num_output_layers = kwargs.get("num_output_layers", 3)
        self.comenet = DIGComENet(
            cutoff=self.cutoff,
            num_layers=self.num_layers,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            num_spherical=self.num_spherical,
            num_radial=self.num_radial,
            num_output_layers=self.num_output_layers,
        )

    @conditional_grad(torch.enable_grad())
    def forces_forward(self, preds):
        return

    @conditional_grad(torch.enable_grad())
    def energy_forward(self, data):
        # Rewire the graph
        z = data.atomic_numbers.long()
        batch_data = deepcopy(data)
        batch_data.z = z

        return {"energy": self.comenet.forward(batch_data)}
