from dig.threedgraph.method import SphereNet as DIGSphereNet
from ocpmodels.models.base_model import BaseModel
import torch
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad
from copy import deepcopy


@registry.register_model("spherenet")
class SphereNet(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.energy_and_force = kwargs.get("energy_and_force", False)
        self.regress_forces = "from_energy" if self.energy_and_force else False
        self.cutoff = kwargs.get("cutoff", 5.0)
        self.num_layers = kwargs.get("num_layers", 4)
        self.hidden_channels = kwargs.get("hidden_channels", 128)
        self.out_channels = kwargs.get("out_channels", 1)
        self.int_emb_size = kwargs.get("int_emb_size", 64)
        self.basis_emb_size_dist = kwargs.get("basis_emb_size_dist", 8)
        self.basis_emb_size_angle = kwargs.get("basis_emb_size_angle", 8)
        self.basis_emb_size_torsion = kwargs.get("basis_emb_size_torsion", 8)
        self.out_emb_channels = kwargs.get("out_emb_channels", 256)
        self.num_spherical = kwargs.get("num_spherical", 3)
        self.num_radial = kwargs.get("num_radial", 6)
        self.envelope_exponent = kwargs.get("envelope_exponent", 5)
        self.num_before_skip = kwargs.get("num_before_skip", 1)
        self.num_after_skip = kwargs.get("num_after_skip", 2)
        self.num_output_layers = kwargs.get("num_output_layers", 3)
        self.spherenet = DIGSphereNet(
            energy_and_force=self.energy_and_force,
            cutoff=self.cutoff,
            num_layers=self.num_layers,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            int_emb_size=self.int_emb_size,
            basis_emb_size_dist=self.basis_emb_size_dist,
            basis_emb_size_angle=self.basis_emb_size_angle,
            basis_emb_size_torsion=self.basis_emb_size_torsion,
            out_emb_channels=self.out_emb_channels,
            num_spherical=self.num_spherical,
            num_radial=self.num_radial,
            envelope_exponent=self.envelope_exponent,
            num_before_skip=self.num_before_skip,
            num_after_skip=self.num_after_skip,
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

        return {"energy": self.spherenet.forward(batch_data)}
