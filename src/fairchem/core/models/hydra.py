from __future__ import annotations

import torch

from fairchem.core.common.registry import registry
from fairchem.core.models.base import BaseModel


@registry.register_model("base_hydra")
class BaseHydra(BaseModel):
    def __init__(
        self,
        num_atoms: int,  # not used
        bond_feat_dim: int,  # not used
        num_targets: int,  # not used
        backbone: dict,
        heads: dict,
        otf_graph: bool = True,
    ):
        super().__init__()
        self.otf_graph = otf_graph

        backbone_model_name = backbone.pop("model")
        self.backbone = registry.get_model_class(backbone_model_name)(
            num_atoms,
            bond_feat_dim,
            num_targets,
            **backbone,
        )

        # Iterate through outputs_cfg and create heads
        self.output_heads = {}
        self.second_order_output_heads = {}

        head_names_sorted = sorted(heads.keys())
        for head_name in head_names_sorted:
            head_config = heads[head_name]
            if "module" not in head_config:
                raise ValueError(
                    f"{head_name} head does not specify module to use for the head"
                )

            module_name = head_config.pop("module")
            self.output_heads[head_name] = registry.get_model_class(module_name)(
                self.backbone,
                backbone,
                head_config,
            )  # .to(self.backbone.device)

        self.output_heads = torch.nn.ModuleDict(self.output_heads)

    def forward(self, x):
        emb = self.backbone(x)
        # Predict all output properties for all structures in the batch for now.
        out = {}
        for k in self.output_heads:
            out.update(self.output_heads[k](x, emb))

        return out
