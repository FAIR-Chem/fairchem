import torch
from torch import nn
from torch.nn import Linear
from torch_scatter import scatter
from ocpmodels.models.faenet import FAENet, OutputBlock


class DeupOutputBlock(OutputBlock):
    def __init__(
        self, energy_head, hidden_channels, act, dropout_lin, deup_features={}
    ):
        super().__init__(energy_head, hidden_channels, act, dropout_lin)

        self.deup_features = deup_features
        self.deup_data_keys = [f"deup_{k}" for k in deup_features]
        self.deup_extra_dim = 0
        if "s" in deup_features:
            self.deup_extra_dim += 1
        if "energy_pred_std" in deup_features:
            self.deup_extra_dim += 1

        if self.deup_extra_dim > 0:
            self.deup_lin = Linear(
                hidden_channels + self.deup_extra_dim, hidden_channels
            )

    def forward(self, h, edge_index, edge_weight, batch, alpha, data=None):
        if self.energy_head == "weighted-av-final-embeds":
            alpha = self.w_lin(h)

        if self.deup_extra_dim > 0:
            assert data is not None
            data_keys = set(data.keys())
            assert all(self.deup_data_keys in data_keys)
            h = torch.cat([h] + [data[k][:, None] for k in self.deup_features], dim=-1)
            h = self.deup_lin(h)
            h = self.act(h)

        # MLP
        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        if self.energy_head in {
            "weighted-av-initial-embeds",
            "weighted-av-final-embeds",
        }:
            h = h * alpha

        # Global pooling
        out = scatter(h, batch, dim=0, reduce="add")

        return out


class DeupFAENet(FAENet):
    def __init__(self, *args, **kwargs):
        kwargs["dropout_edge"] = 0
        kwargs["dropout_edge"] = 0
        super().__init__(*args, **kwargs)
        self.output_block = DeupOutputBlock(
            self.energy_head,
            kwargs["hidden_channels"],
            self.act,
            self.dropout_lin,
            kwargs.get("deup_features", {}),
        )
