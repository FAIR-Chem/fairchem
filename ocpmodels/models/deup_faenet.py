import torch
from torch import nn
from torch.nn import Linear
from torch_scatter import scatter
from ocpmodels.common.registry import registry
from ocpmodels.models.faenet import FAENet, OutputBlock


class DeupOutputBlock(OutputBlock):
    def __init__(
        self, energy_head, hidden_channels, act, dropout_lin, deup_features={}
    ):
        super().__init__(energy_head, hidden_channels, act, dropout_lin)

        self.deup_features = deup_features
        self.deup_data_keys = [f"deup_{k}" for k in deup_features]
        self.deup_extra_dim = 0
        self._set_q_dim = False

        if "s" in deup_features:
            self.deup_extra_dim += 1
        if "energy_pred_std" in deup_features:
            self.deup_extra_dim += 1
        if "q" in deup_features:
            self._set_q_dim = True

        if self.deup_extra_dim > 0:
            self.deup_lin = Linear(
                self.lin1.out_features + self.deup_extra_dim, self.lin1.out_features
            )

    def forward(self, h, edge_index, edge_weight, batch, alpha, data=None):
        if self._set_q_dim:
            assert data is not None
            assert "deup_q" in data.to_dict().keys()
            self.deup_extra_dim += data.deup_q.shape[-1]
            self.deup_lin = Linear(
                self.lin1.out_features + self.deup_extra_dim, self.lin1.out_features
            )
            print("\nLazy loading deup extra dim from q. New dim:", self.deup_extra_dim)
            print("⚠️ OutputBlock will be reinitialized.\n")
            self.reset_parameters()
            self._set_q_dim = False

        if self.energy_head == "weighted-av-final-embeds":
            alpha = self.w_lin(h)

        # OutputBlock to get final atom rep
        h = self.lin1(h)
        h = self.act(h)
        if self.deup_extra_dim <= 0:
            h = self.lin2(h)

        if self.energy_head in {
            "weighted-av-initial-embeds",
            "weighted-av-final-embeds",
        }:
            h = h * alpha

        # Global pooling -- get final graph rep
        out = scatter(
            h,
            batch,
            dim=0,
            reduce="mean" if self.deup_extra_dim > 0 else "add",
        )

        # Concat graph representation with deup features (s, kde(q), std)
        # and apply MLPs
        if self.deup_extra_dim > 0:
            assert data is not None
            data_keys = set(data.to_dict().keys())
            assert all(dk in data_keys for dk in self.deup_data_keys), (
                f"Some deup data keys ({self.deup_data_keys}) are missing"
                + f" from the data dict ({data_keys})"
            )
            out = torch.cat(
                [out]
                + [data[f"deup_{k}"][:, None].float() for k in self.deup_features],
                dim=-1,
            )
            out = self.deup_lin(out)
            out = self.act(out)
            out = self.lin2(out)

        return out


@registry.register_model("deup_faenet")
class DeupFAENet(FAENet):
    def __init__(self, *args, **kwargs):
        kwargs["dropout_edge"] = 0
        super().__init__(*args, **kwargs)
        self.output_block = DeupOutputBlock(
            self.energy_head,
            kwargs["hidden_channels"],
            self.act,
            self.dropout_lin,
            kwargs.get("deup_features", {}),
        )
        assert (
            self.energy_head != "weighted-av-initial-embeds"
        ), "Unsupported head weighted-av-initial-embeds"
        assert self.skip_co != "concat", "Unsupported skip connection concat"
        assert self.skip_co != "add", "Unsupported skip connection add"
