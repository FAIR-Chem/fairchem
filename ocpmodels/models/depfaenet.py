import torch
from torch.nn import Linear
from torch import nn
from torch_scatter import scatter
import torch.nn.functional as F

from ocpmodels.models.faenet import FAENet
from ocpmodels.models.faenet import OutputBlock as conOutputBlock
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad
from ocpmodels.models.utils.activations import swish

from torch_geometric.data import Batch


class DiscOutputBlock(conOutputBlock):
    def __init__(self, energy_head, hidden_channels, act, dropout_lin, disconnected_mlp=False):
        super(DiscOutputBlock, self).__init__(energy_head, hidden_channels, act, dropout_lin)

        # We modify the last output linear function to make the output a vector
        self.lin2 = Linear(hidden_channels // 2, hidden_channels // 2)

        self.disconnected_mlp = disconnected_mlp
        if self.disconnected_mlp:
            self.ads_lin = Linear(hidden_channels // 2, hidden_channels // 2)
            self.cat_lin = Linear(hidden_channels // 2, hidden_channels // 2)

        # Combines the hidden representation of each to a scalar.
        self.combination = nn.Sequential(
            Linear(hidden_channels // 2 * 2, hidden_channels // 2),
            swish,
            Linear(hidden_channels // 2, 1),
        )

    def tags_saver(self, tags):
        self.current_tags = tags

    def forward(self, h, edge_index, edge_weight, batch, alpha, data):
        if (
            self.energy_head == "weighted-av-final-embeds"
        ):  # Right now, this is the only available option.
            alpha = self.w_lin(h)

        # MLP
        h = F.dropout(
            h, p=self.dropout_lin, training=self.training or self.deup_inference
        )
        h = self.lin1(h)
        h = self.act(h)
        h = F.dropout(
            h, p=self.dropout_lin, training=self.training or self.deup_inference
        )
        h = self.lin2(h)

        if self.energy_head in {
            "weighted-av-initial-embeds",
            "weighted-av-final-embeds",
        }:
            h = h * alpha

        # We pool separately and then we concatenate.
        ads = self.current_tags == 2
        cat = ~ads

        ads_out = scatter(h, batch * ads, dim=0, reduce="add")
        cat_out = scatter(h, batch * cat, dim=0, reduce="add")

        if self.disconnected_mlp:
            ads_out = self.ads_lin(ads_out)
            cat_out = self.cat_lin(cat_out)

        system = torch.cat([ads_out, cat_out], dim=1)

        # Finally, we predict a number.
        energy = self.combination(system)

        return energy


@registry.register_model("depfaenet")
class DepFAENet(FAENet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # We replace the old output block by the new output block
        self.disconnected_mlp = kwargs.get("disconnected_mlp", False)
        self.output_block = DiscOutputBlock(
            self.energy_head, kwargs["hidden_channels"], self.act, self.dropout_lin, self.disconnected_mlp,
        )

    @conditional_grad(torch.enable_grad())
    def energy_forward(self, data, q=None):
        # We need to save the tags so this step is necessary.
        self.output_block.tags_saver(data.tags)
        pred = super().energy_forward(data, q)

        return pred
