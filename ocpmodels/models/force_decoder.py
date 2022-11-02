import torch.nn as nn


class LambdaLayer(nn.Module):
    def __init__(self, func):
        super(LambdaLayer, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class ForceDecoder(nn.Module):
    def __init__(self, type, input_channel, model_configs, act):
        """
        Decoder predicting a force scalar per atom

        Args:
            type (str): Type of force decoder to use
            model_config (dict): Dictionary of config parameters for the decoder's model
            act (callable): Activation function (NOT a module)

        Raises:
            ValueError: Unknown type of decoder
        """
        super().__init__()
        self.type = type
        self.act = act
        assert type in model_configs, f"Unknown type of force decoder: `{type}`"
        self.model_config = model_configs[type]
        if self.type == "simple":
            self.model = nn.Sequential(
                nn.Linear(
                    input_channel,
                    self.model_config["hidden_channels"],
                ),
                LambdaLayer(act),
                nn.Linear(self.model_config["hidden_channels"], 3),
            )
        elif self.type == "mlp":  # from forcenet
            self.model = nn.Sequential(
                nn.Linear(
                    self.model_config["hidden_channels"],
                    self.model_config["hidden_channels"],
                ),
                nn.BatchNorm1d(self.model_config["hidden_channels"]),
                LambdaLayer(act),
                nn.Linear(self.model_config["hidden_channels"], 3),
            )
        else:
            raise ValueError(f"Unknown force decoder type: `{self.type}`")

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.model:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
            else:
                if hasattr(layer, "weight"):
                    nn.init.xavier_uniform_(layer.weight)
                if hasattr(layer, "bias"):
                    layer.bias.data.fill_(0)

    def forward(self, h):
        return self.model(h)
