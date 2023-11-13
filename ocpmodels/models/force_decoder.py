import torch.nn as nn


class LambdaLayer(nn.Module):
    def __init__(self, func):
        super(LambdaLayer, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class ForceDecoder(nn.Module):
    def __init__(self, type, input_channels, model_configs, act):
        """
        Decoder predicting a force scalar per atom

        Args:
            type (str): Type of force decoder to use
            input_channels (int): Number of input channels
            model_configs (dict): Dictionary of config parameters for the
                decoder's model
            act (callable): Activation function (NOT a module)

        Raises:
            ValueError: Unknown type of decoder
        """
        super().__init__()
        self.type = type
        self.act = act
        assert type in model_configs, f"Unknown type of force decoder: `{type}`"
        self.model_config = model_configs[type]
        if self.model_config.get("norm", "batch1d") == "batch1d":
            self.norm = lambda n: nn.BatchNorm1d(n)
        elif self.model_config["norm"] == "layer":
            self.norm = lambda n: nn.LayerNorm(n)
        elif self.model_config["norm"] in ["", None]:
            self.norm = lambda n: nn.Identity()
        else:
            raise ValueError(f"Unknown norm type: {self.model_config['norm']}")
        if self.type == "simple":
            assert "hidden_channels" in self.model_config
            self.model = nn.Sequential(
                nn.Linear(
                    input_channels,
                    self.model_config["hidden_channels"],
                ),
                LambdaLayer(act),
                nn.Linear(self.model_config["hidden_channels"], 3),
            )
        elif self.type == "mlp":  # from forcenet
            assert "hidden_channels" in self.model_config
            self.model = nn.Sequential(
                nn.Linear(
                    input_channels,
                    self.model_config["hidden_channels"],
                ),
                self.norm(self.model_config["hidden_channels"]),
                LambdaLayer(act),
                nn.Linear(self.model_config["hidden_channels"], 3),
            )
        elif self.type == "res":
            assert "hidden_channels" in self.model_config
            self.mlp_1 = nn.Sequential(
                nn.Linear(
                    input_channels,
                    input_channels,
                ),
                self.norm(input_channels),
                LambdaLayer(act),
            )
            self.mlp_2 = nn.Sequential(
                nn.Linear(
                    input_channels,
                    input_channels,
                ),
                self.norm(input_channels),
                LambdaLayer(act),
            )
            self.mlp_3 = nn.Sequential(
                nn.Linear(
                    input_channels,
                    self.model_config["hidden_channels"],
                ),
                self.norm(self.model_config["hidden_channels"]),
                LambdaLayer(act),
                nn.Linear(self.model_config["hidden_channels"], 3),
            )
        elif self.type == "res_updown":
            assert "hidden_channels" in self.model_config
            self.mlp_1 = nn.Sequential(
                nn.Linear(
                    input_channels,
                    self.model_config["hidden_channels"],
                ),
                self.norm(self.model_config["hidden_channels"]),
                LambdaLayer(act),
            )
            self.mlp_2 = nn.Sequential(
                nn.Linear(
                    self.model_config["hidden_channels"],
                    self.model_config["hidden_channels"],
                ),
                self.norm(self.model_config["hidden_channels"]),
                LambdaLayer(act),
            )
            self.mlp_3 = nn.Sequential(
                nn.Linear(
                    self.model_config["hidden_channels"],
                    input_channels,
                ),
                self.norm(input_channels),
                LambdaLayer(act),
            )
            self.mlp_4 = nn.Sequential(
                nn.Linear(
                    input_channels,
                    self.model_config["hidden_channels"],
                ),
                self.norm(self.model_config["hidden_channels"]),
                LambdaLayer(act),
                nn.Linear(self.model_config["hidden_channels"], 3),
            )
        else:
            raise ValueError(f"Unknown force decoder type: `{self.type}`")

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
            else:
                if hasattr(layer, "weight"):
                    nn.init.xavier_uniform_(layer.weight)
                if hasattr(layer, "bias"):
                    layer.bias.data.fill_(0)

    def forward(self, h):
        if self.type == "res":
            return self.mlp_3(self.mlp_2(self.mlp_1(h)) + h)
        elif self.type == "res_updown":
            return self.mlp_4(self.mlp_3(self.mlp_2(self.mlp_1(h))) + h)
        return self.model(h)
