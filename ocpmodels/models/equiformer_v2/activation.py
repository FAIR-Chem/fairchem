import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledSiLU(nn.Module):
    def __init__(self, inplace: bool = False) -> None:
        super(ScaledSiLU, self).__init__()
        self.inplace = inplace
        self.scale_factor = 1.6791767923989418

    def forward(self, inputs):
        return F.silu(inputs, inplace=self.inplace) * self.scale_factor

    def extra_repr(self):
        str = "scale_factor={}".format(self.scale_factor)
        if self.inplace:
            str = str + ", inplace=True"
        return str


# Reference: https://github.com/facebookresearch/llama/blob/main/llama/model.py#L175
class ScaledSwiGLU(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, bias: bool = True
    ) -> None:
        super(ScaledSwiGLU, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w = torch.nn.Linear(in_channels, 2 * out_channels, bias=bias)
        self.act = ScaledSiLU()

    def forward(self, inputs):
        w = self.w(inputs)
        w_1 = w.narrow(-1, 0, self.out_channels)
        w_1 = self.act(w_1)
        w_2 = w.narrow(-1, self.out_channels, self.out_channels)
        out = w_1 * w_2
        return out


# Reference: https://github.com/facebookresearch/llama/blob/main/llama/model.py#L175
class SwiGLU(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, bias: bool = True
    ) -> None:
        super(SwiGLU, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w = torch.nn.Linear(in_channels, 2 * out_channels, bias=bias)
        self.act = torch.nn.SiLU()

    def forward(self, inputs):
        w = self.w(inputs)
        w_1 = w.narrow(-1, 0, self.out_channels)
        w_1 = self.act(w_1)
        w_2 = w.narrow(-1, self.out_channels, self.out_channels)
        out = w_1 * w_2
        return out


class SmoothLeakyReLU(torch.nn.Module):
    def __init__(self, negative_slope: float = 0.2) -> None:
        super().__init__()
        self.alpha = negative_slope

    def forward(self, x):
        x1 = ((1 + self.alpha) / 2) * x
        x2 = ((1 - self.alpha) / 2) * x * (2 * torch.sigmoid(x) - 1)
        return x1 + x2

    def extra_repr(self):
        return "negative_slope={}".format(self.alpha)


class ScaledSmoothLeakyReLU(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.act = SmoothLeakyReLU(0.2)
        self.scale_factor = 1.531320475574866

    def forward(self, x):
        return self.act(x) * self.scale_factor

    def extra_repr(self):
        return "negative_slope={}, scale_factor={}".format(
            self.act.alpha, self.scale_factor
        )


class ScaledSigmoid(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale_factor = 1.8467055342154763

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * self.scale_factor


class GateActivation(torch.nn.Module):
    def __init__(self, lmax: int, mmax: int, num_channels: int) -> None:
        super().__init__()

        self.lmax = lmax
        self.mmax = mmax
        self.num_channels = num_channels

        # compute `expand_index` based on `lmax` and `mmax`
        num_components = 0
        for lval in range(1, self.lmax + 1):
            num_m_components = min((2 * lval + 1), (2 * self.mmax + 1))
            num_components = num_components + num_m_components
        expand_index = torch.zeros([num_components]).long()
        start_idx = 0
        for lval in range(1, self.lmax + 1):
            length = min((2 * lval + 1), (2 * self.mmax + 1))
            expand_index[start_idx : (start_idx + length)] = lval - 1
            start_idx = start_idx + length
        self.register_buffer("expand_index", expand_index)

        self.scalar_act = (
            torch.nn.SiLU()
        )  # SwiGLU(self.num_channels, self.num_channels)  # #
        self.gate_act = torch.nn.Sigmoid()  # torch.nn.SiLU() # #

    def forward(self, gating_scalars, input_tensors):
        """
        `gating_scalars`: shape [N, lmax * num_channels]
        `input_tensors`: shape  [N, (lmax + 1) ** 2, num_channels]
        """

        gating_scalars = self.gate_act(gating_scalars)
        gating_scalars = gating_scalars.reshape(
            gating_scalars.shape[0], self.lmax, self.num_channels
        )
        gating_scalars = torch.index_select(
            gating_scalars, dim=1, index=self.expand_index
        )

        input_tensors_scalars = input_tensors.narrow(1, 0, 1)
        input_tensors_scalars = self.scalar_act(input_tensors_scalars)

        input_tensors_vectors = input_tensors.narrow(
            1, 1, input_tensors.shape[1] - 1
        )
        input_tensors_vectors = input_tensors_vectors * gating_scalars

        output_tensors = torch.cat(
            (input_tensors_scalars, input_tensors_vectors), dim=1
        )

        return output_tensors


class S2Activation(torch.nn.Module):
    """
    Assume we only have one resolution
    """

    def __init__(self, lmax: int, mmax: int) -> None:
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.act = torch.nn.SiLU()

    def forward(self, inputs, SO3_grid):
        to_grid_mat = SO3_grid[self.lmax][self.mmax].get_to_grid_mat(
            device=None
        )  # `device` is not used
        from_grid_mat = SO3_grid[self.lmax][self.mmax].get_from_grid_mat(
            device=None
        )
        x_grid = torch.einsum("bai, zic -> zbac", to_grid_mat, inputs)
        x_grid = self.act(x_grid)
        outputs = torch.einsum("bai, zbac -> zic", from_grid_mat, x_grid)
        return outputs


class SeparableS2Activation(torch.nn.Module):
    def __init__(self, lmax: int, mmax: int) -> None:
        super().__init__()

        self.lmax = lmax
        self.mmax = mmax

        self.scalar_act = torch.nn.SiLU()
        self.s2_act = S2Activation(self.lmax, self.mmax)

    def forward(self, input_scalars, input_tensors, SO3_grid):
        output_scalars = self.scalar_act(input_scalars)
        output_scalars = output_scalars.reshape(
            output_scalars.shape[0], 1, output_scalars.shape[-1]
        )
        output_tensors = self.s2_act(input_tensors, SO3_grid)
        outputs = torch.cat(
            (
                output_scalars,
                output_tensors.narrow(1, 1, output_tensors.shape[1] - 1),
            ),
            dim=1,
        )
        return outputs
