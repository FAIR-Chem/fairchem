import torch
from torch import nn

from ocpmodels.common import distutils


class L2MAELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        assert reduction in ["mean", "sum"]

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        dists = torch.norm(input - target, p=2, dim=-1)
        if self.reduction == "mean":
            return torch.mean(dists)
        elif self.reduction == "sum":
            return torch.sum(dists)


class NormL2MAELoss(nn.Module):
    def __init__(self, reduction="mean", min_norm=1e-3):
        super().__init__()
        self.reduction = reduction
        assert reduction in ["mean", "sum"]
        self.min_norm = min_norm

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        input_norm = torch.norm(input, p=2, dim=-1, keepdim=True).clamp(
            min=self.min_norm
        )
        input = input / input_norm
        target_norm = torch.norm(target, p=2, dim=-1, keepdim=True).clamp(
            min=self.min_norm
        )
        target = target / target_norm
        dists = torch.norm(input - target, p=2, dim=-1)
        if self.reduction == "mean":
            return torch.mean(dists)
        elif self.reduction == "sum":
            return torch.sum(dists)


class CosineLoss(nn.Module):
    def __init__(self, reduction="mean", eps=1e-6):
        super().__init__()
        self.reduction = reduction
        assert reduction in ["mean", "sum"]
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        neg_cos = -torch.cosine_similarity(input, target, eps=self.eps)
        if self.reduction == "mean":
            return torch.mean(neg_cos)
        elif self.reduction == "sum":
            return torch.sum(neg_cos)


class CombinedLoss(nn.Module):
    def __init__(self, loss_fns, weights=None):
        super().__init__()
        self.loss_fns = loss_fns
        if weights is None:
            self.weights = [1.0 for _ in range(len(loss_fns))]
        else:
            self.weights = weights

    @property
    def reduction(self):
        return self.loss_fns[0].reduction

    @reduction.setter
    def reduction(self, reduction):
        for loss_fn in self.loss_fns:
            loss_fn.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        return sum(
            [
                weight * loss_fn(input, target)
                for weight, loss_fn in zip(self.weights, self.loss_fns)
            ]
        )


def get_loss(loss_name, loss_kwargs):
    if loss_name in ["l1", "mae"]:
        return nn.L1Loss(**loss_kwargs)
    elif loss_name == "mse":
        return nn.MSELoss(**loss_kwargs)
    elif loss_name == "l2mae":
        return L2MAELoss(**loss_kwargs)
    elif loss_name == "norml2mae":
        return NormL2MAELoss(**loss_kwargs)
    elif loss_name == "cos":
        return CosineLoss(**loss_kwargs)
    else:
        raise NotImplementedError(f"Unknown loss function name: {loss_name}")


class DDPLoss(nn.Module):
    def __init__(self, loss_fn, reduction="mean"):
        super().__init__()
        self.loss_fn = loss_fn
        self.loss_fn.reduction = "sum"
        self.reduction = reduction
        assert reduction in ["mean", "sum"]

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        loss = self.loss_fn(input, target)
        if self.reduction == "mean":
            num_samples = input.shape[0]
            num_samples = distutils.all_reduce(
                num_samples, device=input.device
            )
            # Multiply by world size since gradients are averaged
            # across DDP replicas
            return loss * distutils.get_world_size() / num_samples
        else:
            return loss
