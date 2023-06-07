from typing import Iterable, cast

import torch
from lightning.pytorch.utilities import grad_norm
from typing_extensions import override

from ...util.typing_utils import mixin_base_type
from ..config import BaseConfig
from .distributed import DistributedMixin


def _compute_grad_norm(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    p: float | str = 2.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return torch.tensor(0.0)

    if device is None:
        device = grads[0].device
    return torch.norm(
        torch.stack(
            [
                torch.norm(
                    g.detach(),
                    p=p,  # type: ignore
                ).to(device)
                for g in grads
            ]
        ),
        p=p,  # type: ignore
    )


class OptimizerModuleMixin(mixin_base_type(DistributedMixin)):
    def log_gradient_norms(self, p: float | str = 2.0):
        grad_norm = _compute_grad_norm(
            self.parameters(),
            p=2.0 if p is True else p,
            device=self.device,
        )
        self.log("train/grad_norm", grad_norm, on_step=True, on_epoch=False)
        return grad_norm

    @override
    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: int | float | None = None,
        gradient_clip_algorithm: str | None = None,
    ):
        config = cast(BaseConfig, self.hparams)
        if log_grad_norm := config.trainer.logging.log_grad_norm:
            _ = self.log_gradient_norms(log_grad_norm)
        if log_grad_norm_per_param := config.trainer.logging.log_grad_norm_per_param:
            self.log_dict(
                {
                    f"train/{k}": v
                    for k, v in grad_norm(
                        self,
                        norm_type=2.0
                        if log_grad_norm_per_param is True
                        else log_grad_norm_per_param,
                    ).items()
                }
            )

        super().configure_gradient_clipping(
            optimizer, gradient_clip_val, gradient_clip_algorithm
        )

    def ddp_average_loss(
        self,
        loss: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        if (trainer := self._trainer) is None:
            raise ValueError("ddp_average_loss called without trainer.")

        if mask is not None:
            # compute batch size based on valid entries only
            if mask.shape != loss.shape:
                raise ValueError(f"{mask.shape=} != {loss.shape=}")
            batch_size = mask.sum().to(loss.dtype)
        else:
            # compute batch size based on all entries
            batch_size = torch.tensor(
                loss.numel(), dtype=loss.dtype, device=loss.device
            )

        # sum locally computed loss
        loss = loss.sum()
        # average loss over all processes
        loss = loss * (trainer.world_size / self.reduce(batch_size, "sum"))
        return loss
