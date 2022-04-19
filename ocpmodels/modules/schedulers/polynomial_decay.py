from typing import Optional

from torch.optim.lr_scheduler import _LRScheduler


class PolynomialLRDecay(_LRScheduler):
    # credits to https://github.com/cmpark0126/pytorch-polynomial-lr-decay
    def __init__(
        self,
        optimizer,
        max_steps: int,
        final_lr: float = 0.0001,
        power: float = 1.0,
    ):
        self.max_steps = max_steps
        self.final_lr = final_lr
        self.power = power

        self.last_step = 0

        super().__init__(optimizer)

    def get_lr(self):
        if self.last_step > self.max_steps:
            return [self.final_lr for _ in self.base_lrs]

        return [
            (base_lr - self.final_lr)
            * ((1 - self.last_step / self.max_steps) ** (self.power))
            + self.final_lr
            for base_lr in self.base_lrs
        ]

    def step(self, step: Optional[int] = None):
        if step is None:
            step = self.last_step + 1

        self.last_step = step if step != 0 else 1
        if self.last_step <= self.max_steps:
            decay_lrs = [
                (base_lr - self.final_lr)
                * ((1 - self.last_step / self.max_steps) ** (self.power))
                + self.final_lr
                for base_lr in self.base_lrs
            ]
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group["lr"] = lr
