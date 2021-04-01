import inspect

import torch.optim.lr_scheduler as lr_scheduler

from ocpmodels.common.utils import warmup_lr_lambda


class LRScheduler:
    def __init__(self, optimizer, config):
        self.config = config
        if "scheduler" in self.config:
            self.scheduler_type = self.config["scheduler"]
        else:
            self.scheduler_type = "LambdaLR"
            scheduler_lambda_fn = lambda x: warmup_lr_lambda(x, self.config)
            self.config["lr_lambda"] = scheduler_lambda_fn

        self.scheduler = getattr(lr_scheduler, self.scheduler_type)
        scheduler_args = self.filter_kwargs(config)
        self.scheduler = self.scheduler(optimizer, **scheduler_args)

    def step(self, metrics=None, epoch=None):
        if self.scheduler_type == "ReduceLROnPlateau":
            if not metrics:
                raise Exception(
                    "Validation set required for ReduceLROnPlateau."
                )
            self.scheduler.step(metrics)
        else:
            self.scheduler.step()

    def filter_kwargs(self, config):
        # adapted from https://stackoverflow.com/questions/26515595/
        sig = inspect.signature(self.scheduler)
        filter_keys = [
            param.name
            for param in sig.parameters.values()
            if param.kind == param.POSITIONAL_OR_KEYWORD
        ]
        scheduler_args = {
            arg: self.config[arg] for arg in self.config if arg in filter_keys
        }
        return scheduler_args
