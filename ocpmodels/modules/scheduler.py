import inspect

import torch.optim.lr_scheduler as lr_scheduler

from ocpmodels.common.utils import warmup_lr_lambda


class LRScheduler:
    """
    Learning rate scheduler class for torch.optim learning rate schedulers

    Notes:
        If no learning rate scheduler is specified in the config the default
        scheduler is warmup_lr_lambda (ocpmodels.common.utils) not no scheduler,
        this is for backward-compatibility reasons. To run without a lr scheduler
        specify scheduler: "Null" in the optim section of the config.

    Args:
        optimizer (obj): torch optim object
        config (dict): Optim dict from the input config
    """

    def __init__(self, optimizer, config) -> None:
        self.optimizer = optimizer
        self.config = config.copy()
        if "scheduler" in self.config:
            self.scheduler_type = self.config["scheduler"]
        else:
            self.scheduler_type = "LambdaLR"
            scheduler_lambda_fn = lambda x: warmup_lr_lambda(x, self.config)
            self.config["lr_lambda"] = scheduler_lambda_fn

        if self.scheduler_type != "Null":
            self.scheduler = getattr(lr_scheduler, self.scheduler_type)
            scheduler_args = self.filter_kwargs(config)
            self.scheduler = self.scheduler(optimizer, **scheduler_args)

    def step(self, metrics=None, epoch=None) -> None:
        if self.scheduler_type == "Null":
            return
        if self.scheduler_type == "ReduceLROnPlateau":
            if metrics is None:
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
        filter_keys.remove("optimizer")
        scheduler_args = {
            arg: self.config[arg] for arg in self.config if arg in filter_keys
        }
        return scheduler_args

    def get_lr(self):
        for group in self.optimizer.param_groups:
            return group["lr"]
