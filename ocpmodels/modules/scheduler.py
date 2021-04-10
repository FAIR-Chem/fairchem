import inspect

import torch.optim.lr_scheduler as lr_scheduler

from ocpmodels.common.utils import warmup_lr_lambda


class LRScheduler:
    def __init__(self, optimizer, config):
        self.optimizer = optimizer
        self.config = config.copy()
        if "scheduler" in self.config:
            self.scheduler_type = self.config["scheduler"]
        else:
            self.scheduler_type = "LambdaLR"
            scheduler_lambda_fn = lambda x: warmup_lr_lambda(x, self.config)
            self.config["lr_lambda"] = scheduler_lambda_fn

        self.scheduler = getattr(lr_scheduler, self.scheduler_type)
        scheduler_args = self.filter_kwargs(config)
        self.scheduler = self.scheduler(optimizer, **scheduler_args)
        # sets the learning rate update type i.e. update on step, epoch, or val
        (
            self.update_lr_on_step,
            self.update_lr_on_epoch,
            self.update_lr_on_val,
        ) = self.set_lr_update_type(self.config)

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

    @staticmethod
    def set_lr_update_type(config):

        update_lr_on_step = False
        update_lr_on_epoch = False
        update_lr_on_val = False

        if "update_lr_on" in config:
            # checks for allowed strings
            if config["update_lr_on"] in ["step", "epoch", "val"]:
                if config["update_lr_on"] == "step":
                    update_lr_on_step = True
                if config["update_lr_on"] == "epoch":
                    update_lr_on_epoch = True
                if config["update_lr_on"] == "val":
                    update_lr_on_val = True
            else:
                raise Exception(
                    "update_lr_on in the config expects one of following strings: step, epoch, or val"
                )
        # If update_lr_on is not defined in the config default is update_lr_on_epoch
        else:
            update_lr_on_epoch = True

        return update_lr_on_step, update_lr_on_epoch, update_lr_on_val

    def get_lr(self):
        for group in self.optimizer.param_groups:
            return group["lr"]
