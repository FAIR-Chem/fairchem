import inspect
import math
from bisect import bisect

import torch


def multiply(obj, num):
    if isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = obj[i] * num
    else:
        obj = obj * num
    return obj


def cosine_lr_lambda(current_step, scheduler_params):
    warmup_epochs = scheduler_params["warmup_epochs"]
    lr_warmup_factor = scheduler_params["warmup_factor"]
    max_epochs = scheduler_params["epochs"]
    lr_min_factor = scheduler_params["lr_min_factor"]

    # `warmup_epochs` is already multiplied with the num of iterations
    if current_step <= warmup_epochs:
        alpha = current_step / float(warmup_epochs)
        return lr_warmup_factor * (1.0 - alpha) + alpha
    else:
        if current_step >= max_epochs:
            return lr_min_factor
        lr_scale = lr_min_factor + 0.5 * (1 - lr_min_factor) * (
            1 + math.cos(math.pi * (current_step / max_epochs))
        )
        return lr_scale


class CosineLRLambda:
    def __init__(self, scheduler_params):
        self.warmup_epochs = scheduler_params["warmup_epochs"]
        self.lr_warmup_factor = scheduler_params["warmup_factor"]
        self.max_epochs = scheduler_params["epochs"]
        self.lr_min_factor = scheduler_params["lr_min_factor"]

    def __call__(self, current_step):
        # `warmup_epochs` is already multiplied with the num of iterations
        if current_step <= self.warmup_epochs:
            alpha = current_step / float(self.warmup_epochs)
            return self.lr_warmup_factor * (1.0 - alpha) + alpha
        else:
            if current_step >= self.max_epochs:
                return self.lr_min_factor
            lr_scale = self.lr_min_factor + 0.5 * (1 - self.lr_min_factor) * (
                1 + math.cos(math.pi * (current_step / self.max_epochs))
            )
            return lr_scale


def multistep_lr_lambda(current_step, scheduler_params):
    warmup_epochs = scheduler_params["warmup_epochs"]
    lr_warmup_factor = scheduler_params["warmup_factor"]
    lr_decay_epochs = scheduler_params["decay_epochs"]
    lr_gamma = scheduler_params["decay_rate"]

    if current_step <= warmup_epochs:
        alpha = current_step / float(warmup_epochs)
        return lr_warmup_factor * (1.0 - alpha) + alpha
    else:
        idx = bisect(lr_decay_epochs, current_step)
        return pow(lr_gamma, idx)


class MultistepLRLambda:
    def __init__(self, scheduler_params):
        self.warmup_epochs = scheduler_params["warmup_epochs"]
        self.lr_warmup_factor = scheduler_params["warmup_factor"]
        self.lr_decay_epochs = scheduler_params["decay_epochs"]
        self.lr_gamma = scheduler_params["decay_rate"]

    def __call__(self, current_step):
        if current_step <= self.warmup_epochs:
            alpha = current_step / float(self.warmup_epochs)
            return self.lr_warmup_factor * (1.0 - alpha) + alpha
        else:
            idx = bisect(self.lr_decay_epochs, current_step)
            return pow(self.lr_gamma, idx)


class LRScheduler:
    """
    Notes:
        1. scheduler.step() is called for every step for OC20 training.
        2. We use "scheduler_params" in .yml to specify scheduler parameters.
        3. For cosine learning rate, we use LambdaLR with lambda function being cosine:
            scheduler: LambdaLR
            scheduler_params:
                lambda_type: cosine
                ...
        4. Following 3., if `cosine` is used, `scheduler_params` in .yml looks like:
            scheduler: LambdaLR
            scheduler_params:
                lambda_type: cosine
                warmup_epochs: ...
                warmup_factor: ...
                lr_min_factor: ...
        5. Following 3., if `multistep` is used, `scheduler_params` in .yml looks like:
            scheduler: LambdaLR
            scheduler_params:
                lambda_type: multistep
                warmup_epochs: ...
                warmup_factor: ...
                decay_epochs: ... (list)
                decay_rate: ...

    Args:
        optimizer (obj): torch optim object
        config (dict): Optim dict from the input config
    """

    def __init__(self, optimizer, config):
        self.optimizer = optimizer
        self.config = config.copy()

        assert "scheduler" in self.config.keys()
        assert "scheduler_params" in self.config.keys()
        self.scheduler_type = self.config["scheduler"]
        self.scheduler_params = self.config["scheduler_params"].copy()

        # Use `LambdaLR` for multi-step and cosine learning rate
        if self.scheduler_type == "LambdaLR":
            scheduler_lambda_fn = None
            self.lambda_type = self.scheduler_params["lambda_type"]

            if self.lambda_type == "cosine":
                scheduler_lambda_fn = CosineLRLambda(self.scheduler_params)
            elif self.lambda_type == "multistep":
                scheduler_lambda_fn = MultistepLRLambda(self.scheduler_params)
            else:
                raise ValueError
            self.scheduler_params["lr_lambda"] = scheduler_lambda_fn

        if self.scheduler_type != "Null":
            self.scheduler = getattr(
                torch.optim.lr_scheduler, self.scheduler_type
            )
            scheduler_args = self.filter_kwargs(self.scheduler_params)
            self.scheduler = self.scheduler(optimizer, **scheduler_args)

    def step(self, metrics=None, epoch=None):
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
            arg: config[arg] for arg in config if arg in filter_keys
        }
        return scheduler_args

    def get_lr(self):
        for group in self.optimizer.param_groups:
            return group["lr"]
