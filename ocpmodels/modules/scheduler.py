"""scheduler.py
"""
import inspect
import torch.optim.lr_scheduler as lr_scheduler

from ocpmodels.common.utils import warmup_lr_lambda
import pytorch_warmup as warmup


class LRScheduler:
    """
    Learning rate scheduler class for torch.optim learning rate schedulers

    Notes:
        If no learning rate scheduler is specified in the config the default
        scheduler is warmup_lr_lambda (ocpmodels.common.utils) not no scheduler,
        this is for backward-compatibility reasons. To run without a lr scheduler
        specify scheduler: "Null" in the optim section of the config.

    Args:
        config (dict): Optim dict from the input config
        optimizer (obj): torch optim object
    """

    def __init__(self, optimizer, optim_config):
        self.optimizer = optimizer
        self.optim_config = optim_config.copy()
        self.warmup_scheduler = None
        if self.optim_config.get("scheduler"):
            self.scheduler_type = self.optim_config["scheduler"]
        else:
            self.scheduler_type = "LambdaLR"

            def scheduler_lambda_fn(x):
                return warmup_lr_lambda(x, self.optim_config)

            self.optim_config["lr_lambda"] = scheduler_lambda_fn

        if (
            self.scheduler_type != "Null"
            and self.scheduler_type != "LinearWarmupCosineAnnealingLR"
        ):
            self.scheduler = getattr(lr_scheduler, self.scheduler_type)
            scheduler_args = self.filter_kwargs(self.optim_config)
            self.scheduler = self.scheduler(optimizer, **scheduler_args)
        elif self.scheduler_type == "LinearWarmupCosineAnnealingLR":
            T_max = (
                self.optim_config.get("fidelity_max_steps")
                or self.optim_config["max_steps"]
            )
            self.warmup_scheduler = warmup.ExponentialWarmup(
                self.optimizer, warmup_period=self.optim_config["warmup_steps"]
            )
            self.scheduler = lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=T_max, eta_min=1e-7
            )

    def step(self, metrics=None, epoch=None):
        if self.scheduler_type == "Null":
            return
        if self.scheduler_type == "ReduceLROnPlateau":
            if metrics is None:
                raise Exception("Validation set required for ReduceLROnPlateau.")
            self.scheduler.step(metrics)
        else:
            if self.warmup_scheduler:
                with self.warmup_scheduler.dampening():
                    self.scheduler.step(epoch)
            else:
                self.scheduler.step()

    def filter_kwargs(self, optim_config):
        # adapted from https://stackoverflow.com/questions/26515595/
        sig = inspect.signature(self.scheduler)
        filter_keys = [
            param.name
            for param in sig.parameters.values()
            if param.kind == param.POSITIONAL_OR_KEYWORD
        ]
        filter_keys.remove("optimizer")
        scheduler_args = {
            arg: optim_config[arg] for arg in optim_config if arg in filter_keys
        }
        return scheduler_args

    def get_lr(self):
        for group in self.optimizer.param_groups:
            return group["lr"]


class EarlyStopper:
    """
    Class that stores the current best metric score and monitors whether
    it's improving or not. If it does not decrease for a certain number
    of validation calls (with some minimal improvement) then it tells the trainer
    to stop.
    """

    def __init__(
        self, patience=7, mode="min", min_abs_change=1e-5, store_all_steps=True
    ):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.min_abs_change = min_abs_change
        self.store_all_steps = store_all_steps
        self.metrics = []

        if self.mode == "min":
            self.best_score = float("inf")
        elif self.mode == "max":
            self.best_score = float("-inf")
        else:
            raise ValueError("mode must be either min or max")

        self.early_stop = False

    def should_stop(self, metric):
        """
        Returns True if the metric has not improved for a certain number of
        steps. False otherwise. Stores the metric in `self.metrics`: all the steps if
        `self.store_all_steps` is `True`, otherwise only the last `n=self.patience`.

        Args:
            metric (Number): Metric to track.

        Returns:
            bool: Wether to stop training or not
        """
        metric = float(metric)
        self.metrics.append(metric)
        if not self.store_all_steps:
            self.metrics = self.metrics[-self.patience :]

        if self.mode == "min":
            if metric < self.best_score - self.min_abs_change:
                self.best_score = metric
                self.counter = 0
            else:
                self.counter += 1
        elif self.mode == "max":
            if metric > self.best_score + self.min_abs_change:
                self.best_score = metric
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop

    @property
    def reason(self):
        return (
            f"Early stopping after {self.counter} steps with no improvement:\n"
            + " -> ".join([f"{m:.6f}" for m in self.metrics[-self.patience :]])
        )
