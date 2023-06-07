from abc import ABC, abstractmethod
from functools import wraps
from logging import getLogger
from typing import TYPE_CHECKING, Iterable

import torch.nn as nn
import torchmetrics
from typing_extensions import override

log = getLogger(__name__)


class MetricBase(nn.Module, ABC):
    Metrics = Iterable[dict[str, torchmetrics.Metric]]

    @override
    def __init__(self, prefix: str | None = None):
        super().__init__()

        self.prefix = prefix

        # update `self.metrics` to include prefix in the keys
        if self.prefix is not None:
            old_metrics = self.metrics

            @wraps(old_metrics)
            def new_metrics() -> Iterable[dict[str, torchmetrics.Metric]]:
                for m in old_metrics():
                    yield {f"{self.prefix}{k}": v for k, v in m.items()}

            self.metrics = new_metrics
            log.info(
                f"Updated {self.__class__.__name__}.metrics to include {self.prefix=}."
            )

    @abstractmethod
    def compute(self, *args, **kwargs) -> None:
        ...

    @abstractmethod
    def metrics(self) -> Metrics:
        ...

    def flattened_metrics(self) -> dict[str, torchmetrics.Metric]:
        metrics: dict[str, torchmetrics.Metric] = {}
        for m in self.metrics():
            metrics.update(m)
        return metrics

    @override
    def forward(self, *args, **kwargs):
        self.compute(*args, **kwargs)
        return self.flattened_metrics()

    if TYPE_CHECKING:

        @override
        def __call__(self, *args, **kwargs) -> dict[str, torchmetrics.Metric]:
            ...
