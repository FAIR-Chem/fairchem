import logging
import math
from contextlib import contextmanager
from typing import Callable, Optional, TypedDict

import torch
import torch.nn as nn


class _Stats(TypedDict):
    variance_in: float
    variance_out: float
    n_samples: int


class ScaleFactor(nn.Module):
    scale: torch.Tensor
    fitted: torch.Tensor

    index_fn: Optional[Callable[["ScaleFactor"], None]] = None
    name: Optional[str] = None
    stats: Optional[_Stats] = None

    def __init__(self):
        super().__init__()

        self.index_fn = None
        self.name = None
        self.stats = None

        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("fitted", torch.tensor(False))

    def initialize(
        self,
        *,
        name: Optional[str] = None,
        index_fn: Optional[Callable[["ScaleFactor"], None]] = None,
    ):
        self.name = name
        self.index_fn = index_fn

    @torch.no_grad()
    @torch.jit.unused
    def fit(self):
        assert self.stats, "Stats not set"
        for k, v in self.stats.items():
            assert v > 0, f"{k} is {v}"

        self.stats["variance_in"] = (
            self.stats["variance_in"] / self.stats["n_samples"]
        )
        self.stats["variance_out"] = (
            self.stats["variance_out"] / self.stats["n_samples"]
        )

        ratio = self.stats["variance_out"] / self.stats["variance_in"]
        value = math.sqrt(1 / ratio)

        self.scale = self.scale * value
        self.fitted[...] = True

        logging.info(
            f"Variable: {self.name}, "
            f"Var_in: {self.stats['variance_in']:.3f}, "
            f"Var_out: {self.stats['variance_out']:.3f}, "
            f"Ratio: {ratio:.3f} => Scaling factor: {value:.3f}"
        )

        del self.stats
        self.stats = None

    @torch.no_grad()
    @torch.jit.unused
    @contextmanager
    def observe_and_fit(self):
        with torch.no_grad():
            self.stats = _Stats(variance_in=0.0, variance_out=0.0, n_samples=0)
            yield
            self.fit()

    @torch.no_grad()
    @torch.jit.unused
    def _observe(self, x: torch.Tensor, ref: Optional[torch.Tensor] = None):
        if self.fitted:
            return

        if self.stats is None:
            logging.debug("Observer not initialized but self.observe() called")
            return

        n_samples = x.shape[0]

        self.stats["variance_out"] += (
            torch.mean(torch.var(x, dim=0)).item() * n_samples
        )

        if ref is None:
            self.stats["variance_in"] += n_samples
        else:
            self.stats["variance_in"] += (
                torch.mean(torch.var(ref, dim=0)).item() * n_samples
            )
        self.stats["n_samples"] += n_samples

    def forward(self, x: torch.Tensor, ref: Optional[torch.Tensor] = None):
        if self.index_fn is not None:
            self.index_fn(self)

        x = x * self.scale
        if not torch.jit.is_scripting():
            self._observe(x, ref=ref)

        return x
