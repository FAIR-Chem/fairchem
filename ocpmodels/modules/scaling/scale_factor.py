import itertools
import logging
import math
from contextlib import contextmanager
from typing import Callable, Optional, TypedDict, Union

import torch
import torch.nn as nn


class _Stats(TypedDict):
    variance_in: float
    variance_out: float
    n_samples: int


IndexFn = Callable[[], None]


def _check_consistency(old: torch.Tensor, new: torch.Tensor, key: str) -> None:
    if not torch.allclose(old, new):
        raise ValueError(
            f"Scale factor parameter {key} is inconsistent with the loaded state dict.\n"
            f"Old: {old}\n"
            f"Actual: {new}"
        )


class ScaleFactor(nn.Module):
    scale_factor: torch.Tensor

    name: Optional[str] = None
    index_fn: Optional[IndexFn] = None
    stats: Optional[_Stats] = None

    def __init__(
        self,
        name: Optional[str] = None,
        enforce_consistency: bool = True,
    ) -> None:
        super().__init__()

        self.name = name
        self.index_fn = None
        self.stats = None

        self.scale_factor = nn.parameter.Parameter(
            torch.tensor(0.0), requires_grad=False
        )
        if enforce_consistency:
            self._register_load_state_dict_pre_hook(self._enforce_consistency)

    def _enforce_consistency(
        self,
        state_dict,
        prefix,
        _local_metadata,
        _strict,
        _missing_keys,
        _unexpected_keys,
        _error_msgs,
    ) -> None:
        if not self.fitted:
            return

        persistent_buffers = {
            k: v
            for k, v in self._buffers.items()
            if k not in self._non_persistent_buffers_set
        }
        local_name_params = itertools.chain(
            self._parameters.items(), persistent_buffers.items()
        )
        local_state = {k: v for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if key not in state_dict:
                continue

            input_param = state_dict[key]
            _check_consistency(old=param, new=input_param, key=key)

    @property
    def fitted(self) -> bool:
        return bool((self.scale_factor != 0.0).item())

    @torch.jit.unused
    def reset_(self) -> None:
        self.scale_factor.zero_()

    @torch.jit.unused
    def set_(self, scale: Union[float, torch.Tensor]) -> None:
        if self.fitted:
            _check_consistency(
                old=self.scale_factor,
                new=torch.tensor(scale) if isinstance(scale, float) else scale,
                key="scale_factor",
            )
        self.scale_factor.fill_(scale)

    @torch.jit.unused
    def initialize_(self, *, index_fn: Optional[IndexFn] = None) -> None:
        self.index_fn = index_fn

    @contextmanager
    @torch.jit.unused
    def fit_context_(self):
        self.stats = _Stats(variance_in=0.0, variance_out=0.0, n_samples=0)
        yield
        del self.stats
        self.stats = None

    @torch.jit.unused
    def fit_(self):
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

        self.set_(value)

        stats = dict(**self.stats)
        return stats, ratio, value

    @torch.no_grad()
    @torch.jit.unused
    def _observe(
        self, x: torch.Tensor, ref: Optional[torch.Tensor] = None
    ) -> None:
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

    def forward(
        self,
        x: torch.Tensor,
        *,
        ref: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.index_fn is not None:
            self.index_fn()

        if self.fitted:
            x = x * self.scale_factor

        if not torch.jit.is_scripting():
            self._observe(x, ref=ref)

        return x
