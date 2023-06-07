from logging import getLogger

import torch
import torch.distributed
from varname import argname

log = getLogger(__name__)


class DebugModuleMixin:
    @torch.jit.unused
    def breakpoint(self, rank_zero_only: bool = True):
        if (
            not rank_zero_only
            or not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        ):
            breakpoint()

        if rank_zero_only and torch.distributed.is_initialized():
            _ = torch.distributed.barrier()

    @torch.jit.unused
    def ensure_finite(
        self,
        tensor: torch.Tensor,
        name: str | None = None,
        throw: bool = False,
    ):
        if name is None:
            arg_name = argname("tensor", vars_only=False)

            if arg_name is None:
                raise ValueError("Could not infer name for `tensor`")

            name = str(arg_name)

        not_finite = ~torch.isfinite(tensor)
        if not_finite.any():
            msg = f"Tensor {name} has {not_finite.sum().item()}/{not_finite.numel()} non-finite values."
            if throw:
                raise RuntimeError(msg)
            else:
                log.warning(msg)
            return False
        return True
