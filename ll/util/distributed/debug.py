from datetime import timedelta
from functools import wraps
from logging import getLogger

import torch.distributed

logger = getLogger(__name__)


def _wrap(name: str, log: bool = False):
    fn = getattr(torch.distributed, name)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        nonlocal log

        if log:
            logger.critical(f"Calling {fn.__name__} with args: {args}, kwargs: {kwargs}")
        _ = torch.distributed.og_barrier()

        return fn(*args, **kwargs)

    setattr(torch.distributed, name, wrapper)


def debug_distributed(
    timeout: int | float | timedelta = 120.0,
    log: bool = False,
):
    if not isinstance(timeout, timedelta):
        timeout = timedelta(seconds=timeout)
    logger.critical(f"Patching torch.distributed for debug. Timeout: {timeout}. ")

    @wraps(torch.distributed.barrier)
    def barrier_fn(group=torch.distributed.GroupMember.WORLD):
        logger.critical(f"Calling torch.distributed.barrier.")
        return torch.distributed.monitored_barrier(
            group, timeout=timeout, wait_all_ranks=True
        )

    torch.distributed.og_barrier = torch.distributed.barrier
    torch.distributed.barrier = barrier_fn

    fn_names = [
        # "send",
        # "recv",
        "broadcast",
        "all_reduce",
        "reduce",
        "all_gather",
        "all_gather_object",
        "gather",
        "scatter",
        "reduce_scatter",
        "all_to_all",
    ]
    for name in fn_names:
        _wrap(name, log=log)
        logger.critical(f"Wrapped torch.distributed.{name}.")
