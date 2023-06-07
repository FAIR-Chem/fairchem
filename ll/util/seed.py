from contextlib import contextmanager
from logging import getLogger

import lightning_fabric.utilities.seed as LS

log = getLogger(__name__)


def seed_everything(seed: int, *, workers: bool = False):
    seed = LS.seed_everything(seed, workers=workers)
    log.critical(f"Set global seed to {seed}.")
    return seed


def reset_seed():
    LS.reset_seed()
    log.critical("Reset global seed.")


@contextmanager
def seed_context(seed: int | None, *, workers: bool = False):
    if seed is None:
        seed = LS._select_seed_randomly()
        log.warning(f"No seed provided, using random seed {seed}.")

    try:
        seed = seed_everything(seed, workers=workers)
        yield
    finally:
        reset_seed()
