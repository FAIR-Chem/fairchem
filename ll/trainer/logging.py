from logging import getLogger
from pathlib import Path
from typing import Literal

from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.loggers.wandb import WandbLogger

from ..model.config import BaseConfig

log = getLogger(__name__)


def default_root_dir(
    config: BaseConfig,
    *,
    create_symlinks: bool = True,
    logs_dirname: str = "lightning_logs",
):
    base_path = (Path.cwd() / logs_dirname).resolve().absolute()
    path = base_path / config.id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _default_loggers(
    *,
    base_path: str | Path = ".",
    id: str | None = None,
    name: str | None = None,
    project: str | None = None,
    csv: bool = True,
    tensorboard: bool = True,
    wandb: bool = True,
    log_model: bool | Literal["all"] = False,
    notes: str | None = None,
    tags: list[str] | None = None,
) -> list[Logger]:
    base_path = Path(base_path)

    loggers: list[Logger] = []
    if wandb:
        log.info(f"Creating W&B logger for {project or 'lightning_logs'} with {id=}.")
        loggers.append(
            WandbLogger(
                save_dir=base_path,
                project=project or "lightning_logs",
                name=name,
                version=id,
                log_model=log_model,
                notes=notes,
                tags=tags,
            )
        )
    if csv:
        log.info(f"Creating CSV logger for {base_path / 'csv'} with {id=}.")
        loggers.append(
            CSVLogger(
                save_dir=base_path / "csv",
                name=name or ".",
                version=id,
            )
        )
    if tensorboard:
        log.info(
            f"Creating TensorBoard logger for {base_path / 'tensorboard'} with {id=}."
        )
        loggers.append(
            TensorBoardLogger(
                save_dir=base_path / "tensorboard",
                name=name,
                version=id,
            )
        )
    return loggers


def loggers_from_config(config: BaseConfig):
    logging_config = config.trainer.logging
    if not logging_config.enabled or config.trainer.logger is False:
        return []

    wandb_log_model = False
    if logging_config.wandb is not None:
        match (wandb_log_model := logging_config.wandb.log_model):
            case True | False | "all":
                log.info(f"W&B logging model: {wandb_log_model}.")
            case _:
                raise ValueError(f"Invalid wandb log_model value {wandb_log_model}.")

    return _default_loggers(
        base_path=default_root_dir(config),
        id=config.id,
        name=config.name,
        project=config.project,
        csv=logging_config.csv is not None and logging_config.csv.enabled,
        tensorboard=logging_config.tensorboard is not None
        and logging_config.tensorboard.enabled,
        wandb=logging_config.wandb is not None and logging_config.wandb.enabled,
        log_model=wandb_log_model,
        tags=config.tags,
        notes=(
            "\n".join(f"- {note}" for note in config.notes) if config.notes else None
        ),
    )


def validate_logger(logger: Logger, run_id: str):
    match logger:
        case CSVLogger() | TensorBoardLogger() | WandbLogger():
            if logger.version != run_id:
                raise ValueError(
                    f"{logger.__class__.__qualname__} version {logger.version} does not match run_id {run_id}"
                )
        case _:
            log.warning(
                f"Logger {logger.__class__.__qualname__} does not support run_id, ignoring."
            )


def finalize_loggers(loggers: list[Logger]):
    for logger in loggers:
        match logger:
            case WandbLogger(_experiment=experiment) if experiment is not None:
                _ = experiment.finish()
            case _:
                pass
