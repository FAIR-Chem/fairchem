import getpass
from datetime import timedelta
from logging import getLogger
from pathlib import Path

from submitit import AutoExecutor

from .snapshot import snapshot_modules

log = getLogger(__name__)


def create_executor(
    *,
    tasks_per_node: int,
    cpus_per_task: int,
    gpus_per_task: int,
    nodes: int,
    partition: str,
    timeout: timedelta = timedelta(hours=72),
    memory: int = 480,
    email: str | None = None,
    constraints: list[str] | None = None,
    volta16gb: bool | None = None,
    volta32gb: bool | None = None,
    slurm_additional_parameters: dict[str, str] | None = None,
    slurm_setup: list[str] | None = None,
    snapshot: bool | Path,
    snapshot_base: Path | None = None,
    env: dict[str, str] | None = None,
    job_name: str = "ll",
    snapshot_env_name: str = "LL_SNAPSHOT",
):
    if volta16gb and volta32gb:
        raise ValueError("Cannot have both volta16gb and volta32gb")
    elif volta16gb is None and volta32gb is None:
        volta16gb = False
        volta32gb = True

    if volta16gb is None:
        volta16gb = False
    if volta32gb is None:
        volta32gb = False

    if snapshot_base is None:
        current_user = getpass.getuser()
        snapshot_base = Path(f"/checkpoint/{current_user}/ll_snapshots/")

    if snapshot is True:
        snapshot = snapshot_modules(snapshot_base, ["st", "ll", "submitit"]).absolute()

    base_path = Path(".") / "slurm_logs"
    base_path.mkdir(exist_ok=True, parents=True)

    additional_parameters = {}
    if not constraints:
        constraints = []
    if email:
        additional_parameters.update({"mail_user": email, "mail_type": "FAIL"})
    if volta16gb:
        # additional_parameters.update({"constraint": "volta16gb"})
        constraints.append("volta16gb")
    if volta32gb:
        # additional_parameters.update({"constraint": "volta32gb"})
        constraints.append("volta32gb")
    if slurm_additional_parameters:
        additional_parameters.update(slurm_additional_parameters)

    # add constraints from slurm_additional_parameters
    if (constraint := additional_parameters.pop("constraint", None)) is not None:
        constraints.append(constraint)

    # remove duplicates
    constraints = list(set(constraints))
    if constraints:
        additional_parameters.update({"constraint": ",".join(constraints)})

    setup = []
    if env:
        setup.extend(f"export {k}={v}" for k, v in env.items())
    if slurm_setup:
        setup.extend(slurm_setup)
    if snapshot:
        snapshot_str = str(snapshot.resolve().absolute())
        setup.append(f"export {snapshot_env_name}={snapshot_str}")
        setup.append(f"export PYTHONPATH={snapshot_str}:$PYTHONPATH")

    executor = AutoExecutor(folder=base_path / "%j")
    executor.update_parameters(
        name=job_name,
        mem_gb=memory,
        timeout_min=int(timeout.total_seconds() / 60),
        cpus_per_task=cpus_per_task,
        tasks_per_node=tasks_per_node,
        nodes=nodes,
        slurm_gpus_per_task=gpus_per_task,
        slurm_partition=partition,
        slurm_additional_parameters=additional_parameters,
        slurm_setup=setup,
    )
    return executor
