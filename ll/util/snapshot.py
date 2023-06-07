import importlib.util
import subprocess
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Sequence

log = getLogger(__name__)


@dataclass(kw_only=True)
class SnapshotInformation:
    snapshot_dir: Path
    moved_modules: dict[str, list[tuple[Path, Path]]]


def _copy(source: Path, location: Path):
    ignored_files = (
        subprocess.check_output(
            [
                "git",
                "-C",
                str(source),
                "ls-files",
                "--exclude-standard",
                "-oi",
                "--directory",
            ]
        )
        .decode("utf-8")
        .splitlines()
    )

    # run rsync with .git folder and `ignored_files` excluded
    _ = subprocess.run(
        [
            "rsync",
            "-a",
            "--exclude",
            ".git",
            *(f"--exclude={file}" for file in ignored_files),
            str(source),
            str(location),
        ],
        check=True,
    )


def snapshot_modules(
    base: str | Path,
    modules: Sequence[str],
    *,
    id: str | None = None,
    add_date_to_dir: bool = True,
    error_on_existing: bool = True,
):
    if not id:
        id = str(uuid.uuid4())

    snapshot_dir = Path(base)
    if add_date_to_dir:
        snapshot_dir = snapshot_dir / datetime.now().strftime("%Y-%m-%d")
    snapshot_dir = snapshot_dir / id
    snapshot_dir.mkdir(parents=True, exist_ok=not error_on_existing)

    log.critical(f"Snapshotting to {snapshot_dir}")

    moved_modules = defaultdict[str, list[tuple[Path, Path]]](list)
    for module in modules:
        spec = importlib.util.find_spec(module)
        if spec is None:
            log.warning(f"Module {module} not found")
            continue

        assert (
            spec.submodule_search_locations
            and len(spec.submodule_search_locations) == 1
        ), f"Could not find module {module} in a single location."
        location = Path(spec.submodule_search_locations[0])
        assert (
            location.is_dir()
        ), f"Module {module} has a non-directory location {location}"

        (*parent_modules, module_name) = module.split(".")

        destination = snapshot_dir
        for part in parent_modules:
            destination = destination / part
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "__init__.py").touch(exist_ok=True)

        _copy(location, destination)

        destination = destination / module_name
        log.info(f"Moved {location} to {destination} for {module=}")
        moved_modules[module].append((location, destination))

    return snapshot_dir
