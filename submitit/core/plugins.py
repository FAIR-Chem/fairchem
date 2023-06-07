# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import functools
import os
from typing import TYPE_CHECKING, List, Mapping, Tuple, Type

from ..core import logger

if TYPE_CHECKING:
    # Breaks the import cycle
    from ..core.core import Executor
    from ..core.job_environment import JobEnvironment


@functools.lru_cache()
def _get_plugins() -> Tuple[List[Type["Executor"]], List["JobEnvironment"]]:
    # pylint: disable=cyclic-import,import-outside-toplevel
    # Load dynamically to avoid import cycle
    # pkg_resources goes through all modules on import.
    import pkg_resources

    from ..local import debug, local
    from ..slurm import slurm

    # TODO: use sys.modules.keys() and importlib.resources to find the files
    # We load both kind of entry points at the same time because we have to go through all module files anyway.
    executors: List[Type["Executor"]] = [slurm.SlurmExecutor, local.LocalExecutor, debug.DebugExecutor]
    job_envs = [slurm.SlurmJobEnvironment(), local.LocalJobEnvironment(), debug.DebugJobEnvironment()]
    for entry_point in pkg_resources.iter_entry_points("submitit"):
        if entry_point.name not in ("executor", "job_environment"):
            logger.warning(f"Found unknown entry point in package {entry_point.module_name}: {entry_point}")
            continue

        try:
            # call `load` rather than `resolve`.
            # `load` also checks the module and its dependencies are correctly installed.
            cls = entry_point.load()
        except Exception as e:
            # This may happen if the plugin haven't been correctly installed
            logger.exception(f"Failed to load submitit plugin '{entry_point.module_name}': {e}")
            continue

        if entry_point.name == "executor":
            executors.append(cls)
        else:
            try:
                job_env = cls()
            except Exception as e:
                logger.exception(
                    f"Failed to init JobEnvironment '{cls.name}' ({cls}) from submitit plugin '{entry_point.module_name}': {e}"
                )
                continue
            job_envs.append(job_env)

    return (executors, job_envs)


@functools.lru_cache()
def get_executors() -> Mapping[str, Type["Executor"]]:
    # TODO: check collisions between executor names
    return {ex.name(): ex for ex in _get_plugins()[0]}


def get_job_environment() -> "JobEnvironment":
    # Don't cache this function. It makes testing harder.
    # The slow part is the plugin discovery anyway.
    envs = get_job_environments()
    # bypassing can be helful for testing
    if "_TEST_CLUSTER_" in os.environ:
        c = os.environ["_TEST_CLUSTER_"]
        assert c in envs, f"Unknown $_TEST_CLUSTER_='{c}', available: {envs.keys()}."
        return envs[c]
    for env in envs.values():
        # TODO? handle the case where several envs are valid
        if env.activated():
            return env
    raise RuntimeError(
        f"Could not figure out which environment the job is runnning in. Known environments: {', '.join(envs.keys())}."
    )


@functools.lru_cache()
def get_job_environments() -> Mapping[str, "JobEnvironment"]:
    return {env.name(): env for env in _get_plugins()[1]}
