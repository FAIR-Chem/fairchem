# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import re
from pathlib import Path
from typing import Any, Iterator

import pkg_resources
import pytest

from . import core, plugins
from .job_environment import JobEnvironment


@pytest.mark.parametrize("env", plugins.get_job_environments().values())
def test_env(env: JobEnvironment) -> None:
    assert isinstance(env, JobEnvironment)
    # We are not inside a submitit job
    assert not env.activated()
    assert type(env)._requeue is not JobEnvironment._requeue, "_requeue need to be overridden"


@pytest.mark.parametrize("ex", plugins.get_executors().values())
def test_executors(ex: core.Executor) -> None:
    assert isinstance(ex, type)
    assert issubclass(ex, core.Executor)
    assert ex.affinity() >= -1


def test_finds_default_environments() -> None:
    envs = plugins.get_job_environments()
    assert len(envs) >= 3
    assert "slurm" in envs
    assert "local" in envs
    assert "debug" in envs


def test_finds_default_executors() -> None:
    ex = plugins.get_executors()
    assert len(ex) >= 3
    assert "slurm" in ex
    assert "local" in ex
    assert "debug" in ex


def test_job_environment_works(monkeypatch):
    monkeypatch.setenv("_TEST_CLUSTER_", "slurm")
    env = plugins.get_job_environment()
    assert env.cluster == "slurm"
    assert type(env).__name__ == "SlurmJobEnvironment"

    env2 = JobEnvironment()
    assert env2.cluster == "slurm"
    assert type(env2).__name__ == "SlurmJobEnvironment"


def test_job_environment_raises_outside_of_job() -> None:
    with pytest.raises(RuntimeError, match=r"which environment.*slurm.*local.*debug"):
        plugins.get_job_environment()


class PluginCreator:
    def __init__(self, tmp_path: Path, monkeypatch):
        self.tmp_path = tmp_path
        self.monkeypatch = monkeypatch

    def add_plugin(self, name: str, entry_points: str, init: str):
        plugin = self.tmp_path / name
        plugin.mkdir(mode=0o777)
        plugin_egg = plugin.with_suffix(".egg-info")
        plugin_egg.mkdir(mode=0o777)

        (plugin_egg / "entry_points.txt").write_text(entry_points)
        (plugin / "__init__.py").write_text(init)

        # also fix pkg_resources since it already has loaded old packages in other tests.
        working_set = pkg_resources.WorkingSet([str(self.tmp_path)])
        self.monkeypatch.setattr(pkg_resources, "iter_entry_points", working_set.iter_entry_points)

    def __enter__(self) -> None:
        _clear_plugin_cache()
        self.monkeypatch.syspath_prepend(self.tmp_path)

    def __exit__(self, *exception: Any) -> None:
        _clear_plugin_cache()


def _clear_plugin_cache() -> None:
    plugins._get_plugins.cache_clear()
    plugins.get_executors.cache_clear()


@pytest.fixture(name="plugin_creator")
def _plugin_creator(tmp_path: Path, monkeypatch) -> Iterator[PluginCreator]:
    creator = PluginCreator(tmp_path, monkeypatch)
    with creator:
        yield creator


def test_find_good_plugin(plugin_creator: PluginCreator) -> None:
    plugin_creator.add_plugin(
        "submitit_good",
        entry_points="""[submitit]
executor = submitit_good:GoodExecutor
job_environment = submitit_good:GoodJobEnvironment
unsupported_key = submitit_good:SomethingElse
""",
        init="""
import submitit

class GoodExecutor(submitit.Executor):
    pass

class GoodJobEnvironment:
    pass
""",
    )

    executors = plugins.get_executors().keys()
    # Only the plugins declared with plugin_creator are visible.
    assert set(executors) == {"good", "slurm", "local", "debug"}


def test_skip_bad_plugin(caplog, plugin_creator: PluginCreator) -> None:
    caplog.set_level(logging.WARNING, logger="submitit")
    plugin_creator.add_plugin(
        "submitit_bad",
        entry_points="""[submitit]
executor = submitit_bad:NonExisitingExecutor
job_environment = submitit_bad:BadEnvironment
unsupported_key = submitit_bad:SomethingElse
""",
        init="""
import submitit

class BadEnvironment:
    name = "bad"

    def __init__(self):
        raise Exception("this is a bad environment")
""",
    )

    executors = plugins.get_executors().keys()
    assert {"slurm", "local", "debug"} == set(executors)
    assert "bad" not in executors
    expected = [
        (logging.ERROR, r"'submitit_bad'.*no attribute 'NonExisitingExecutor'"),
        (logging.ERROR, r"'submitit_bad'.*this is a bad environment"),
        (logging.WARNING, "unsupported_key = submitit_bad:SomethingElse"),
    ]
    assert len(caplog.records) == len(expected)
    for record, ex_record in zip(caplog.records, expected):
        assert record.name == "submitit"
        assert record.levelno == ex_record[0]
        assert re.search(ex_record[1], record.getMessage())
