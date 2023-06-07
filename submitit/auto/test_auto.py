# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
from pathlib import Path

import pytest

from ..local import debug
from ..slurm import test_slurm
from . import auto


def test_slurm_executor(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(debug.DebugExecutor, "_valid_parameters", lambda: {"blabla"})
    with test_slurm.mocked_slurm():
        executor = auto.AutoExecutor(folder=tmp_path)
    assert executor.cluster == "slurm"

    # local_xxx parameter is ignored
    executor.update_parameters(mem_gb=2, name="machin", debug_blabla="blublu")
    params = executor._executor.parameters
    assert params == {"mem": "2GB", "job_name": "machin"}

    # shared parameter with wrong type
    with pytest.raises(AssertionError):
        executor.update_parameters(mem_gb="2.0GB")  # should be int
    # unknown shared parameter
    with pytest.raises(NameError):
        executor.update_parameters(blublu=2.0)
    # unknown slurm parameter
    with pytest.raises(NameError):
        executor.update_parameters(slurm_host_filter="blublu")
    # check that error message contains all
    with pytest.raises(NameError, match=r"debug_blublu.*\n.*local_num_threads"):
        executor.update_parameters(debug_blublu=2.0, local_num_threads=4)


def test_local_executor(tmp_path: Path) -> None:
    with test_slurm.mocked_slurm():
        executor = auto.AutoExecutor(folder=tmp_path, cluster="local")
    assert executor.cluster == "local"
    executor.update_parameters(local_cpus_per_task=2)


def test_executor_argument(tmp_path: Path) -> None:
    with test_slurm.mocked_slurm():
        executor = auto.AutoExecutor(folder=tmp_path, slurm_max_num_timeout=22)
    assert getattr(executor._executor, "max_num_timeout", None) == 22

    # Local executor
    executor = auto.AutoExecutor(folder=tmp_path, cluster="local", slurm_max_num_timeout=22)
    assert getattr(executor._executor, "max_num_timeout", None) != 22


def test_executor_unknown_argument(tmp_path: Path) -> None:
    with test_slurm.mocked_slurm():
        with pytest.raises(TypeError):
            auto.AutoExecutor(folder=tmp_path, slurm_foobar=22)


def test_executor_deprecated_arguments(tmp_path: Path) -> None:
    with test_slurm.mocked_slurm():
        with pytest.warns(UserWarning, match="slurm_max_num_timeout"):
            auto.AutoExecutor(folder=tmp_path, max_num_timeout=22)


def test_deprecated_argument(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(debug.DebugExecutor, "_valid_parameters", lambda: {"blabla"})
    with test_slurm.mocked_slurm():
        executor = auto.AutoExecutor(folder=tmp_path)
    assert executor.cluster == "slurm"

    # debug 'blabla' parameter is ignored
    with pytest.warns(UserWarning, match=r"blabla.*debug_blabla"):
        executor.update_parameters(mem_gb=2, blabla="blublu")


def test_overriden_arguments(tmp_path: Path) -> None:
    with test_slurm.mocked_slurm():
        slurm_ex = auto.AutoExecutor(folder=tmp_path, cluster="slurm")

    slurm_ex.update_parameters(
        timeout_min=60, slurm_timeout_min=120, tasks_per_node=2, slurm_ntasks_per_node=3
    )
    slurm_params = slurm_ex._executor.parameters
    # slurm use time
    assert slurm_params == {"time": 120, "ntasks_per_node": 3}

    # others use timeout_min
    local_ex = auto.AutoExecutor(folder=tmp_path, cluster="local")
    local_ex.update_parameters(timeout_min=60, slurm_time=120)


def test_auto_batch_watcher(tmp_path: Path) -> None:
    with test_slurm.mocked_slurm():
        executor = auto.AutoExecutor(folder=tmp_path)
        with executor.batch():
            job = executor.submit(print, "hi")
        assert not job.done()


def test_redirect_stdout_stderr(executor) -> None:
    def log_to_stderr_and_stdout():
        print("hello")
        print("world", file=sys.stderr)

    executor.update_parameters(stderr_to_stdout=True)
    job = executor.submit(log_to_stderr_and_stdout)
    job.wait()
    assert job.stderr() is None
    stdout = job.stdout()
    assert "hello" in stdout
    assert "world" in stdout

    executor.update_parameters(stderr_to_stdout=False)
    job = executor.submit(log_to_stderr_and_stdout)
    job.wait()
    assert "world" in job.stderr()
    assert "hello" in job.stdout()
