# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import functools
import os
from pathlib import Path
from typing import Any, Tuple

import pytest

from ..core import utils
from ..core.core import Job
from ..core.job_environment import JobEnvironment
from . import debug


class CheckFunction:
    """Function used for checking that computations are correct"""

    def __init__(self, n: int) -> None:
        self.data1 = list(range(n))
        self.data2 = list(range(10, 10 + n))

    def __call__(self, x: float, y: float) -> float:
        assert x in self.data1
        assert y in self.data2
        return x + y


def test_debug_job(tmp_path: Path) -> None:
    def func(p: int) -> int:
        return 2 * p

    executor = debug.DebugExecutor(tmp_path)
    job = executor.submit(func, 4)
    assert job.result() == 8
    with executor.batch():
        job2 = executor.submit(func, 5)
    assert job2.result() == 10
    # Check that job results are cached.
    job2.submission().function = None  # type: ignore
    assert job2.result() == 10


def test_debug_map_array(tmp_path: Path) -> None:
    g = CheckFunction(5)
    executor = debug.DebugExecutor(tmp_path)
    jobs = executor.map_array(g, g.data1, g.data2)
    print(type(jobs[0]))
    print(jobs)
    assert list(map(g, g.data1, g.data2)) == [j.result() for j in jobs]


def test_debug_submit_array(tmp_path: Path) -> None:
    g = CheckFunction(5)
    executor = debug.DebugExecutor(tmp_path)
    fns = [functools.partial(g, x, y) for x, y in zip(g.data1, g.data2)]
    jobs = executor.submit_array(fns)
    assert list(map(g, g.data1, g.data2)) == [j.result() for j in jobs]


def test_debug_error(tmp_path: Path) -> None:
    def failing_job() -> None:
        raise Exception("Failed on purpose")

    executor = debug.DebugExecutor(tmp_path)
    job = executor.submit(failing_job)
    exception = job.exception()
    assert isinstance(exception, Exception)
    message = exception.args[0]
    assert "Failed on purpose" in message


def f_42() -> int:
    return 42


def test_debug_triggered(tmp_path: Path) -> None:
    def get_result(job: Job) -> Tuple[bool, Any]:
        assert isinstance(job, debug.DebugJob)
        return (job._submission._done, job._submission._result)

    executor = debug.DebugExecutor(tmp_path)
    for trigger in ("wait", "done", "exception", "results"):
        job = executor.submit(f_42)
        assert job.state == "QUEUED"
        assert get_result(job) == (False, None)
        getattr(job, trigger)()
        assert job.state == "DONE"
        assert get_result(job) == (True, 42)


def test_cancel(tmp_path: Path) -> None:
    executor = debug.DebugExecutor(tmp_path)
    job = executor.submit(f_42)
    assert job.state == "QUEUED"
    job.cancel()
    assert job.state == "CANCELLED"
    with pytest.raises(utils.UncompletedJobError, match="was cancelled"):
        job.result()


def test_job_environment(tmp_path: Path) -> None:
    executor = debug.DebugExecutor(tmp_path)

    def use_env():
        env = JobEnvironment()
        assert env.num_nodes == 1
        assert env.num_tasks == 1
        assert env.node == 0
        assert env.global_rank == 0
        assert env.local_rank == 0
        assert "DEBUG" in env.job_id

    job = executor.submit(use_env)
    job.result()
    # Check that we clean up the env after us.
    assert "SUBMITIT_DEBUG_JOB_ID" not in os.environ
