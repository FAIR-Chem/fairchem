# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import functools
import os
import re
import signal
import sys
import time
from pathlib import Path

import pytest

from .. import helpers
from ..core import job_environment, test_core, utils
from . import local, test_debug


def test_local_job(tmp_path: Path) -> None:
    def func(p: int) -> int:
        job_env = job_environment.JobEnvironment()
        return p * job_env.local_rank

    executor = local.LocalExecutor(tmp_path)
    executor.update_parameters(tasks_per_node=3, nodes=1)
    job1 = executor.submit(func, 1)

    executor.update_parameters(tasks_per_node=1)

    with executor.batch():
        with pytest.raises(RuntimeError, match="with executor.batch"):
            executor.update_parameters(tasks_per_node=1)
        job2 = executor.submit(func, 2)
    assert job1.results() == [0, 1, 2]
    assert job1.task(1).result() == 1
    assert job1.task(2).result() == 2
    assert job1.task(2).result() == 2
    assert job1.exception() is None
    assert job1.done()

    with pytest.raises(ValueError, match="must be between 0 and 2"):
        job1.task(4).result()

    assert job2.results() == [0]
    assert job2.task(0).result() == 0
    # single task job is a regular job
    assert job2.task(0) is job2
    assert job2.done()


def test_local_map_array(tmp_path: Path) -> None:
    g = test_debug.CheckFunction(5)
    executor = local.LocalExecutor(tmp_path)
    jobs = executor.map_array(g, g.data1, g.data2)
    assert list(map(g, g.data1, g.data2)) == [j.result() for j in jobs]


def test_local_submit_array(tmp_path: Path) -> None:
    g = test_debug.CheckFunction(5)
    executor = local.LocalExecutor(tmp_path)
    fns = [functools.partial(g, x, y) for x, y in zip(g.data1, g.data2)]
    jobs = executor.submit_array(fns)
    assert list(map(g, g.data1, g.data2)) == [j.result() for j in jobs]


def test_local_error(tmp_path: Path) -> None:
    def failing_job() -> None:
        raise Exception("Failed on purpose")

    executor = local.LocalExecutor(tmp_path)
    job = executor.submit(failing_job)
    exception = job.exception()
    assert isinstance(exception, utils.FailedJobError)
    traceback = exception.args[0]
    assert "Traceback" in traceback
    assert "Failed on purpose" in traceback


def test_pickle_output_from_main(tmp_path: Path) -> None:
    class MyClass:
        pass

    executor = local.LocalExecutor(tmp_path)
    job = executor.submit(MyClass.__call__)
    assert isinstance(job.result(), MyClass)


def test_get_first_task_error(tmp_path: Path) -> None:
    def flaky() -> None:
        job_env = job_environment.JobEnvironment()
        if job_env.local_rank > 0:
            raise Exception(f"Failed on purpose: {job_env.local_rank}")

    executor = local.LocalExecutor(tmp_path)
    executor.update_parameters(tasks_per_node=3, nodes=1)
    job = executor.submit(flaky)
    exception = job.exception()
    assert isinstance(exception, utils.FailedJobError)
    traceback = exception.args[0]
    assert "Traceback" in traceback
    assert "Failed on purpose: 1" in traceback


def test_stdout(tmp_path: Path) -> None:
    def hello() -> None:
        job_env = job_environment.JobEnvironment()
        print("hello from", job_env.local_rank)
        print("bye from", job_env.local_rank, file=sys.stderr)

    executor = local.LocalExecutor(tmp_path)
    executor.update_parameters(tasks_per_node=2, nodes=1)
    job = executor.submit(hello)

    job.wait()
    stdout = job.stdout()
    assert stdout is not None
    assert "hello from 0\n" in stdout
    assert "hello from 1\n" in stdout

    stderr = job.stderr()
    assert stderr is not None
    assert "bye from 0\n" in stderr
    assert "bye from 1\n" in stderr


def test_killed(tmp_path: Path) -> None:
    def failing_job() -> None:
        time.sleep(120)
        raise Exception("Failed on purpose")

    executor = local.LocalExecutor(tmp_path)
    job = executor.submit(failing_job)
    assert job.state == "RUNNING"
    job._process.send_signal(signal.SIGKILL)  # type: ignore
    time.sleep(1)
    assert job.state == "INTERRUPTED"


@pytest.mark.skipif(not os.environ.get("SUBMITIT_SLOW_TESTS", False), reason="slow")  # type: ignore
def test_long_running_job(tmp_path: Path) -> None:
    def f(x: int, y: int, sleep: int = 120) -> int:
        time.sleep(sleep)
        return x + y

    executor = local.LocalExecutor(tmp_path)
    executor.update_parameters(timeout_min=5)
    job = executor.submit(f, 40, 2)
    assert job.result() == 42


def test_requeuing(tmp_path: Path) -> None:
    func = helpers.FunctionSequence(verbose=True)
    for x in range(20):
        func.add(test_core.do_nothing, x=x, sleep=1)
    executor = local.LocalExecutor(tmp_path, max_num_timeout=1)
    executor.update_parameters(timeout_min=3 / 60, signal_delay_s=1)
    job = executor.submit(func)
    job.wait()
    stdout = job.stdout()
    assert stdout is not None
    match = re.search(r"Starting from [123]/20", stdout)
    assert match, f"Should have resumed from a checkpoint:\n{stdout}"
    assert "timed-out too many times" in stdout, f"Unexpected stdout:\n{stdout}"
    assert "(0 remaining timeouts)" in stdout, f"Unexpected stdout:\n{stdout}"


def test_custom_checkpoint(tmp_path: Path) -> None:
    class Slacker(helpers.Checkpointable):
        def __call__(self, slack: bool = True):
            if slack:
                print("Slacking", flush=True)
                time.sleep(10)
                raise Exception("I really don't want to work")
            print("Working hard", flush=True)
            return "worked hard"

        def __submitit_checkpoint__(self, slack: bool = True):
            if slack:
                print("Interrupted while slacking. I won't slack next time.", flush=True)
            return utils.DelayedSubmission(self, slack=False)

    executor = local.LocalExecutor(tmp_path, max_num_timeout=1)
    executor.update_parameters(timeout_min=2 / 60, signal_delay_s=1)
    job = executor.submit(Slacker(True))
    job.wait()
    stdout = job.stdout()
    assert stdout
    assert "I won't slack next time." in stdout


def test_make_subprocess(tmp_path: Path) -> None:
    process = local.start_controller(
        tmp_path, "python -c 'import os;print(os.environ[\"SUBMITIT_LOCAL_JOB_ID\"])'", timeout_min=1
    )
    paths = utils.JobPaths(tmp_path, str(process.pid), 0)
    pg = process.pid
    process.wait()
    stdout = paths.stdout.read_text()
    stderr = paths.stderr.read_text()
    assert stdout and int(stdout.strip()) == pg, f"PID link is broken (stderr: {stderr})"


def test_cancel(tmp_path: Path) -> None:
    executor = local.LocalExecutor(tmp_path)
    job = executor.submit(time.sleep, 10)
    assert job.state == "RUNNING"
    job.cancel()
    time.sleep(0.1)
    # Note: with local job we don't have a precise status.
    assert job.state == "INTERRUPTED"

    job = executor.submit(time.sleep, 10)
    process = job._process  # type: ignore
    job.cancel_at_deletion()
    assert job.state == "RUNNING"
    assert process.poll() is None
    del job
    time.sleep(0.1)
    assert process.poll() == -2


def f66(x: int, y: int = 0) -> int:  # pylint: disable=unused-argument
    return 66


def test_load_submission(tmp_path: Path) -> None:
    """Check we can load submission just from a path and job id."""
    executor = local.LocalExecutor(tmp_path)
    job = executor.submit(f66, 67, y=68)

    submission = local.LocalJob(tmp_path, job.job_id).submission()
    # It's important that f66 isn't a local function for the equality to work
    assert submission.function is f66
    assert submission.args == (67,)
    assert submission.kwargs == {"y": 68}
    # Loading submission doesn't evaluate them.
    assert submission._result is None


def test_weird_dir(weird_tmp_path: Path) -> None:
    executor = local.LocalExecutor(weird_tmp_path / "%j")
    executor.submit(f66, 67, 68).result()
