# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from pathlib import Path

import pytest

from . import submission, utils
from .test_core import FakeExecutor, _three_time


@pytest.mark.asyncio
async def test_result(tmp_path: Path, event_loop):
    executor = FakeExecutor(folder=tmp_path)
    job = executor.submit(_three_time, 8)
    result_task = event_loop.create_task(job.awaitable().result())
    with utils.environment_variables(_TEST_CLUSTER_="slurm", SLURM_JOB_ID=str(job.job_id)):
        submission.process_job(folder=job.paths.folder)
    result = await result_task
    assert result == 24


@pytest.mark.asyncio
async def test_results_single(tmp_path: Path, event_loop):
    executor = FakeExecutor(folder=tmp_path)
    job = executor.submit(_three_time, 8)
    result_task = event_loop.create_task(job.awaitable().results())
    with utils.environment_variables(_TEST_CLUSTER_="slurm", SLURM_JOB_ID=str(job.job_id)):
        submission.process_job(folder=job.paths.folder)
    result = await result_task
    assert result == [24]


@pytest.mark.asyncio
async def test_results_ascompleted_single(tmp_path: Path):
    executor = FakeExecutor(folder=tmp_path)
    job = executor.submit(_three_time, 8)
    with utils.environment_variables(_TEST_CLUSTER_="slurm", SLURM_JOB_ID=str(job.job_id)):
        submission.process_job(folder=job.paths.folder)
    count = 0
    for aws in job.awaitable().results_as_completed():
        result = await aws
        count += 1
        assert result == 24
    assert count == 1
