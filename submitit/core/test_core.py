# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pylint: disable=redefined-outer-name
import contextlib
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union
from unittest.mock import patch

import pytest

from . import core, submission, utils


# pylint: disable=no-self-use
class MockedSubprocess:
    """Helper for mocking subprocess calls"""

    SACCT_HEADER = "JobID|State"
    SACCT_JOB = "{j}|{state}\n{j}.ext+|{state}\n{j}.0|{state}"

    def __init__(self, known_cmds: Sequence[str] = None) -> None:
        self.job_sacct: Dict[str, str] = {}
        self.last_job: str = ""
        self._subprocess_check_output = subprocess.check_output
        self.known_cmds = known_cmds or []
        self.job_count = 12

    def __call__(self, command: Sequence[str], **kwargs: Any) -> bytes:
        program = command[0]
        if program in ["sacct", "sbatch", "scancel"]:
            return getattr(self, program)(command[1:]).encode()
        elif program == "tail":
            return self._subprocess_check_output(command, **kwargs)
        else:
            raise ValueError(f'Unknown command to mock "{command}".')

    def sacct(self, _: Sequence[str]) -> str:
        return "\n".join(self.job_sacct.values())

    def sbatch(self, args: Sequence[str]) -> str:
        """Create a "RUNNING" job."""
        job_id = str(self.job_count)
        self.job_count += 1
        sbatch_file = Path(args[0])
        array = 0
        if sbatch_file.exists():
            array_lines = [l for l in sbatch_file.read_text().splitlines() if "--array" in l]
            if array_lines:
                # SBATCH --array=0-4%3
                array = int(array_lines[0].split("=0-")[-1].split("%")[0])
                array += 1
        self.set_job_state(job_id, "RUNNING", array)
        return f"Running job {job_id}\n"

    def scancel(self, _: Sequence[str]) -> str:
        # TODO:should we call set_job_state ?
        return ""

    def set_job_state(self, job_id: str, state: str, array: int = 0) -> None:
        self.job_sacct[job_id] = self._sacct(state, job_id, array)
        self.last_job = job_id

    def _sacct(self, state: str, job_id: str, array: int) -> str:
        if array == 0:
            lines = self.SACCT_JOB.format(j=job_id, state=state)
        else:
            lines = "\n".join(self.SACCT_JOB.format(j=f"{job_id}_{i}", state=state) for i in range(array))
        return "\n".join((self.SACCT_HEADER, lines))

    def which(self, name: str) -> Optional[str]:
        return "here" if name in self.known_cmds else None

    def mock_cmd_fn(self, *args, **_):
        # CommandFunction(cmd)() ~= subprocess.check_output(cmd)
        return lambda: self(*args)

    @contextlib.contextmanager
    def context(self) -> Iterator[None]:
        with patch("submitit.core.utils.CommandFunction", new=self.mock_cmd_fn):
            with patch("subprocess.check_output", new=self):
                with patch("shutil.which", new=self.which):
                    with patch("subprocess.check_call", new=self):
                        yield None

    @contextlib.contextmanager
    def job_context(self, job_id: str) -> Iterator[None]:
        with utils.environment_variables(
            _USELESS_TEST_ENV_VAR_="1", SUBMITIT_EXECUTOR="slurm", SLURM_JOB_ID=str(job_id)
        ):
            yield None


class FakeInfoWatcher(core.InfoWatcher):

    # pylint: disable=abstract-method
    def get_state(self, job_id: str, mode: str = "standard") -> str:
        return "running"


class FakeJob(core.Job[core.R]):

    watcher = FakeInfoWatcher()
    _cancel_at_deletion = False


class FakeExecutor(core.PicklingExecutor):

    job_class = FakeJob

    @property
    def _submitit_command_str(self) -> str:
        return "echo 1"

    def _num_tasks(self) -> int:
        return 1

    def _make_submission_file_text(self, command: str, uid: str) -> str:  # pylint: disable=unused-argument
        """Creates the text of a file which will be created and run
        for the submission (for slurm, this is sbatch file).
        """
        return command + "2"  # this makes "echo 12"

    def _make_submission_command(self, submission_file_path: Path) -> List[str]:
        """Create the submission command."""
        with submission_file_path.open("r") as f:
            text: str = f.read()
        return text.split()  # this makes ["echo", "12"]

    @staticmethod
    def _get_job_id_from_submission_command(string: Union[bytes, str]) -> str:
        return string if isinstance(string, str) else string.decode()  # this returns "12"


def _three_time(x: int) -> int:
    return 3 * x


def do_nothing(*args: Any, **kwargs: Any) -> int:
    print("my args", args, flush=True)
    print("my kwargs", kwargs, flush=True)
    if "sleep" in kwargs:
        print("Waiting", flush=True)
        time.sleep(int(kwargs["sleep"]))
    if kwargs.get("error", False):
        print("Raising", flush=True)
        raise ValueError("Too bad")
    print("Finishing", flush=True)
    return 12


def test_fake_job(tmp_path: Path) -> None:
    job: FakeJob[int] = FakeJob(job_id="12", folder=tmp_path)
    repr(job)
    assert not job.done(force_check=True)
    # logs
    assert job.stdout() is None
    assert job.stderr() is None
    with job.paths.stderr.open("w") as f:
        f.write("blublu")
    assert job.stderr() == "blublu"
    # result
    utils.cloudpickle_dump(("success", 12), job.paths.result_pickle)
    assert job.result() == 12
    # exception
    assert job.exception() is None
    utils.cloudpickle_dump(("error", "blublu"), job.paths.result_pickle)
    assert isinstance(job.exception(), Exception)
    with pytest.raises(core.utils.FailedJobError):
        job.result()


def test_fake_job_cancel_at_deletion(tmp_path: Path) -> None:
    job: FakeJob[Any] = FakeJob(job_id="12", folder=tmp_path).cancel_at_deletion()  # type: ignore
    with patch("subprocess.call", return_value=None) as mock:
        assert mock.call_count == 0
        del job
        assert mock.call_count == 1


def test_fake_executor(tmp_path: Path) -> None:
    executor = FakeExecutor(folder=tmp_path)
    job = executor.submit(_three_time, 8)
    assert job.job_id == "12"
    assert job.paths.submission_file.exists()
    with utils.environment_variables(_TEST_CLUSTER_="slurm", SLURM_JOB_ID=str(job.job_id)):
        submission.process_job(folder=job.paths.folder)
    assert job.result() == 24


def test_fake_executor_batch(tmp_path: Path) -> None:
    executor = FakeExecutor(folder=tmp_path)
    with executor.batch():
        job = executor.submit(_three_time, 8)
        assert isinstance(job, core.DelayedJob)
    assert isinstance(job, FakeJob)
    with executor.batch():  # make sure we can send a new batch
        job = executor.submit(_three_time, 8)
        assert isinstance(job, core.DelayedJob)
    assert isinstance(job, FakeJob)
    # bad update
    with pytest.raises(RuntimeError):
        with executor.batch():
            executor.update_parameters(blublu=12)
    # bad access
    with pytest.raises(AttributeError):
        with executor.batch():
            job = executor.submit(_three_time, 8)
            assert isinstance(job, core.DelayedJob)
            job.job_id  # pylint: disable=pointless-statement
        assert isinstance(job, core.DelayedJob)

    with executor.batch(allow_implicit_submissions=True):
        job = executor.submit(_three_time, 8)
        assert isinstance(job, core.DelayedJob)
        job.job_id  # pylint: disable=pointless-statement
        assert isinstance(job, FakeJob)
        assert not executor._delayed_batch

    # empty context
    with pytest.warns(RuntimeWarning):
        with executor.batch():
            pass
    # multi context
    with pytest.raises(RuntimeError):
        with executor.batch():
            with executor.batch():
                job = executor.submit(_three_time, 8)
                assert isinstance(job, core.DelayedJob)
            assert isinstance(job, FakeJob)


def test_unpickling_watcher_registration(tmp_path: Path) -> None:
    executor = FakeExecutor(folder=tmp_path)
    job = executor.submit(_three_time, 4)
    original_job_id = job._job_id
    job._job_id = "007"  # pylint: disable=attribute-defined-outside-init
    assert job.watcher._registered == {original_job_id}  # still holds the old job id
    pkl = pickle.dumps(job)
    newjob = pickle.loads(pkl)
    assert newjob.job_id == "007"
    assert newjob.watcher._registered == {original_job_id, "007"}


if __name__ == "__main__":
    args, kwargs = [], {}  # oversimplisitic parser
    for argv in sys.argv[1:]:
        if "=" in argv:
            key, val = argv.split("=")
            kwargs[key.strip("-")] = val
        else:
            args.append(argv)
    do_nothing(*args, **kwargs)
