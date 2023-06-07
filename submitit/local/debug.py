# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import os
import typing as tp
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..core.core import Executor, InfoWatcher, Job, R
from ..core.job_environment import JobEnvironment
from ..core.utils import DelayedSubmission, UncompletedJobError


class DebugInfoWatcher(InfoWatcher):
    # pylint: disable=abstract-method
    def register_job(self, job_id: str) -> None:
        pass


class DebugJobEnvironment(JobEnvironment):
    _env = {
        "job_id": "SUBMITIT_DEBUG_JOB_ID",
        # We don't set those, and rely on the default values from JobEnvironment
        "num_nodes": "SUBMITIT_DEBUG_NOT_SET",
        "num_tasks": "SUBMITIT_DEBUG_NOT_SET",
        "node": "SUBMITIT_DEBUG_NOT_SET",
        "global_rank": "SUBMITIT_DEBUG_NOT_SET",
        "local_rank": "SUBMITIT_DEBUG_NOT_SET",
    }

    def activated(self) -> bool:
        return "SUBMITIT_DEBUG_JOB_ID" in os.environ

    def _requeue(self, countdown: int) -> None:
        pass


# pylint in python 3.6 is confused by generics.
class DebugJob(Job[R]):
    watcher = DebugInfoWatcher()

    def __init__(self, folder: Path, submission: DelayedSubmission) -> None:
        job_id = f"DEBUG_{id(submission)}"
        super().__init__(folder=folder, job_id=job_id)
        self._submission = submission
        self.cancelled = False
        self.environ = dict(os.environ)
        self.environ["SUBMITIT_DEBUG_JOB_ID"] = self.job_id

    def submission(self) -> DelayedSubmission:
        return self._submission

    @property
    def num_tasks(self) -> int:
        return 1

    def cancel(self, check: bool = True) -> None:  # pylint: disable=unused-argument
        self.cancelled = True

    def _check_not_cancelled(self) -> None:
        if self.cancelled:
            raise UncompletedJobError(f"Job {self} was cancelled.")

    def results(self) -> List[R]:
        self._check_not_cancelled()
        if self._submission.done():
            return [self._submission._result]

        environ_backup = dict(os.environ)
        # Restore os.environ from job creation time.
        os.environ.clear()
        os.environ.update(self.environ)

        root_logger = logging.getLogger("")
        self.paths.stdout.parent.mkdir(exist_ok=True, parents=True)
        stdout_handler = logging.FileHandler(self.paths.stdout)
        stdout_handler.setLevel(logging.DEBUG)
        stderr_handler = logging.FileHandler(self.paths.stderr)
        stderr_handler.setLevel(logging.WARNING)
        root_logger.addHandler(stdout_handler)
        root_logger.addHandler(stderr_handler)
        root_logger.warning(
            f"Logging is written both to stderr/stdout and to {self.paths.stdout}/err. "
            "But call to print will only appear in the console."
        )
        try:
            return [self._submission.result()]
        except Exception as e:
            print(e)
            # Try to mimic `breakpoint()` behavior
            # pylint: disable=import-outside-toplevel
            if os.environ.get("PYTHONBREAKPOINT", "").startswith("ipdb"):
                import ipdb  # pylint: disable=import-error

                ipdb.post_mortem()
            else:
                import pdb

                pdb.post_mortem()
            raise
        finally:
            os.environ.clear()
            os.environ.update(environ_backup)
            root_logger.removeHandler(stdout_handler)
            root_logger.removeHandler(stderr_handler)

    def exception(self) -> Optional[BaseException]:  # type: ignore
        self._check_not_cancelled()
        try:
            self._submission.result()
            return None
        except Exception as e:
            # Note that we aren't wrapping the error contrary to what is done in
            # other Executors. It makes the stacktrace smaller and debugging easier.
            return e

    def wait(self) -> None:
        # forces execution.
        self.results()

    def done(self, force_check: bool = False) -> bool:  # pylint: disable=unused-argument
        # forces execution, in case the client is waiting on it to become True.
        self.results()
        return self._submission.done()

    @property
    def state(self) -> str:
        if self._submission.done():
            return "DONE"
        if self.cancelled:
            return "CANCELLED"
        return "QUEUED"

    def get_info(self, mode: str = "force") -> Dict[str, str]:  # pylint: disable=unused-argument
        return {"STATE": self.state}

    def __del__(self) -> None:
        # Skip parent code
        return


class DebugExecutor(Executor):

    job_class = DebugJob

    def __init__(self, folder: Union[str, Path]):
        super().__init__(folder)

    def _internal_process_submissions(
        self, delayed_submissions: tp.List[DelayedSubmission]
    ) -> tp.List[Job[tp.Any]]:
        return [DebugJob(self.folder, ds) for ds in delayed_submissions]
