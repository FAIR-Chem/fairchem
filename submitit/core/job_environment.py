# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import signal
import socket
import sys
import time
import types
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Sequence

from . import logger, utils
from .utils import DelayedSubmission, JobPaths

_PREEMPT_SIG_ENV = "SUBMITIT_PREEMPT_SIGNAL"


class JobEnvironment:
    """Describe the environment inside which the job is running.
    This includes job id, number of GPUs available, ...

    This class can only be instantiated from a running submitit job.

    @plugin-dev: default implementation look for information into environment variables.
    Override _env to map environment variable to each property.
    """

    # preemption signal uses USR2 as default, but this behavior
    # can be overiden (eg: export SUBMITIT_PREEMPT_SIGNAL=USR2)
    # CAUTION: NCCL may catch USR1 so it should be avoided
    USR_SIG = os.environ.get(_PREEMPT_SIG_ENV, "USR1")
    _env: ClassVar[Dict[str, str]] = {}

    def __new__(cls, *args: Any) -> "JobEnvironment":
        if cls is not JobEnvironment:
            return super().__new__(cls, *args)  # type: ignore

        from . import plugins  # pylint: disable=cyclic-import,import-outside-toplevel

        return plugins.get_job_environment()

    def __init__(self) -> None:
        self.cluster = self.name()

    @classmethod
    def name(cls) -> str:
        n = cls.__name__
        if n.endswith("JobEnvironment"):
            n = n[: -len("JobEnvironment")]
        return n.lower()

    @property
    def paths(self) -> JobPaths:
        """Provides the paths used by submitit, including
        stdout, stderr, submitted_pickle and folder.
        """
        folder = os.environ["SUBMITIT_FOLDER"]
        return JobPaths(folder, job_id=self.job_id, task_id=self.global_rank)

    def activated(self) -> bool:
        """Tests if we are running inside this environment.

        @plugin-dev: assumes that the SUBMITIT_EXECUTOR variable has been
        set to the executor name
        """
        return os.environ.get("SUBMITIT_EXECUTOR", "") == self.name()

    @property
    def hostname(self) -> str:
        return socket.gethostname()

    @property
    def hostnames(self) -> Sequence[str]:
        return [self.hostname]

    @property
    def job_id(self) -> str:
        if self.array_job_id:
            return f"{self.array_job_id}_{self.array_task_id}"
        else:
            return self.raw_job_id

    @property
    def raw_job_id(self) -> str:
        return os.environ[self._env["job_id"]]

    @property
    def array_job_id(self) -> Optional[str]:
        n = "array_job_id"
        return None if n not in self._env else os.environ.get(self._env[n], None)

    @property
    def array_task_id(self) -> Optional[str]:
        n = "array_task_id"
        return None if n not in self._env else os.environ.get(self._env[n], None)

    @property
    def num_tasks(self) -> int:
        """Total number of tasks for the job"""
        return int(os.environ.get(self._env["num_tasks"], 1))

    @property
    def num_nodes(self) -> int:
        """Total number of nodes for the job"""
        return int(os.environ.get(self._env["num_nodes"], 1))

    @property
    def node(self) -> int:
        """Id of the current node"""
        return int(os.environ.get(self._env["node"], 0))

    @property
    def global_rank(self) -> int:
        """Global rank of the task"""
        return int(os.environ.get(self._env["global_rank"], 0))

    @property
    def local_rank(self) -> int:
        """Local rank of the task, ie on the current node."""
        return int(os.environ.get(self._env["local_rank"], 0))

    def __repr__(self) -> str:
        # should look like this:
        # JobEnvironment(job_id=17015819, hostname=learnfair0218, local_rank=2(3), node=1(2), global_rank=5(6))
        info = [f"{n}={getattr(self, n)}" for n in ("job_id", "hostname")]
        names = ("local_rank", "node", "global_rank")
        totals = [self.num_tasks // self.num_nodes, self.num_nodes, self.num_tasks]
        info += [f"{n}={getattr(self, n)}({t})" for n, t in zip(names, totals)]
        info_str = ", ".join(info)
        return f"JobEnvironment({info_str})"

    @classmethod
    def _usr_sig(cls) -> Any:
        name = "SIG" + cls.USR_SIG
        out = getattr(signal, name, None)
        if out is None:
            raise RuntimeError(
                f"Unknown signal {name}, you may need to unset or update env var {_PREEMPT_SIG_ENV} (Eg: USR2)"
            )
        return out

    def _handle_signals(self, paths: JobPaths, submission: DelayedSubmission) -> None:
        """Set up signals handler for the current executable.

        The default implementation checkpoint the given submission and requeues it.
        @plugin-dev: Should be adapted to the signals used in this cluster.
        """
        handler = SignalHandler(self, paths, submission)
        signal.signal(self._usr_sig(), handler.checkpoint_and_try_requeue)
        # A priori we don't need other signals anymore,
        # but still log them to make it easier to debug.
        signal.signal(signal.SIGTERM, handler.bypass)
        signal.signal(signal.SIGCONT, handler.bypass)

    # pylint: disable=unused-argument
    def _requeue(self, countdown: int) -> None:
        """Requeue the current job.

        @plugin-dev:Must be overridden by JobEnvironment implementations.
            Use self.job_id to find what need to be requeued.
        """


class SignalHandler:
    def __init__(self, env: JobEnvironment, job_paths: JobPaths, delayed: DelayedSubmission) -> None:
        self.env = env
        self._job_paths = job_paths
        self._delayed = delayed
        self._logger = logger.get_logger()
        self._start_time = time.time()

    def has_timed_out(self) -> bool:
        # SignalHandler is created by submitit as soon as the process start,
        # so _start_time is an accurate measure of the global runtime of the job.
        walltime = time.time() - self._start_time
        max_walltime = self._delayed._timeout_min * 60
        guaranteed_walltime = min(max_walltime * 0.8, max_walltime - 10 * 60)

        timed_out = walltime >= guaranteed_walltime
        if timed_out:
            self._logger.info(
                f"Job has timed out. Ran {walltime / 60:.0f} minutes out of requested {max_walltime / 60:.0f} minutes."
            )
        else:
            self._logger.info(
                f"Job has not timed out. Ran {walltime / 60:.0f} minutes out of requested {max_walltime / 60:.0f} minutes."
            )
        return timed_out

    def bypass(self, signum: int, frame: types.FrameType = None) -> None:  # pylint:disable=unused-argument
        self._logger.warning(f"Bypassing signal {signal.Signals(signum).name}")

    def checkpoint_and_try_requeue(
        self, signum: int, frame: types.FrameType = None  # pylint:disable=unused-argument
    ) -> None:
        timed_out = self.has_timed_out()
        case = "timed-out" if timed_out else "preempted"
        self._logger.warning(
            f"Caught signal {signal.Signals(signum).name} on {socket.gethostname()}: this job is {case}."
        )

        procid = self.env.global_rank
        if procid != 0:
            self._logger.info(f"Not checkpointing nor requeuing since I am a slave (procid={procid}).")
            # do not sys.exit, because it might kill the master task
            return

        delayed = self._delayed
        countdown = delayed._timeout_countdown - timed_out
        no_requeue_reason = ""
        if hasattr(delayed.function, "checkpoint"):
            no_requeue_reason = _checkpoint(delayed, self._job_paths.submitted_pickle, countdown)
        elif timed_out:
            no_requeue_reason = "timed-out and not checkpointable"
        if countdown < 0:  # this is the end
            no_requeue_reason = "timed-out too many times"
        if no_requeue_reason:
            # raise an error so as to create "result_pickle" file which notifies the job is over
            # this is caught by the try/except in "process_job"
            message = f"Job not requeued because: {no_requeue_reason}."
            self._logger.info(message)
            raise utils.UncompletedJobError(message)
        # if everything went well, requeue!
        self.env._requeue(countdown)
        self._exit()

    def checkpoint_and_exit(
        self, signum: int, frame: types.FrameType = None  # pylint:disable=unused-argument
    ) -> None:
        # Note: no signal is actually bound to `checkpoint_and_exit` but this is used by plugins.
        self._logger.info(f"Caught signal {signal.Signals(signum).name} on {socket.gethostname()}")

        procid = self.env.global_rank
        if procid:
            self._logger.info(f"Not checkpointing since I am a slave (procid={procid}).")
            # do not sys.exit, because it might kill the master task
            return

        delayed = self._delayed
        if hasattr(delayed.function, "checkpoint"):
            _checkpoint(self._delayed, self._job_paths.submitted_pickle, self._delayed._timeout_countdown)
        self._exit()

    def _exit(self) -> None:
        # extracted for mocking
        self._logger.info("Exiting gracefully after preemption/timeout.")
        sys.exit(-1)


def _checkpoint(delayed: DelayedSubmission, filepath: Path, countdown: int) -> str:
    """Call the checkpoint method and dump the updated delayed.

    Returns:
    --------
        no_requeue_reason: str
            a string explaining while there was no requeuing, else empty string if requeuing works
    """
    logger.get_logger().info("Calling checkpoint method.")
    ckpt_delayed = delayed._checkpoint_function()
    if ckpt_delayed is None:
        return "checkpoint function returned None"
    ckpt_delayed.set_timeout(delayed._timeout_min, countdown)
    with utils.temporary_save_path(filepath) as tmp:
        ckpt_delayed.dump(tmp)
    return ""  # requeues
