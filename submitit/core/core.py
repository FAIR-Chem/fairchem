# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import abc
import asyncio
import contextlib
import subprocess
import time as _time
import typing as tp
import uuid
import warnings
from pathlib import Path

from typing_extensions import TypedDict

from . import logger, utils

# R as in "Result", so yes it's covariant.
# pylint: disable=typevar-name-incorrect-variance
R = tp.TypeVar("R", covariant=True)


class InfoWatcher:
    """An instance of this class is shared by all jobs, and is in charge of calling slurm to check status for
    all jobs at once (so as not to overload it). It is also in charge of dealing with errors.
    Cluster is called at 0s, 2s, 4s, 8s etc... in the begginning of jobs, then at least every delay_s (default: 60)

    Parameters
    ----------
    delay_s: int
        Maximum delay before each non-forced call to the cluster.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, delay_s: int = 60) -> None:
        self._delay_s = delay_s
        self._registered: tp.Set[str] = set()
        self._finished: tp.Set[str] = set()
        self._info_dict: tp.Dict[str, tp.Dict[str, str]] = {}
        self._output = b""  # for the record
        self._start_time = 0.0
        self._last_status_check = float("-inf")
        self._num_calls = 0

    def read_info(self, string: tp.Union[bytes, str]) -> tp.Dict[str, tp.Dict[str, str]]:
        raise NotImplementedError

    def _make_command(self) -> tp.Optional[tp.List[str]]:
        raise NotImplementedError

    def get_state(self, job_id: str, mode: str = "standard") -> str:
        raise NotImplementedError

    @property
    def num_calls(self) -> int:
        """Number of calls to sacct"""
        return self._num_calls

    def clear(self) -> None:
        """Clears cache.
        This should hopefully not be used. If you have to use it, please add a github issue.
        """
        self._finished = set()
        self._start_time = _time.time()
        self._last_status_check = float("-inf")
        self._info_dict = {}
        self._output = b""

    def get_info(self, job_id: str, mode: str = "standard") -> tp.Dict[str, str]:
        """Returns a dict containing info about the job.
        State of finished jobs are cached (use watcher.clear() to remove all cache)

        Parameters
        ----------
        job_id: str
            id of the job on the cluster
        mode: str
            one of "force" (forces a call), "standard" (calls regularly) or "cache" (does not call)
        """
        if job_id is None:
            raise RuntimeError("Cannot call sacct without a slurm id")
        if job_id not in self._registered:
            self.register_job(job_id)
        # check with a call to sacct/cinfo
        self.update_if_long_enough(mode)
        return self._info_dict.get(job_id, {})

    def is_done(self, job_id: str, mode: str = "standard") -> bool:
        """Returns whether the job is finished.

        Parameters
        ----------
        job_id: str
            id of the job on the cluster
        mode: str
            one of "force" (forces a call), "standard" (calls regularly) or "cache" (does not call)
        """
        state = self.get_state(job_id, mode=mode)
        return state.upper() not in ["READY", "PENDING", "RUNNING", "UNKNOWN", "REQUEUED", "COMPLETING"]

    def update_if_long_enough(self, mode: str) -> None:
        """Updates if forced to, or if the delay is reached
        (Force-updates with less than 1ms delay are ignored)
        Also checks for finished jobs
        """
        assert mode in ["standard", "force", "cache"]
        if mode == "cache":
            return
        last_check_delta = _time.time() - self._last_status_check
        last_job_delta = _time.time() - self._start_time
        refresh_delay = min(self._delay_s, max(2, last_job_delta / 2))
        if mode == "force":
            refresh_delay = 0.001

        # the following will call update at time 0s, 2s, 4, 8, 16, 32, 64, 124 (delta 60), 184 (delta 60) etc... of last added job
        # (for delay_s = 60)
        if last_check_delta > refresh_delay:
            self.update()

    def update(self) -> None:
        """Updates the info of all registered jobs with a call to sacct"""
        command = self._make_command()
        if command is None:
            return
        self._num_calls += 1
        try:
            logger.get_logger().debug(f"Call #{self.num_calls} - Command {' '.join(command)}")
            self._output = subprocess.check_output(command, shell=False)
        except Exception as e:
            logger.get_logger().warning(
                f"Call #{self.num_calls} - Bypassing sacct error {e}, status may be inaccurate."
            )
        else:
            self._info_dict.update(self.read_info(self._output))
        self._last_status_check = _time.time()
        # check for finished jobs
        to_check = self._registered - self._finished
        for job_id in to_check:
            if self.is_done(job_id, mode="cache"):
                self._finished.add(job_id)

    def register_job(self, job_id: str) -> None:
        """Register a job on the instance for shared update"""
        assert isinstance(job_id, str)
        self._registered.add(job_id)
        self._start_time = _time.time()
        self._last_status_check = float("-inf")


# pylint: disable=too-many-public-methods
class Job(tp.Generic[R]):
    """Access to a cluster job information and result.

    Parameters
    ----------
    folder: Path/str
        A path to the submitted job file
    job_id: str
        the id of the cluster job
    tasks: List[int]
        The ids of the tasks associated to this job.
        If None, the job has only one task (with id = 0)
    """

    _cancel_command = "dummy"
    _results_timeout_s = 15
    watcher = InfoWatcher()

    def __init__(self, folder: tp.Union[Path, str], job_id: str, tasks: tp.Sequence[int] = (0,)) -> None:
        self._job_id = job_id
        self._tasks = tuple(tasks)
        self._sub_jobs: tp.Sequence["Job[R]"] = []
        self._cancel_at_deletion = False
        if len(tasks) > 1:
            # This is a meta-Job
            self._sub_jobs = [self.__class__(folder=folder, job_id=job_id, tasks=(k,)) for k in self._tasks]
        self._paths = utils.JobPaths(folder, job_id=job_id, task_id=self.task_id)
        self._start_time = _time.time()
        self._last_status_check = self._start_time  # for the "done()" method
        # register for state updates with watcher
        self._register_in_watcher()

    def _register_in_watcher(self) -> None:
        if not self._tasks[0]:  # only register for task=0
            self.watcher.register_job(self.job_id)

    @property
    def job_id(self) -> str:
        return self._job_id

    @property
    def paths(self) -> utils.JobPaths:
        return self._paths

    @property
    def num_tasks(self) -> int:
        """Returns the number of tasks in the Job"""
        if not self._sub_jobs:
            return 1
        return len(self._sub_jobs)

    def submission(self) -> utils.DelayedSubmission:
        """Returns the submitted object, with attributes `function`, `args` and `kwargs`"""
        assert (
            self.paths.submitted_pickle.exists()
        ), f"Cannot find job submission pickle: {self.paths.submitted_pickle}"
        return utils.DelayedSubmission.load(self.paths.submitted_pickle)

    def cancel_at_deletion(self, value: bool = True) -> "Job[R]":
        """Sets whether the job deletion in the python environment triggers
        cancellation of the corresponding job in the cluster
        By default, jobs are not cancelled unless this method is called to turn the
        option on.

        Parameters
        ----------
        value: bool
            if True, the cluster job will be cancelled at the instance deletion, if False, it
            will not.

        Returns
        -------
        Job
            the current job (for chaining at submission for instance: "job = executor.submit(...).cancel_at_deletion()")
        """
        self._cancel_at_deletion = value
        return self

    def task(self, task_id: int) -> "Job[R]":
        """Returns a given sub-Job (task).

        Parameters
        ----------
        task_id
            The id of the task. Must be between 0 and self.num_tasks
        Returns
        -------
        job
            The sub_job. You can call all Job methods on it (done, stdout, ...)
            If the job doesn't have sub jobs, return the job itself.
        """
        if not 0 <= task_id < self.num_tasks:
            raise ValueError(f"task_id {task_id} must be between 0 and {self.num_tasks - 1}")

        if not self._sub_jobs:
            return self
        return self._sub_jobs[task_id]

    def cancel(self, check: bool = True) -> None:
        """Cancels the job

        Parameters
        ----------
        check: bool
            whether to wait for completion and check that the command worked
        """
        (subprocess.check_call if check else subprocess.call)(
            [self._cancel_command, f"{self.job_id}"], shell=False
        )

    def result(self) -> R:
        r = self.results()
        assert not self._sub_jobs, "You should use `results()` if your job has subtasks."
        return r[0]

    def results(self) -> tp.List[R]:
        """Waits for and outputs the result of the submitted function

        Returns
        -------
        output
            the output of the submitted function.
            If the job has several tasks, it will return the output of every tasks in a List

        Raises
        ------
        Exception
            Any exception raised by the job
        """
        self.wait()

        if self._sub_jobs:
            return [tp.cast(R, sub_job.result()) for sub_job in self._sub_jobs]

        outcome, result = self._get_outcome_and_result()
        if outcome == "error":
            job_exception = self.exception()
            if job_exception is None:
                raise RuntimeError("Unknown job exception")
            raise job_exception  # pylint: disable=raising-bad-type
        return [result]

    def exception(self) -> tp.Optional[tp.Union[utils.UncompletedJobError, utils.FailedJobError]]:
        """Waits for completion and returns (not raise) the
        exception containing the error log of the job

        Returns
        -------
        Exception/None
            the exception if any was raised during the job.
            If the job has several tasks, it returns the exception of the task with
            smallest id that failed.

        Raises
        ------
        UncompletedJobError
            In case the job never completed
        """
        self.wait()

        if self._sub_jobs:
            all_exceptions = [sub_job.exception() for sub_job in self._sub_jobs]
            # unexpected pylint issue on correct code:
            exceptions = [
                e for e in all_exceptions if e is not None  # pylint: disable=used-before-assignment
            ]
            if not exceptions:
                return None
            return exceptions[0]

        try:
            outcome, trace = self._get_outcome_and_result()
        except utils.UncompletedJobError as e:
            return e
        if outcome == "error":
            return utils.FailedJobError(
                f"Job (task={self.task_id}) failed during processing with trace:\n"
                f"----------------------\n{trace}\n"
                "----------------------\n"
                f"You can check full logs with 'job.stderr({self.task_id})' and 'job.stdout({self.task_id})'"
                f"or at paths:\n  - {self.paths.stderr}\n  - {self.paths.stdout}"
            )
        return None

    def _get_outcome_and_result(self) -> tp.Tuple[str, tp.Any]:
        """Getter for the output of the submitted function.

        Returns
        -------
        outcome
            the outcome of the job: either "error" or "success"
        result
            the output of the submitted function

        Raises
        ------
        UncompletedJobError
            if the job is not finished or failed outside of the job (from slurm)
        """
        assert not self._sub_jobs, "This should not be called for a meta-job"

        p = self.paths.folder
        timeout = self._results_timeout_s
        try:
            # trigger cache update: https://stackoverflow.com/questions/3112546/os-path-exists-lies/3112717
            p.chmod(p.stat().st_mode)
        except PermissionError:
            # chmod requires file ownership and might fail.
            # Increase the timeout since we can't force cache refresh.
            timeout *= 2
        # if filesystem is slow, we need to wait a bit for result_pickle.
        start_wait = _time.time()
        while not self.paths.result_pickle.exists() and _time.time() - start_wait < timeout:
            _time.sleep(1)
        if not self.paths.result_pickle.exists():
            message = [
                f"Job {self.job_id} (task: {self.task_id}) with path {self.paths.result_pickle}",
                f"has not produced any output (state: {self.state})",
            ]
            log = self.stderr()
            if log:
                message.extend(["Error stream produced:", "-" * 40, log])
            elif self.paths.stdout.exists():
                log = subprocess.check_output(["tail", "-40", str(self.paths.stdout)], encoding="utf-8")
                message.extend(
                    [f"No error stream produced. Look at stdout: {self.paths.stdout}", "-" * 40, log]
                )
            else:
                message.append(f"No output/error stream produced ! Check: {self.paths.stdout}")
            raise utils.UncompletedJobError("\n".join(message))
        try:
            output: tp.Tuple[str, tp.Any] = utils.pickle_load(self.paths.result_pickle)
        except EOFError:
            warnings.warn(f"EOFError on file {self.paths.result_pickle}, trying again in 2s")  # will it work?
            _time.sleep(2)
            output = utils.pickle_load(self.paths.result_pickle)
        return output

    def wait(self) -> None:
        """Wait while no result find is found and the state is
        either PENDING or RUNNING.
        The state is checked from slurm at least every min and the result path
        every second.
        """
        while not self.done():
            _time.sleep(1)

    def done(self, force_check: bool = False) -> bool:
        """Checks whether the job is finished.
        This is done by checking if the result file is present,
        or checking the job state regularly (at least every minute)
        If the job has several tasks, the job is done once all tasks are done.

        Parameters
        ----------
        force_check: bool
            Forces the slurm state update

        Returns
        -------
        bool
            whether the job is finished or not

        Note
        ----
        This function is not foolproof, and may say that the job is not terminated even
        if it is when the job failed (no result file, but job not running) because
        we avoid calling sacct/cinfo everytime done is called
        """
        # TODO: keep state info once job is finished?
        if self._sub_jobs:
            return all(sub_job.done() for sub_job in self._sub_jobs)
        p = self.paths.folder
        try:
            # trigger cache update: https://stackoverflow.com/questions/3112546/os-path-exists-lies/3112717
            p.chmod(p.stat().st_mode)
        except OSError:
            pass
        if self.paths.result_pickle.exists():
            return True
        # check with a call to sacct/cinfo
        if self.watcher.is_done(self.job_id, mode="force" if force_check else "standard"):
            return True
        return False

    @property
    def task_id(self) -> tp.Optional[int]:
        return None if len(self._tasks) > 1 else self._tasks[0]

    @property
    def state(self) -> str:
        """State of the job (does not force an update)"""
        return self.watcher.get_state(self.job_id, mode="standard")

    def get_info(self, mode: str = "force") -> tp.Dict[str, str]:
        """Returns informations about the job as a dict (sacct call)"""
        return self.watcher.get_info(self.job_id, mode=mode)

    def _get_logs_string(self, name: str) -> tp.Optional[str]:
        """Returns a string with the content of the log file
        or None if the file does not exist yet

        Parameter
        ---------
        name: str
            either "stdout" or "stderr"
        """
        paths = {"stdout": self.paths.stdout, "stderr": self.paths.stderr}
        if name not in paths:
            raise ValueError(f'Unknown "{name}", available are {list(paths.keys())}')
        if not paths[name].exists():
            return None
        with paths[name].open("r") as f:
            string: str = f.read()
        return string

    def stdout(self) -> tp.Optional[str]:
        """Returns a string with the content of the print log file
        or None if the file does not exist yet
        """
        if self._sub_jobs:
            stdout_ = [sub_job.stdout() for sub_job in self._sub_jobs]
            stdout_not_none = [s for s in stdout_ if s is not None]
            if not stdout_not_none:
                return None
            return "\n".join(stdout_not_none)

        return self._get_logs_string("stdout")

    def stderr(self) -> tp.Optional[str]:
        """Returns a string with the content of the error log file
        or None if the file does not exist yet
        """
        if self._sub_jobs:
            stderr_ = [sub_job.stderr() for sub_job in self._sub_jobs]
            stderr_not_none: tp.List[str] = [s for s in stderr_ if s is not None]
            if not stderr_not_none:
                return None
            return "\n".join(stderr_not_none)
        return self._get_logs_string("stderr")

    def awaitable(self) -> "AsyncJobProxy[R]":
        """Returns a proxy object that provides asyncio methods
        for this Job.
        """
        return AsyncJobProxy(self)

    def __repr__(self) -> str:
        state = "UNKNOWN"
        try:
            state = self.state
        except Exception as e:
            logger.get_logger().warning(f"Bypassing state error:\n{e}")
        return f'{self.__class__.__name__}<job_id={self.job_id}, task_id={self.task_id}, state="{state}">'

    def __del__(self) -> None:
        if self._cancel_at_deletion:
            if not self.watcher.is_done(self.job_id, mode="cache"):
                self.cancel(check=False)

    def __getstate__(self) -> tp.Dict[str, tp.Any]:
        return self.__dict__  # for pickling (see __setstate__)

    def __setstate__(self, state: tp.Dict[str, tp.Any]) -> None:
        """Make sure jobs are registered when loaded from a pickle"""
        self.__dict__.update(state)
        self._register_in_watcher()


class DelayedJob(Job[R]):
    """
    Represents a Job that have been queue for submission by an executor,
    but hasn't yet been scheduled.
    Typically obtained by calling `ex.submit` within a `ex.batch()` context

    Trying to read the attributes of the job will, by default, fail.
    But if you passed `ex.batch(allow_implicit_submission=True)` then
    the attribute read will in fact force the job submission,
    and you'll obtain a real job instead.
    """

    def __init__(self, ex: "Executor"):
        # pylint: disable = super-init-not-called
        self._submitit_executor = ex

    def __getattr__(self, name: str) -> tp.Any:
        # _cancel_at_deletion is used in __del__, we don't want it to trigger submission
        if name == "_cancel_at_deletion":
            return False

        ex = self.__dict__["_submitit_executor"]
        # this submits the batch so as to fill the instance attributes
        # this may return false if we try to submit within executor.batch()
        # without passing `executor.batch(allow_implicit_submission=True)`
        if not ex._allow_implicit_submissions:
            raise AttributeError(
                "Accesssing job attributes is forbidden within 'with executor.batch()' context"
            )
        ex._submit_delayed_batch()
        # Ensure that _promote did get called, otherwise getattr will trigger a stack overflow
        assert self.__class__ != DelayedJob, f"Executor {ex} didn't properly submitted {self} !"
        return getattr(self, name)

    def _promote(self, new_job: Job[tp.Any]) -> None:
        # fill in the empty shell, the pickle way
        self.__dict__.pop("_submitit_executor", None)
        self.__dict__.update(new_job.__dict__)
        # pylint: disable=attribute-defined-outside-init
        self.__class__ = new_job.__class__  # type: ignore

    def __repr__(self) -> str:
        return object.__repr__(self)


class AsyncJobProxy(tp.Generic[R]):
    def __init__(self, job: Job[R]):
        self.job = job

    async def wait(self, poll_interval: tp.Union[int, float] = 1) -> None:
        """same as wait() but with asyncio sleep."""
        while not self.job.done():
            await asyncio.sleep(poll_interval)

    async def result(self, poll_interval: tp.Union[int, float] = 1) -> R:
        """asyncio version of the result() method.
        Wait asynchornously for the result to be available by polling the self.done() method.
        Parameters
        ----------
        poll_interval: int or float
            how often to check if the result is available, in seconds
        """
        await self.wait(poll_interval)
        return self.job.result()

    async def results(self, poll_interval: tp.Union[int, float] = 1) -> tp.List[R]:
        """asyncio version of the results() method.

        Waits asynchornously for ALL the results to be available by polling the self.done() method.

        Parameters
        ----------
        poll_interval: int or float
            how often to check if the result is available, in seconds
        """
        await self.wait(poll_interval)
        # results are ready now
        return self.job.results()

    def results_as_completed(self, poll_interval: tp.Union[int, float] = 1) -> tp.Iterator[asyncio.Future]:
        """awaits for all tasks results concurrently. Note that the order of results is not guaranteed to match the order
        of the tasks anymore as the earliest task coming back might not be the first one you sent.

        Returns
        -------
        an iterable of Awaitables that can be awaited on to get the earliest result available of the remaining tasks.

        Parameters
        ----------
        poll_interval: int or float
            how often to check if the result is available, in seconds

        (see https://docs.python.org/3/library/asyncio-task.html#asyncio.as_completed)
        """
        if self.job.num_tasks > 1:
            yield from asyncio.as_completed(
                [self.job.task(i).awaitable().result(poll_interval) for i in range(self.job.num_tasks)]
            )

        # there is only one result anyway, let's just use async result
        yield asyncio.ensure_future(self.result())


_MSG = (
    "Interactions with jobs are not allowed within "
    '"with executor.batch()" context (submissions/creations only happens at exit time).'
)


class EquivalenceDict(TypedDict):
    """Gives the specific name of the params shared across all plugins."""

    # Note that all values are typed as string, even though they correspond to integer.
    # This allow to have a static typing on the "_equivalence_dict" method implemented
    # by plugins.
    # We could chose to put the proper types, but that wouldn't be enough to typecheck
    # the calls to `update_parameters` which uses kwargs.
    name: str
    timeout_min: str
    mem_gb: str
    nodes: str
    cpus_per_task: str
    gpus_per_node: str
    tasks_per_node: str


class Executor(abc.ABC):
    """Base job executor.

    Parameters
    ----------
    folder: Path/str
        folder for storing job submission/output and logs.
    """

    job_class: tp.Type[Job[tp.Any]] = Job

    def __init__(self, folder: tp.Union[str, Path], parameters: tp.Optional[tp.Dict[str, tp.Any]] = None):
        self.folder = Path(folder).expanduser().absolute()
        self.parameters = {} if parameters is None else parameters
        # storage for the batch context manager, for batch submissions:
        self._delayed_batch: tp.Optional[tp.List[tp.Tuple[Job[tp.Any], utils.DelayedSubmission]]] = None
        self._allow_implicit_submissions = False

    @classmethod
    def name(cls) -> str:
        n = cls.__name__
        if n.endswith("Executor"):
            n = n[: -len("Executor")]
        return n.lower()

    @contextlib.contextmanager
    def batch(self, allow_implicit_submissions: bool = False) -> tp.Iterator[None]:
        """Creates a context within which all submissions are packed into a job array.
        By default the array submissions happens when leaving the context

        Parameter
        ---------
        allow_implicit_submissions: bool
            submits the current batch whenever a job attribute is accessed instead of raising an exception

        Example
        -------
        jobs = []
        with executor.batch():
            for k in range(12):
                jobs.append(executor.submit(add, k, 1))

        Raises
        ------
        AttributeError
            if trying to access a job instance attribute while the batch is not exited, and
            intermediate submissions are not allowed.
        """
        self._allow_implicit_submissions = allow_implicit_submissions
        if self._delayed_batch is not None:
            raise RuntimeError('Nesting "with executor.batch()" contexts is not allowed.')
        self._delayed_batch = []
        try:
            yield None
        except Exception as e:
            logger.get_logger().error(
                'Caught error within "with executor.batch()" context, submissions are dropped.\n '
            )
            raise e
        else:
            self._submit_delayed_batch()
        finally:
            self._delayed_batch = None

    def _submit_delayed_batch(self) -> None:
        assert self._delayed_batch is not None
        if not self._delayed_batch:
            if not self._allow_implicit_submissions:
                warnings.warn(
                    'No submission happened during "with executor.batch()" context.', category=RuntimeWarning
                )
            return
        jobs, submissions = zip(*self._delayed_batch)
        new_jobs = self._internal_process_submissions(submissions)
        for j, new_j in zip(jobs, new_jobs):
            j._promote(new_j)
        self._delayed_batch = []

    def submit(self, fn: tp.Callable[..., R], *args: tp.Any, **kwargs: tp.Any) -> Job[R]:
        ds = utils.DelayedSubmission(fn, *args, **kwargs)
        if self._delayed_batch is not None:
            job: Job[R] = DelayedJob(self)
            self._delayed_batch.append((job, ds))
        else:
            job = self._internal_process_submissions([ds])[0]
        if type(job) is Job:  # pylint: disable=unidiomatic-typecheck
            raise RuntimeError("Executors should never return a base Job class (implementation issue)")
        return job

    @abc.abstractmethod
    def _internal_process_submissions(
        self, delayed_submissions: tp.List[utils.DelayedSubmission]
    ) -> tp.List[Job[tp.Any]]:
        ...

    def map_array(self, fn: tp.Callable[..., R], *iterable: tp.Iterable[tp.Any]) -> tp.List[Job[R]]:
        """A distributed equivalent of the map() built-in function

        Parameters
        ----------
        fn: callable
            function to compute
        *iterable: Iterable
            lists of arguments that are passed as arguments to fn.

        Returns
        -------
        List[Job]
            A list of Job instances.

        Example
        -------
        a = [1, 2, 3]
        b = [10, 20, 30]
        executor.map_array(add, a, b)
        # jobs will compute 1 + 10, 2 + 20, 3 + 30
        """
        submissions = [utils.DelayedSubmission(fn, *args) for args in zip(*iterable)]
        if len(submissions) == 0:
            warnings.warn("Received an empty job array")
            return []
        return self._internal_process_submissions(submissions)

    def submit_array(self, fns: tp.Sequence[tp.Callable[[], R]]) -> tp.List[Job[R]]:
        """Submit a list of job. This is useful when submiting different Checkpointable functions.
        Be mindful that all those functions will be run with the same requirements
        (cpus, gpus, timeout, ...). So try to make group of similar function calls.

        Parameters
        ----------
        fns: list of callable
            functions to compute. Those functions must not need any argument.
            Tyically those are "Checkpointable" instance whose arguments
            have been specified in the constructor, or partial functions.

        Returns
        -------
        List[Job]
            A list of Job instances.

        Example
        -------
        a_vals = [1, 2, 3]
        b_vals = [10, 20, 30]
        fns = [functools.partial(int.__add__, a, b) for (a, b) in zip (a_vals, b_vals)]
        executor.submit_array(fns)
        # jobs will compute 1 + 10, 2 + 20, 3 + 30
        """
        submissions = [utils.DelayedSubmission(fn) for fn in fns]
        if len(submissions) == 0:
            warnings.warn("Received an empty job array")
            return []
        return self._internal_process_submissions(submissions)

    def update_parameters(self, **kwargs: tp.Any) -> None:
        """Update submision parameters."""
        if self._delayed_batch is not None:
            raise RuntimeError(
                'Changing parameters within batch context "with executor.batch():" is not allowed'
            )
        self._internal_update_parameters(**kwargs)

    @classmethod
    def _equivalence_dict(cls) -> tp.Optional[EquivalenceDict]:
        return None

    @classmethod
    def _valid_parameters(cls) -> tp.Set[str]:
        """Parameters that can be set through update_parameters"""
        return set()

    def _convert_parameters(self, params: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Any]:
        """Convert generic parameters to their specific equivalent.
        This has to be called **before** calling `update_parameters`.

        The default implementation only renames the key using `_equivalence_dict`.
        """
        eq_dict = tp.cast(tp.Optional[tp.Dict[str, str]], self._equivalence_dict())
        if eq_dict is None:
            return params
        return {eq_dict.get(k, k): v for k, v in params.items()}

    def _internal_update_parameters(self, **kwargs: tp.Any) -> None:
        """Update submission parameters."""
        self.parameters.update(kwargs)

    @classmethod
    def affinity(cls) -> int:
        """The 'score' of this executor on the current environment.

        -> -1 means unavailable
        ->  0 means available but won't be started unless asked (eg debug executor)
        ->  1 means available
        ->  2 means available and is a highly scalable executor (cluster)
        """
        return 1


class PicklingExecutor(Executor):
    """Base job executor.

    Parameters
    ----------
    folder: Path/str
        folder for storing job submission/output and logs.
    """

    def __init__(self, folder: tp.Union[Path, str], max_num_timeout: int = 3) -> None:
        super().__init__(folder)
        self.max_num_timeout = max_num_timeout
        self._throttling = 0.2
        self._last_job_submitted = 0.0

    def _internal_process_submissions(
        self, delayed_submissions: tp.List[utils.DelayedSubmission]
    ) -> tp.List[Job[tp.Any]]:
        """Submits a task to the cluster.

        Parameters
        ----------
        fn: callable
            The function to compute
        *args: any positional argument for the function
        **kwargs: any named argument for the function

        Returns
        -------
        Job
            A Job instance, providing access to the job information,
            including the output of the function once it is computed.
        """
        eq_dict = self._equivalence_dict()
        timeout_min = self.parameters.get(eq_dict["timeout_min"] if eq_dict else "timeout_min", 5)
        jobs = []
        for delayed in delayed_submissions:
            tmp_uuid = uuid.uuid4().hex
            pickle_path = utils.JobPaths.get_first_id_independent_folder(self.folder) / f"{tmp_uuid}.pkl"
            pickle_path.parent.mkdir(parents=True, exist_ok=True)
            delayed.set_timeout(timeout_min, self.max_num_timeout)
            delayed.dump(pickle_path)

            self._throttle()
            self._last_job_submitted = _time.time()
            job = self._submit_command(self._submitit_command_str)
            job.paths.move_temporary_file(pickle_path, "submitted_pickle")
            jobs.append(job)
        return jobs

    def _throttle(self) -> None:
        while _time.time() - self._last_job_submitted < self._throttling:
            _time.sleep(self._throttling)

    @property
    def _submitit_command_str(self) -> str:
        # this is the command submitted from "submit" to "_submit_command"
        return "dummy"

    def _submit_command(self, command: str) -> Job[tp.Any]:
        """Submits a command to the cluster
        It is recommended not to use this function since the Job instance assumes pickle
        files will be created at the end of the job, and hence it will not work correctly.
        You may use a CommandFunction as argument to the submit function instead. The only
        problem with this latter solution is that stdout is buffered, and you will therefore
        not be able to monitor the logs in real time.

        Parameters
        ----------
        command: str
            a command string

        Returns
        -------
        Job
            A Job instance, providing access to the crun job information.
            Since it has no output, some methods will not be efficient
        """
        tmp_uuid = uuid.uuid4().hex
        submission_file_path = (
            utils.JobPaths.get_first_id_independent_folder(self.folder) / f"submission_file_{tmp_uuid}.sh"
        )
        with submission_file_path.open("w") as f:
            f.write(self._make_submission_file_text(command, tmp_uuid))
        command_list = self._make_submission_command(submission_file_path)
        # run
        output = utils.CommandFunction(command_list, verbose=False)()  # explicit errors
        job_id = self._get_job_id_from_submission_command(output)
        tasks_ids = list(range(self._num_tasks()))
        job: Job[tp.Any] = self.job_class(folder=self.folder, job_id=job_id, tasks=tasks_ids)
        job.paths.move_temporary_file(submission_file_path, "submission_file")
        self._write_job_id(job.job_id, tmp_uuid)
        self._set_job_permissions(job.paths.folder)
        return job

    def _write_job_id(self, job_id: str, uid: str) -> None:
        """Write the job id in a file named {job-independent folder}/parent_job_id_{uid}.
        This can create files read by plugins to get the job_id of the parent job
        """

    @abc.abstractmethod
    def _num_tasks(self) -> int:
        """Returns the number of tasks associated to the job"""
        raise NotImplementedError

    @abc.abstractmethod
    def _make_submission_file_text(self, command: str, uid: str) -> str:
        """Creates the text of a file which will be created and run
        for the submission (for slurm, this is sbatch file).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _make_submission_command(self, submission_file_path: Path) -> tp.List[str]:
        """Create the submission command."""
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def _get_job_id_from_submission_command(string: tp.Union[bytes, str]) -> str:
        """Recover the job id from the output of the submission command."""
        raise NotImplementedError

    @staticmethod
    def _set_job_permissions(folder: Path) -> None:
        pass
