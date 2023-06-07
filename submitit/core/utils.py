# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import contextlib
import io
import itertools
import os
import pickle
import select
import shutil
import subprocess
import sys
import tarfile
import typing as tp
from pathlib import Path

import cloudpickle


@contextlib.contextmanager
def environment_variables(**kwargs: tp.Any) -> tp.Iterator[None]:
    backup = {x: os.environ[x] for x in kwargs if x in os.environ}
    os.environ.update({x: str(y) for x, y in kwargs.items()})
    try:
        yield
    finally:
        for x in kwargs:
            del os.environ[x]
        os.environ.update(backup)


class UncompletedJobError(RuntimeError):
    """Job is uncomplete: either unfinished or failed"""


class FailedJobError(UncompletedJobError):
    """Job failed during processing"""


class FailedSubmissionError(RuntimeError):
    """Job Submission failed"""


class JobPaths:
    """Creates paths related to the slurm job and its submission"""

    def __init__(
        self, folder: tp.Union[Path, str], job_id: tp.Optional[str] = None, task_id: tp.Optional[int] = None
    ) -> None:
        self._folder = Path(folder).expanduser().absolute()
        self.job_id = job_id
        self.task_id = task_id or 0

    @property
    def folder(self) -> Path:
        return self._format_id(self._folder)

    @property
    def submission_file(self) -> Path:
        if self.job_id and "_" in self.job_id:
            # We only have one submission file per job array
            return self._format_id(self.folder / "%A_submission.sh")
        return self._format_id(self.folder / "%j_submission.sh")

    @property
    def submitted_pickle(self) -> Path:
        return self._format_id(self.folder / "%j_submitted.pkl")

    @property
    def result_pickle(self) -> Path:
        return self._format_id(self.folder / "%j_%t_result.pkl")

    @property
    def stderr(self) -> Path:
        return self._format_id(self.folder / "%j_%t_log.err")

    @property
    def stdout(self) -> Path:
        return self._format_id(self.folder / "%j_%t_log.out")

    def _format_id(self, path: tp.Union[Path, str]) -> Path:
        """Replace id tag by actual id if available"""
        if self.job_id is None:
            return Path(path)
        replaced_path = str(path).replace("%j", str(self.job_id)).replace("%t", str(self.task_id))
        array_id, *array_index = str(self.job_id).split("_", 1)
        if "%a" in replaced_path:
            if len(array_index) != 1:
                raise ValueError("%a is in the folder path but this is not a job array")
            replaced_path = replaced_path.replace("%a", array_index[0])
        return Path(replaced_path.replace("%A", array_id))

    def move_temporary_file(self, tmp_path: tp.Union[Path, str], name: str) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        Path(tmp_path).rename(getattr(self, name))

    @staticmethod
    def get_first_id_independent_folder(folder: tp.Union[Path, str]) -> Path:
        """Returns the closest folder which is id independent"""
        parts = Path(folder).expanduser().absolute().parts
        tags = ["%j", "%t", "%A", "%a"]
        indep_parts = itertools.takewhile(lambda x: not any(tag in x for tag in tags), parts)
        return Path(*indep_parts)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.folder})"


class DelayedSubmission:
    """Object for specifying the function/callable call to submit and process later.
    This is only syntactic sugar to make sure everything is well formatted:
    If what you want to compute later is func(*args, **kwargs), just instanciate:
    DelayedSubmission(func, *args, **kwargs).
    It also provides convenient tools for dumping and loading.
    """

    def __init__(self, function: tp.Callable[..., tp.Any], *args: tp.Any, **kwargs: tp.Any) -> None:
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self._result: tp.Any = None
        self._done = False
        self._timeout_min: int = 0
        self._timeout_countdown: int = 0  # controlled in submission and execution

    def result(self) -> tp.Any:
        if self._done:
            return self._result

        self._result = self.function(*self.args, **self.kwargs)
        self._done = True
        return self._result

    def done(self) -> bool:
        return self._done

    def dump(self, filepath: tp.Union[str, Path]) -> None:
        cloudpickle_dump(self, filepath)

    def set_timeout(self, timeout_min: int, max_num_timeout: int) -> None:
        self._timeout_min = timeout_min
        self._timeout_countdown = max_num_timeout

    @classmethod
    def load(cls: tp.Type["DelayedSubmission"], filepath: tp.Union[str, Path]) -> "DelayedSubmission":
        obj = pickle_load(filepath)
        # following assertion is relaxed compared to isinstance, to allow flexibility
        # (Eg: copying this class in a project to be able to have checkpointable jobs without adding submitit as dependency)
        assert obj.__class__.__name__ == cls.__name__, f"Loaded object is {type(obj)} but should be {cls}."
        return obj  # type: ignore

    def _checkpoint_function(self) -> tp.Optional["DelayedSubmission"]:
        checkpoint = getattr(self.function, "__submitit_checkpoint__", None)
        if checkpoint is None:
            checkpoint = getattr(self.function, "checkpoint", None)
        if checkpoint is None:
            return None
        return checkpoint(*self.args, **self.kwargs)  # type: ignore


@contextlib.contextmanager
def temporary_save_path(filepath: tp.Union[Path, str]) -> tp.Iterator[Path]:
    """Yields a path where to save a file and moves it
    afterward to the provided location (and replaces any
    existing file)
    This is useful to avoid processes monitoring the filepath
    to break if trying to read when the file is being written.

    Note
    ----
    The temporary path is the provided path appended with .save_tmp
    """
    filepath = Path(filepath)
    tmppath = filepath.with_suffix(filepath.suffix + ".save_tmp")
    assert not tmppath.exists(), "A temporary saved file already exists."
    yield tmppath
    if not tmppath.exists():
        raise FileNotFoundError("No file was saved at the temporary path.")
    if filepath.exists():
        os.remove(filepath)
    os.rename(tmppath, filepath)


def archive_dev_folders(
    folders: tp.List[tp.Union[str, Path]], outfile: tp.Optional[tp.Union[str, Path]] = None
) -> Path:
    """Creates a tar.gz file with all provided folders"""
    assert isinstance(folders, (list, tuple)), "Only lists and tuples of folders are allowed"
    if outfile is None:
        outfile = "_dev_folders_.tar.gz"
    outfile = Path(outfile)
    assert str(outfile).endswith(".tar.gz"), "Archive file must have extension .tar.gz"
    with tarfile.TarFile(outfile, mode="w") as tf:
        for folder in folders:
            tf.add(str(folder), arcname=Path(folder).name)
    return outfile


def copy_par_file(par_file: tp.Union[str, Path], folder: tp.Union[str, Path]) -> Path:
    """Copy the par (or xar) file in the folder

    Parameter
    ---------
    par_file: str/Path
        Par file generated by buck
    folder: str/Path
        folder where the par file must be copied

    Returns
    -------
    Path
        Path of the copied .par file
    """
    par_file = Path(par_file).expanduser().absolute()
    folder = Path(folder).expanduser().absolute()
    folder.mkdir(parents=True, exist_ok=True)
    dst_name = folder / par_file.name
    shutil.copy2(par_file, dst_name)
    return dst_name


def pickle_load(filename: tp.Union[str, Path]) -> tp.Any:
    # this is used by cloudpickle as well
    with open(filename, "rb") as ifile:
        return pickle.load(ifile)


def cloudpickle_dump(obj: tp.Any, filename: tp.Union[str, Path]) -> None:
    with open(filename, "wb") as ofile:
        cloudpickle.dump(obj, ofile, pickle.HIGHEST_PROTOCOL)


# pylint: disable=too-many-locals
def copy_process_streams(
    process: subprocess.Popen, stdout: io.StringIO, stderr: io.StringIO, verbose: bool = False
):
    """
    Reads the given process stdout/stderr and write them to StringIO objects.
    Make sure that there is no deadlock because of pipe congestion.
    If `verbose` the process stdout/stderr are also copying to the interpreter stdout/stderr.
    """

    def raw(stream: tp.Optional[tp.IO[bytes]]) -> tp.IO[bytes]:
        assert stream is not None
        if isinstance(stream, io.BufferedIOBase):
            stream = stream.raw
        return stream

    p_stdout, p_stderr = raw(process.stdout), raw(process.stderr)
    stream_by_fd: tp.Dict[int, tp.Tuple[tp.IO[bytes], io.StringIO, tp.IO[str]]] = {
        p_stdout.fileno(): (p_stdout, stdout, sys.stdout),
        p_stderr.fileno(): (p_stderr, stderr, sys.stderr),
    }
    fds = list(stream_by_fd.keys())
    poller = select.poll()
    for fd in stream_by_fd:
        poller.register(fd, select.POLLIN | select.POLLPRI)
    while fds:
        # `poll` syscall will wait until one of the registered file descriptors has content.
        ready = poller.poll()
        for fd, _ in ready:
            p_stream, string, std = stream_by_fd[fd]
            raw_buf = p_stream.read(2**16)
            if not raw_buf:
                fds.remove(fd)
                poller.unregister(fd)
                continue
            buf = raw_buf.decode()
            string.write(buf)
            string.flush()
            if verbose:
                std.write(buf)
                std.flush()


# used in "_core", so cannot be in "helpers"
class CommandFunction:
    """Wraps a command as a function in order to make sure it goes through the
    pipeline and notify when it is finished.
    The output is a string containing everything that has been sent to stdout.
    WARNING: use CommandFunction only if you know the output won't be too big !
    Otherwise use subprocess.run() that also streams the outputto stdout/stderr.

    Parameters
    ----------
    command: list
        command to run, as a list
    verbose: bool
        prints the command and stdout at runtime
    cwd: Path/str
        path to the location where the command must run from

    Returns
    -------
    str
       Everything that has been sent to stdout
    """

    def __init__(
        self,
        command: tp.List[str],
        verbose: bool = True,
        cwd: tp.Optional[tp.Union[str, Path]] = None,
        env: tp.Optional[tp.Dict[str, str]] = None,
    ) -> None:
        if not isinstance(command, list):
            raise TypeError("The command must be provided as a list")
        self.command = command
        self.verbose = verbose
        self.cwd = None if cwd is None else str(cwd)
        self.env = env

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> str:
        """Call the cammand line with addidional arguments
        The keyword arguments will be sent as --{key}={val}
        The logs bufferized. They will be printed if the job fails, or sent as output of the function
        Errors are provided with the internal stderr.
        """
        full_command = (
            self.command + [str(x) for x in args] + [f"--{x}={y}" for x, y in kwargs.items()]
        )  # TODO bad parsing
        if self.verbose:
            print(f"The following command is sent: \"{' '.join(full_command)}\"")
        with subprocess.Popen(
            full_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            cwd=self.cwd,
            env=self.env,
        ) as process:
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()

            try:
                copy_process_streams(process, stdout_buffer, stderr_buffer, self.verbose)
            except Exception as e:
                process.kill()
                process.wait()
                raise FailedJobError("Job got killed for an unknown reason.") from e
            stdout = stdout_buffer.getvalue().strip()
            stderr = stderr_buffer.getvalue().strip()
            retcode = process.wait()
            if stderr and (retcode and not self.verbose):
                # We don't print is self.verbose, as it already happened before.
                print(stderr, file=sys.stderr)
            if retcode:
                subprocess_error = subprocess.CalledProcessError(
                    retcode, process.args, output=stdout, stderr=stderr
                )
                raise FailedJobError(stderr) from subprocess_error
        return stdout
