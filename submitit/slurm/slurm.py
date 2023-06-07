# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import functools
import inspect
import os
import re
import shlex
import shutil
import subprocess
import sys
import typing as tp
import uuid
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..core import core, job_environment, logger, utils


def read_job_id(job_id: str) -> tp.List[Tuple[str, ...]]:
    """Reads formated job id and returns a tuple with format:
    (main_id, [array_index, [final_array_index])
    """
    pattern = r"(?P<main_id>\d+)_\[(?P<arrays>(\d+(-\d+)?(,)?)+)(\%\d+)?\]"
    match = re.search(pattern, job_id)
    if match is not None:
        main = match.group("main_id")
        array_ranges = match.group("arrays").split(",")
        return [tuple([main] + array_range.split("-")) for array_range in array_ranges]
    else:
        main_id, *array_id = job_id.split("_", 1)
        if not array_id:
            return [(main_id,)]
        # there is an array
        array_num = str(int(array_id[0]))  # trying to cast to int to make sure we understand
        return [(main_id, array_num)]


class SlurmInfoWatcher(core.InfoWatcher):
    def _make_command(self) -> Optional[List[str]]:
        # asking for array id will return all status
        # on the other end, asking for each and every one of them individually takes a huge amount of time
        to_check = {x.split("_")[0] for x in self._registered - self._finished}
        if not to_check:
            return None
        command = ["sacct", "-o", "JobID,State,NodeList", "--parsable2"]
        for jid in to_check:
            command.extend(["-j", str(jid)])
        return command

    def get_state(self, job_id: str, mode: str = "standard") -> str:
        """Returns the state of the job
        State of finished jobs are cached (use watcher.clear() to remove all cache)

        Parameters
        ----------
        job_id: int
            id of the job on the cluster
        mode: str
            one of "force" (forces a call), "standard" (calls regularly) or "cache" (does not call)
        """
        info = self.get_info(job_id, mode=mode)
        return info.get("State") or "UNKNOWN"

    def read_info(self, string: Union[bytes, str]) -> Dict[str, Dict[str, str]]:
        """Reads the output of sacct and returns a dictionary containing main information"""
        if not isinstance(string, str):
            string = string.decode()
        lines = string.splitlines()
        if len(lines) < 2:
            return {}  # one job id does not exist (yet)
        names = lines[0].split("|")
        # read all lines
        all_stats: Dict[str, Dict[str, str]] = {}
        for line in lines[1:]:
            stats = {x: y.strip() for x, y in zip(names, line.split("|"))}
            job_id = stats["JobID"]
            if not job_id or "." in job_id:
                continue
            try:
                multi_split_job_id = read_job_id(job_id)
            except Exception as e:
                # Array id are sometimes displayed with weird chars
                warnings.warn(
                    f"Could not interpret {job_id} correctly (please open an issue):\n{e}", DeprecationWarning
                )
                continue
            for split_job_id in multi_split_job_id:
                all_stats[
                    "_".join(split_job_id[:2])
                ] = stats  # this works for simple jobs, or job array unique instance
                # then, deal with ranges:
                if len(split_job_id) >= 3:
                    for index in range(int(split_job_id[1]), int(split_job_id[2]) + 1):
                        all_stats[f"{split_job_id[0]}_{index}"] = stats
        return all_stats


class SlurmJob(core.Job[core.R]):

    _cancel_command = "scancel"
    watcher = SlurmInfoWatcher(delay_s=600)

    def _interrupt(self, timeout: bool = False) -> None:
        """Sends preemption or timeout signal to the job (for testing purpose)

        Parameter
        ---------
        timeout: bool
            Whether to trigger a job time-out (if False, it triggers preemption)
        """
        cmd = ["scancel", self.job_id, "--signal"]
        # in case of preemption, SIGTERM is sent first
        if not timeout:
            subprocess.check_call(cmd + ["SIGTERM"])
        subprocess.check_call(cmd + [SlurmJobEnvironment.USR_SIG])


class SlurmParseException(Exception):
    pass


def _expand_id_suffix(suffix_parts: str) -> List[str]:
    """Parse the a suffix formatted like "1-3,5,8" into
    the list of numeric values 1,2,3,5,8.
    """
    suffixes = []
    for suffix_part in suffix_parts.split(","):
        if "-" in suffix_part:
            low, high = suffix_part.split("-")
            int_length = len(low)
            for num in range(int(low), int(high) + 1):
                suffixes.append(f"{num:0{int_length}}")
        else:
            suffixes.append(suffix_part)
    return suffixes


def _parse_node_group(node_list: str, pos: int, parsed: List[str]) -> int:
    """Parse a node group of the form PREFIX[1-3,5,8] and return
    the position in the string at which the parsing stopped
    """
    prefixes = [""]
    while pos < len(node_list):
        c = node_list[pos]
        if c == ",":
            parsed.extend(prefixes)
            return pos + 1
        if c == "[":
            last_pos = node_list.index("]", pos)
            suffixes = _expand_id_suffix(node_list[pos + 1 : last_pos])
            prefixes = [prefix + suffix for prefix in prefixes for suffix in suffixes]
            pos = last_pos + 1
        else:
            for i, prefix in enumerate(prefixes):
                prefixes[i] = prefix + c
            pos += 1
    parsed.extend(prefixes)
    return pos


def _parse_node_list(node_list: str):
    try:
        pos = 0
        parsed: List[str] = []
        while pos < len(node_list):
            pos = _parse_node_group(node_list, pos, parsed)
        return parsed
    except ValueError as e:
        raise SlurmParseException(f"Unrecognized format for SLURM_JOB_NODELIST: '{node_list}'", e) from e


class SlurmJobEnvironment(job_environment.JobEnvironment):

    _env = {
        "job_id": "SLURM_JOB_ID",
        "num_tasks": "SLURM_NTASKS",
        "num_nodes": "SLURM_JOB_NUM_NODES",
        "node": "SLURM_NODEID",
        "nodes": "SLURM_JOB_NODELIST",
        "global_rank": "SLURM_PROCID",
        "local_rank": "SLURM_LOCALID",
        "array_job_id": "SLURM_ARRAY_JOB_ID",
        "array_task_id": "SLURM_ARRAY_TASK_ID",
    }

    def _requeue(self, countdown: int) -> None:
        jid = self.job_id
        subprocess.check_call(["scontrol", "requeue", jid], timeout=60)
        logger.get_logger().info(f"Requeued job {jid} ({countdown} remaining timeouts)")

    @property
    def hostnames(self) -> List[str]:
        """Parse the content of the "SLURM_JOB_NODELIST" environment variable,
        which gives access to the list of hostnames that are part of the current job.

        In SLURM, the node list is formatted NODE_GROUP_1,NODE_GROUP_2,...,NODE_GROUP_N
        where each node group is formatted as: PREFIX[1-3,5,8] to define the hosts:
        [PREFIX1, PREFIX2, PREFIX3, PREFIX5, PREFIX8].

        Link: https://hpcc.umd.edu/hpcc/help/slurmenv.html
        """

        node_list = os.environ.get(self._env["nodes"], "")
        if not node_list:
            return [self.hostname]
        return _parse_node_list(node_list)


class SlurmExecutor(core.PicklingExecutor):
    """Slurm job executor
    This class is used to hold the parameters to run a job on slurm.
    In practice, it will create a batch file in the specified directory for each job,
    and pickle the task function and parameters. At completion, the job will also pickle
    the output. Logs are also dumped in the same directory.

    Parameters
    ----------
    folder: Path/str
        folder for storing job submission/output and logs.
    max_num_timeout: int
        Maximum number of time the job can be requeued after timeout (if
        the instance is derived from helpers.Checkpointable)

    Note
    ----
    - be aware that the log/output folder will be full of logs and pickled objects very fast,
      it may need cleaning.
    - the folder needs to point to a directory shared through the cluster. This is typically
      not the case for your tmp! If you try to use it, slurm will fail silently (since it
      will not even be able to log stderr.
    - use update_parameters to specify custom parameters (n_gpus etc...). If you
      input erroneous parameters, an error will print all parameters available for you.
    """

    job_class = SlurmJob

    def __init__(self, folder: Union[Path, str], max_num_timeout: int = 3) -> None:
        super().__init__(folder, max_num_timeout)
        if not self.affinity() > 0:
            raise RuntimeError('Could not detect "srun", are you indeed on a slurm cluster?')

    @classmethod
    def _equivalence_dict(cls) -> core.EquivalenceDict:
        return {
            "name": "job_name",
            "timeout_min": "time",
            "mem_gb": "mem",
            "nodes": "nodes",
            "cpus_per_task": "cpus_per_task",
            "gpus_per_node": "gpus_per_node",
            "tasks_per_node": "ntasks_per_node",
        }

    @classmethod
    def _valid_parameters(cls) -> Set[str]:
        """Parameters that can be set through update_parameters"""
        return set(_get_default_parameters())

    def _convert_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params = super()._convert_parameters(params)
        # replace type in some cases
        if "mem" in params:
            params["mem"] = _convert_mem(params["mem"])
        return params

    def _internal_update_parameters(self, **kwargs: Any) -> None:
        """Updates sbatch submission file parameters

        Parameters
        ----------
        See slurm documentation for most parameters.
        Most useful parameters are: time, mem, gpus_per_node, cpus_per_task, partition
        Below are the parameters that differ from slurm documentation:

        signal_delay_s: int
            delay between the kill signal and the actual kill of the slurm job.
        setup: list
            a list of command to run in sbatch befure running srun
        array_parallelism: int
            number of map tasks that will be executed in parallel

        Raises
        ------
        ValueError
            In case an erroneous keyword argument is added, a list of all eligible parameters
            is printed, with their default values

        Note
        ----
        Best practice (as far as Quip is concerned): cpus_per_task=2x (number of data workers + gpus_per_task)
        You can use cpus_per_gpu=2 (requires using gpus_per_task and not gpus_per_node)
        """
        defaults = _get_default_parameters()
        in_valid_parameters = sorted(set(kwargs) - set(defaults))
        if in_valid_parameters:
            string = "\n  - ".join(f"{x} (default: {repr(y)})" for x, y in sorted(defaults.items()))
            raise ValueError(
                f"Unavailable parameter(s): {in_valid_parameters}\nValid parameters are:\n  - {string}"
            )
        # check that new parameters are correct
        _make_sbatch_string(command="nothing to do", folder=self.folder, **kwargs)
        super()._internal_update_parameters(**kwargs)

    def _internal_process_submissions(
        self, delayed_submissions: tp.List[utils.DelayedSubmission]
    ) -> tp.List[core.Job[tp.Any]]:
        if len(delayed_submissions) == 1:
            return super()._internal_process_submissions(delayed_submissions)
        # array
        folder = utils.JobPaths.get_first_id_independent_folder(self.folder)
        folder.mkdir(parents=True, exist_ok=True)
        timeout_min = self.parameters.get("time", 5)
        pickle_paths = []
        for d in delayed_submissions:
            pickle_path = folder / f"{uuid.uuid4().hex}.pkl"
            d.set_timeout(timeout_min, self.max_num_timeout)
            d.dump(pickle_path)
            pickle_paths.append(pickle_path)
        n = len(delayed_submissions)
        # Make a copy of the executor, since we don't want other jobs to be
        # scheduled as arrays.
        array_ex = SlurmExecutor(self.folder, self.max_num_timeout)
        array_ex.update_parameters(**self.parameters)
        array_ex.parameters["map_count"] = n
        self._throttle()

        first_job: core.Job[tp.Any] = array_ex._submit_command(self._submitit_command_str)
        tasks_ids = list(range(first_job.num_tasks))
        jobs: List[core.Job[tp.Any]] = [
            SlurmJob(folder=self.folder, job_id=f"{first_job.job_id}_{a}", tasks=tasks_ids) for a in range(n)
        ]
        for job, pickle_path in zip(jobs, pickle_paths):
            job.paths.move_temporary_file(pickle_path, "submitted_pickle")
        return jobs

    @property
    def _submitit_command_str(self) -> str:
        return " ".join(
            [shlex.quote(sys.executable), "-u -m submitit.core._submit", shlex.quote(str(self.folder))]
        )

    def _make_submission_file_text(self, command: str, uid: str) -> str:
        return _make_sbatch_string(command=command, folder=self.folder, **self.parameters)

    def _num_tasks(self) -> int:
        nodes: int = self.parameters.get("nodes", 1)
        tasks_per_node: int = max(1, self.parameters.get("ntasks_per_node", 1))
        return nodes * tasks_per_node

    def _make_submission_command(self, submission_file_path: Path) -> List[str]:
        return ["sbatch", str(submission_file_path)]

    @staticmethod
    def _get_job_id_from_submission_command(string: Union[bytes, str]) -> str:
        """Returns the job ID from the output of sbatch string"""
        if not isinstance(string, str):
            string = string.decode()
        output = re.search(r"job (?P<id>[0-9]+)", string)
        if output is None:
            raise utils.FailedSubmissionError(
                f'Could not make sense of sbatch output "{string}"\n'
                "Job instance will not be able to fetch status\n"
                "(you may however set the job job_id manually if needed)"
            )
        return output.group("id")

    @classmethod
    def affinity(cls) -> int:
        return -1 if shutil.which("srun") is None else 2


@functools.lru_cache()
def _get_default_parameters() -> Dict[str, Any]:
    """Parameters that can be set through update_parameters"""
    specs = inspect.getfullargspec(_make_sbatch_string)
    zipped = zip(specs.args[-len(specs.defaults) :], specs.defaults)  # type: ignore
    return {key: val for key, val in zipped if key not in {"command", "folder", "map_count"}}


# pylint: disable=too-many-arguments,unused-argument, too-many-locals
def _make_sbatch_string(
    command: str,
    folder: tp.Union[str, Path],
    job_name: str = "submitit",
    partition: tp.Optional[str] = None,
    time: int = 5,
    nodes: int = 1,
    ntasks_per_node: tp.Optional[int] = None,
    cpus_per_task: tp.Optional[int] = None,
    cpus_per_gpu: tp.Optional[int] = None,
    num_gpus: tp.Optional[int] = None,  # legacy
    gpus_per_node: tp.Optional[int] = None,
    gpus_per_task: tp.Optional[int] = None,
    qos: tp.Optional[str] = None,  # quality of service
    setup: tp.Optional[tp.List[str]] = None,
    mem: tp.Optional[str] = None,
    mem_per_gpu: tp.Optional[str] = None,
    mem_per_cpu: tp.Optional[str] = None,
    signal_delay_s: int = 90,
    comment: tp.Optional[str] = None,
    constraint: tp.Optional[str] = None,
    exclude: tp.Optional[str] = None,
    account: tp.Optional[str] = None,
    gres: tp.Optional[str] = None,
    exclusive: tp.Optional[tp.Union[bool, str]] = None,
    array_parallelism: int = 256,
    wckey: str = "submitit",
    stderr_to_stdout: bool = False,
    map_count: tp.Optional[int] = None,  # used internally
    additional_parameters: tp.Optional[tp.Dict[str, tp.Any]] = None,
    srun_args: tp.Optional[tp.Iterable[str]] = None,
) -> str:
    """Creates the content of an sbatch file with provided parameters

    Parameters
    ----------
    See slurm sbatch documentation for most parameters:
    https://slurm.schedmd.com/sbatch.html

    Below are the parameters that differ from slurm documentation:

    folder: str/Path
        folder where print logs and error logs will be written
    signal_delay_s: int
        delay between the kill signal and the actual kill of the slurm job.
    setup: list
        a list of command to run in sbatch before running srun
    map_size: int
        number of simultaneous map/array jobs allowed
    additional_parameters: dict
        Forces any parameter to a given value in sbatch. This can be useful
        to add parameters which are not currently available in submitit.
        Eg: {"mail-user": "blublu@fb.com", "mail-type": "BEGIN"}
    srun_args: List[str]
        Add each argument in the list to the srun call

    Raises
    ------
    ValueError
        In case an erroneous keyword argument is added, a list of all eligible parameters
        is printed, with their default values
    """
    nonslurm = [
        "nonslurm",
        "folder",
        "command",
        "map_count",
        "array_parallelism",
        "additional_parameters",
        "setup",
        "signal_delay_s",
        "stderr_to_stdout",
        "srun_args",
    ]
    parameters = {k: v for k, v in locals().items() if v is not None and k not in nonslurm}
    # rename and reformat parameters
    parameters["signal"] = f"{SlurmJobEnvironment.USR_SIG}@{signal_delay_s}"
    if num_gpus is not None:
        warnings.warn(
            '"num_gpus" is deprecated, please use "gpus_per_node" instead (overwritting with num_gpus)'
        )
        parameters["gpus_per_node"] = parameters.pop("num_gpus", 0)
    if "cpus_per_gpu" in parameters and "gpus_per_task" not in parameters:
        warnings.warn('"cpus_per_gpu" requires to set "gpus_per_task" to work (and not "gpus_per_node")')
    # add necessary parameters
    paths = utils.JobPaths(folder=folder)
    stdout = str(paths.stdout)
    stderr = str(paths.stderr)
    # Job arrays will write files in the form  <ARRAY_ID>_<ARRAY_TASK_ID>_<TASK_ID>
    if map_count is not None:
        assert isinstance(map_count, int) and map_count
        parameters["array"] = f"0-{map_count - 1}%{min(map_count, array_parallelism)}"
        stdout = stdout.replace("%j", "%A_%a")
        stderr = stderr.replace("%j", "%A_%a")
    parameters["output"] = stdout.replace("%t", "0")
    if not stderr_to_stdout:
        parameters["error"] = stderr.replace("%t", "0")
    parameters["open-mode"] = "append"
    if additional_parameters is not None:
        parameters.update(additional_parameters)
    # now create
    lines = ["#!/bin/bash", "", "# Parameters"]
    for k in sorted(parameters):
        lines.append(_as_sbatch_flag(k, parameters[k]))
    # environment setup:
    if setup is not None:
        lines += ["", "# setup"] + setup
    # commandline (this will run the function and args specified in the file provided as argument)
    # We pass --output and --error here, because the SBATCH command doesn't work as expected with a filename pattern
    stderr_flags = [] if stderr_to_stdout else ["--error", stderr]
    if srun_args is None:
        srun_args = []

    srun_cmd = _shlex_join(["srun", "--unbuffered", "--output", stdout, *stderr_flags, *srun_args])
    lines += [
        "",
        "# command",
        "export SUBMITIT_EXECUTOR=slurm",
        # The input "command" is supposed to be a valid shell command
        " ".join((srun_cmd, command)),
        "",
    ]
    return "\n".join(lines)


def _convert_mem(mem_gb: float) -> str:
    if mem_gb == int(mem_gb):
        return f"{int(mem_gb)}GB"
    return f"{int(mem_gb * 1024)}MB"


def _as_sbatch_flag(key: str, value: tp.Any) -> str:
    key = key.replace("_", "-")
    if value is True:
        return f"#SBATCH --{key}"

    value = shlex.quote(str(value))
    return f"#SBATCH --{key}={value}"


def _shlex_join(split_command: tp.List[str]) -> str:
    """Same as shlex.join, but that was only added in Python 3.8"""
    return " ".join(shlex.quote(arg) for arg in split_command)
