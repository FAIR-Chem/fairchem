# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import typing as tp
import warnings
from pathlib import Path
from typing import Any, List, Optional, Type, Union

from ..core import plugins
from ..core.core import Executor, Job
from ..core.utils import DelayedSubmission


def _convert_deprecated_args(kwargs: tp.Dict[str, Any], deprecated_args: tp.Mapping[str, str]) -> None:
    for arg in list(kwargs):
        new_arg = deprecated_args.get(arg)
        if not new_arg:
            continue
        kwargs[new_arg] = kwargs.pop(arg)
        warnings.warn(f"Setting '{arg}' is deprecated. Use '{new_arg}' instead.")


class AutoExecutor(Executor):
    """Automatic job executor
    This class is used to hold the parameters to run a job either on the cluster
    corresponding to the environment.
    It can also be used to run job locally or in debug mode.
    In practice, it will create a bash file in the specified directory for each job,
    and pickle the task function and parameters. At completion, the job will also pickle
    the output. Logs are also dumped in the same directory.

    Executor specific parameters must be specified by prefixing them with the name
    of the executor they refer to. eg:
        - 'chronos_conda_file' (internal)
        - 'slurm_max_num_timeout'
    See each executor documentation for the list of available parameters.

    Parameters
    ----------
    folder: Path/str
        folder for storing job submission/output and logs.
    warn_ignored: bool
        prints a warning each time a parameter is provided but ignored because it is only
        useful for the other cluster.
    cluster: str
        Forces AutoExecutor to use the given environment. Use "local" to run jobs locally,
        "debug" to run jobs in process.
    kwargs: other arguments must be prefixed by the name of the executor they refer to.
        {exname}_{argname}: see {argname} documentation in {Exname}Executor documentation.

    Note
    ----
    - be aware that the log/output folder will be full of logs and pickled objects very fast,
      it may need cleaning.
    - use update_parameters to specify custom parameters (gpus_per_node etc...). If you
      input erroneous parameters, an error will print all parameters available for you.
    """

    _ctor_deprecated_args = {"max_num_timeout": "slurm_max_num_timeout", "conda_file": "chronos_conda_file"}

    def __init__(self, folder: Union[str, Path], cluster: Optional[str] = None, **kwargs: Any) -> None:
        self.cluster = cluster or self.which()

        executors = plugins.get_executors()
        if self.cluster not in executors:
            raise ValueError(f"AutoExecutor doesn't know any executor named {self.cluster}")

        _convert_deprecated_args(kwargs, self._ctor_deprecated_args)
        err = "Extra arguments must be prefixed by executor named, received unknown arg"
        err_ex_list = f"Known executors: {', '.join(executors)}."
        for name in kwargs:
            assert "_" in name, f"{err} '{name}'. {err_ex_list}"
            prefix = name.split("_")[0]
            assert (
                prefix in executors
            ), f"{err} '{name}', and '{prefix}' executor is also unknown. {err_ex_list}"
        self._executor = flexible_init(executors[self.cluster], folder, **kwargs)

        valid = self._valid_parameters()
        self._deprecated_args = {
            arg: f"{ex_name}_{arg}"
            for ex_name, ex in executors.items()
            for arg in ex._valid_parameters()
            if arg not in valid
        }
        super().__init__(self._executor.folder, self._executor.parameters)

    @staticmethod
    def which() -> str:
        """Returns what is the detected cluster."""
        executors = plugins.get_executors()
        best_ex = max(executors, key=lambda ex: executors[ex].affinity())

        if executors[best_ex].affinity() <= 0:
            raise RuntimeError(f"Did not found an available executor among {executors.keys()}.")

        return best_ex

    def register_dev_folders(self, folders: List[Union[str, Path]]) -> None:
        """Archive a list of folders to be untarred in the job working directory.
        This is only implemented for internal cluster, for running job on non-installed packages.
        This is not useful on slurm since the working directory of jobs is identical to
        your work station working directory.

        folders: list of paths
            The list of folders to archive and untar in the job working directory
        """
        register = getattr(self._executor, "register_dev_folders", None)
        if register is not None:
            register(folders)
        else:
            # TODO this should be done through update parameters
            warnings.warn(
                "Ignoring dev folder registration as it is only supported (and needed) for internal cluster"
            )

    @classmethod
    def _typed_parameters(cls) -> tp.Dict[str, Type]:
        return {
            "name": str,
            "timeout_min": int,
            "mem_gb": float,
            "nodes": int,
            "cpus_per_task": int,
            "gpus_per_node": int,
            "tasks_per_node": int,
            "stderr_to_stdout": bool,
        }

    @classmethod
    def _valid_parameters(cls) -> tp.Set[str]:
        return set(cls._typed_parameters().keys())

    def _internal_update_parameters(self, **kwargs: Any) -> None:
        """Updates submission parameters to srun/crun.

        Parameters
        ----------
        AutoExecutors provides shared parameters that are translated for each specific cluster.
        Those are: timeout_min (int), mem_gb (int), gpus_per_node (int), cpus_per_task (int),
        nodes (int), tasks_per_node (int) and name (str).
        Cluster specific parameters can be specified by prefixing them with the cluster name.

        Notes
        -----
        - Cluster specific parameters win over shared parameters.
            eg: if both `slurm_time` and `timeout_min` are provided, then:
                - `slurm_time` is used on the slurm cluster
                - `timeout_min` is used on other clusters
        """
        # We handle None as not set.
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        # check type of replaced variables
        generics = AutoExecutor._typed_parameters()
        for name, expected_type in generics.items():
            if expected_type == float:
                expected_type = (int, float)  # type: ignore
            if name in kwargs:
                assert isinstance(kwargs[name], expected_type), (
                    f'Parameter "{name}" expected type {expected_type} ' f'(but value: "{kwargs[name]}")'
                )

        _convert_deprecated_args(kwargs, self._deprecated_args)
        specific = [x.split("_", 1) for x in kwargs if x not in generics]

        invalid = []
        executors = plugins.get_executors()
        for ex_arg in specific:
            if len(ex_arg) != 2:
                invalid.append(f"Parameter '{ex_arg[0]}' need to be prefixed by an executor name.")
                continue
            ex, arg = ex_arg

            if ex not in executors:
                invalid.append(f"Unknown executor '{ex}' in parameter '{ex}_{arg}'.")
                continue

            valid = executors[ex]._valid_parameters()
            if arg not in valid and arg not in generics:
                invalid.append(
                    f"Unknown argument '{arg}' for executor '{ex}' in parameter '{ex}_{arg}'."
                    + " Valid arguments: "
                    + ", ".join(valid)
                )
                continue
        if invalid:
            invalid.extend(
                [
                    f"Known executors: {', '.join(executors.keys())}",
                    f"As a reminder, shared/generic (non-prefixed) parameters are: {generics}.",
                ]
            )
            raise NameError("\n".join(invalid))

        # add cluster specific generic overrides
        kwargs.update(
            **{
                arg: kwargs.pop(f"{ex}_{arg}")
                for ex, arg in specific
                if ex == self.cluster and arg in generics
            }
        )
        parameters = self._executor._convert_parameters({k: kwargs[k] for k in kwargs if k in generics})
        # update parameters in the core executor
        for (ex, arg) in specific:
            # update cluster specific non-generic arguments
            if arg not in generics and ex == self.cluster:
                parameters[arg] = kwargs[f"{ex}_{arg}"]

        self._executor._internal_update_parameters(**parameters)

    def _internal_process_submissions(
        self, delayed_submissions: tp.List[DelayedSubmission]
    ) -> tp.List[Job[tp.Any]]:
        return self._executor._internal_process_submissions(delayed_submissions)


def flexible_init(cls: Type[Executor], folder: Union[str, Path], **kwargs: Any) -> Executor:
    prefix = cls.name() + "_"
    return cls(folder, **{k[len(prefix) :]: val for k, val in kwargs.items() if k.startswith(prefix)})
