# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""""Python 3.6+ toolbox for submitting jobs to Slurm"""

# allow explicit reimports (mypy) by renaming all imports
from . import helpers as helpers
from .auto.auto import AutoExecutor as AutoExecutor
from .core.core import Executor as Executor
from .core.core import Job as Job
from .core.job_environment import JobEnvironment as JobEnvironment
from .local.debug import DebugExecutor as DebugExecutor
from .local.debug import DebugJob as DebugJob
from .local.local import LocalExecutor as LocalExecutor
from .local.local import LocalJob as LocalJob
from .slurm.slurm import SlurmExecutor as SlurmExecutor
from .slurm.slurm import SlurmJob as SlurmJob

__version__ = "1.4.5"
