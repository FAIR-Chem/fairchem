"""
Classes to represent electrocatalysts environments
"""
import itertools
import numpy as np
import pandas as pd
from gflownetenv import GFlowNetEnv


class Electrocatalyst(GFlowNetEnv):
    """
    Electrocatalyst environment

    Attributes
    ----------
    """

    def __init__(
        self,
        env_id=None,
        reward_beta=1,
        reward_norm=1.0,
        denorm_proxy=False,
        energies_stats=None,
        proxy=None,
        oracle_func="default",
        debug=False,
    ):
        super(Grid, self).__init__(
            env_id,
            reward_beta,
            reward_norm,
            energies_stats,
            denorm_proxy,
            proxy,
            oracle_func,
            debug,
        )
