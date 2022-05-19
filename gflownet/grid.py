"""
Classes to represent a hyper-grid environments
"""
import itertools
import numpy as np
import pandas as pd
from gflownetenv import GFlowNetEnv


class Grid(GFlowNetEnv):
    """
    Hyper-grid environment

    Attributes
    ----------
    ndim : int
        Dimensionality of the grid

    length : int
        Size of the grid (cells per dimension)

    cell_min : float
        Lower bound of the cells range

    cell_max : float
        Upper bound of the cells range
    """

    def __init__(
        self,
        n_dim=2,
        length=4,
        min_step_len=1,
        max_step_len=1,
        cell_min=-1,
        cell_max=1,
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
        self.state = [0] * self.n_dim
        self.n_dim = n_dim
        self.length = length
        self.obs_dim = self.length * self.n_dim
        self.min_step_len = min_step_len
        self.max_step_len = max_step_len
        self.cells = np.linspace(cell_min, cell_max, length)
        self.oracle = {
            "default": None,
            "cos_N": self.func_cos_N,
            "corners": self.func_corners,
            "corners_floor_A": self.func_corners_floor_A,
            "corners_floor_B": self.func_corners_floor_B,
        }[oracle_func]
        if proxy:
            self.proxy = proxy
        else:
            self.proxy = self.oracle
        self.reward = (
            lambda x: [0]
            if not self.done
            else self.proxy2reward(self.proxy(self.state2oracle(x)))
        )
        self._true_density = None
        self.denorm_proxy = denorm_proxy
        self.action_space = self.get_actions_space()
        self.eos = len(self.action_space)
        # Aliases and compatibility
        self.seq = self.state
        self.seq2obs = self.state2obs
        self.obs2seq = self.obs2state
        self.seq2oracle = self.state2oracle
        self.letters2seq = self.readable2state

    def get_actions_space(self):
        """
        Constructs list with all possible actions
        """
        valid_steplens = np.arange(self.min_step_len, self.max_step_len + 1)
        dims = [a for a in range(self.n_dim)]
        actions = []
        for r in valid_steplens:
            actions_r = [el for el in itertools.product(dims, repeat=r)]
            actions += actions_r
        return actions

    def state2oracle(self, state_list):
        """
        Prepares a list of states in "GFlowNet format" for the oracles: a list of length
        n_dim with values in the range [cell_min, cell_max] for each state.

        Args
        ----
        state_list : list of lists
            List of states.
        """
        return [
            (
                self.state2obs(state).reshape((self.n_dim, self.length))
                * self.cells[None, :]
            ).sum(axis=1)
            for state in state_list
        ]

    def state2obs(self, state=None):
        """
        Transforms the state given as argument (or self.state if None) into a
        one-hot encoding. The output is a list of len length * n_dim,
        where each n-th successive block of length elements is a one-hot encoding of
        the position in the n-th dimension.

        Example:
          - State, state: [0, 3, 1] (n_dim = 3)
          - state2obs(state): [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0] (length = 4)
                              |     0    |      3    |      1    |
        """
        if state is None:
            state = self.state
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        obs[(np.arange(len(state)) * self.length + state)] = 1
        return obs

    def obs2state(self, obs):
        """
        Transforms the one-hot encoding version of a state given as argument
        into a state (list of the position at each dimension).

        Example:
          - obs: [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0] (length = 4, n_dim = 3)
                 |     0    |      3    |      1    |
          - obs2state(obs): [0, 3, 1]
        """
        obs_mat = np.reshape(obs, (self.n_dim, self.length))
        state = np.where(obs_mat)[1]
        return state

    def readable2state(self, readable, alphabet={}):
        """
        Converts a human-readable string representing a state into a state as a list of
        positions.
        """
        return [int(el) for el in readable.strip("[]").split(" ")]

    def reset(self, env_id=None):
        """
        Resets the environment.
        """
        self.state = [0] * self.n_dim
        self.n_actions = 0
        self.done = False
        self.id = env_id
        return self

    def parent_transitions(self, state, action):
        """
        Determines all parents and actions that lead to state.

        Args
        ----
        state : list
            Representation of a state, as a list of length length where each element is
            the position at each dimension.

        action : int
            Last action performed

        Returns
        -------
        parents : list
            List of parents as state2obs(state)

        actions : list
            List of actions that lead to state for each parent in parents
        """
        if action == self.eos:
            return [self.state2obs(state)], [action]
        else:
            parents = []
            actions = []
            for idx, a in enumerate(self.action_space):
                state_aux = state.copy()
                for a_sub in a:
                    if state_aux[a_sub] > 0:
                        state_aux[a_sub] -= 1
                    else:
                        break
                else:
                    parents.append(self.state2obs(state_aux))
                    actions.append(idx)
        return parents, actions

    def step(self, action):
        """
        Executes step given an action.

        Args
        ----
        a : int (tensor)
            Index of action in the action space. a == eos indicates "stop action"

        Returns
        -------
        self.state : list
            The sequence after executing the action

        valid : bool
            False, if the action is not allowed for the current state, e.g. stop at the
            root state
        """
        # All dimensions are at the maximum length
        if all([s == self.length - 1 for s in self.state]):
            self.done = True
            self.n_actions += 1
            return self.state, self.eos, True
        if action < self.eos:
            state_next = self.state.copy()
            if action.ndim == 0:
                action = [action]
            for a in action:
                state_next[a] += 1
            if any([s >= self.length for s in state_next]):
                valid = False
            else:
                self.state = state_next
                valid = True
                self.n_actions += 1
        else:
            self.done = True
            valid = True
            self.n_actions += 1

        return self.state, action, valid

    @staticmethod
    def func_corners(x_list):
        def _func_corners(x):
            ax = abs(x)
            return -1.0 * (
                (ax > 0.5).prod(-1) * 0.5
                + ((ax < 0.8) * (ax > 0.6)).prod(-1) * 2
                + 1e-1
            )

        return np.asarray([_func_corners(x) for x in x_list])

    @staticmethod
    def func_corners_floor_B(x_list):
        def _func_corners_floor_B(x_list):
            ax = abs(x)
            return -1.0 * (
                (ax > 0.5).prod(-1) * 0.5
                + ((ax < 0.8) * (ax > 0.6)).prod(-1) * 2
                + 1e-2
            )

        return np.asarray([_func_corners_floor_B(x) for x in x_list])

    @staticmethod
    def func_corners_floor_A(x_list):
        def _func_corners_floor_A(x_list):
            ax = abs(x)
            return -1.0 * (
                (ax > 0.5).prod(-1) * 0.5
                + ((ax < 0.8) * (ax > 0.6)).prod(-1) * 2
                + 1e-3
            )

        return np.asarray([_func_corners_floor_A(x) for x in x_list])

    @staticmethod
    def func_cos_N(x_list):
        def _func_cos_N(x_list):
            ax = abs(x)
            return -1.0 * (
                ((np.cos(x * 50) + 1) * norm.pdf(x * 5)).prod(-1) + 0.01
            )

        return np.asarray([_func_cos_N(x) for x in x_list])

    def make_train_set(self, ntrain, oracle=None, seed=168, output_csv=None):
        """
        Constructs a randomly sampled train set.

        Args
        ----
        """
        rng = np.random.default_rng(seed)
        samples = rng.integers(
            low=0, high=self.length, size=(ntrain,) + (self.n_dim,)
        )
        if oracle:
            energies = oracle(self.state2oracle(samples))
        else:
            energies = self.oracle(self.state2oracle(samples))
        df_train = pd.DataFrame(
            {"samples": list(samples), "energies": energies}
        )
        if output_csv:
            df_train.to_csv(output_csv)
        return df_train
