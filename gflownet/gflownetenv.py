"""
Base class of GFlowNet environments
"""
import numpy as np
import pandas as pd


class GFlowNetEnv:
    """
    Base class of GFlowNet environments
    """

    def __init__(
        self,
        env_id=None,
        reward_beta=1,
        reward_norm=1.0,
        energies_stats=None,
        denorm_proxy=False,
        proxy=None,
        oracle_func=None,
        debug=False,
    ):
        self.state = []
        self.done = False
        self.n_actions = 0
        self.id = env_id
        self.min_reward = 1e-8
        self.reward_beta = reward_beta
        self.reward_norm = reward_norm
        self.energies_stats = energies_stats
        self.denorm_proxy = denorm_proxy
        self.oracle = oracle_func
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
        self.debug = debug
        self.action_space = []
        self.eos = len(self.action_space)
        self.max_path_len = self.get_max_path_len()
        # Assertions
        assert self.reward_norm > 0
        assert self.reward_beta > 0
        assert self.min_reward > 0

    def set_energies_stats(self, energies_stats):
        self.energies_stats = energies_stats

    def set_reward_norm(self, reward_norm):
        self.reward_norm = reward_norm

    def get_actions_space(self):
        """
        Constructs list with all possible actions (excluding end of sequence)
        """
        return []

    def get_max_path_len(
        self,
    ):
        return 1

    def state2oracle(self, state_list):
        """
        Prepares a list of states in "GFlowNet format" for the oracle

        Args
        ----
        state_list : list of lists
            List of states.
        """
        return state_list

    def reward_batch(self, states, done):
        """
        Computes the rewards of a batch of states, given a list of states and 'dones'
        """
        states = [s for s, d in zip(states, done) if d]
        reward = np.zeros(len(done))
        reward[list(done)] = self.proxy2reward(
            self.proxy(self.state2oracle(states))
        )
        return reward

    def proxy2reward(self, proxy_vals):
        """
        Prepares the output of an oracle for GFlowNet: the inputs proxy_vals is
        expected to be a negative value (energy), unless self.denorm_proxy is True. If
        the latter, the proxy values are first de-normalized according to the mean and
        standard deviation in self.energies_stats. The output of the function is a
        strictly positive reward - provided self.reward_norm and self.reward_beta are
        positive - and larger than self.min_reward.
        """
        if self.denorm_proxy:
            proxy_vals = (
                proxy_vals * self.energies_stats[3] + self.energies_stats[2]
            )
        return np.clip(
            (-1.0 * proxy_vals / self.reward_norm) ** self.reward_beta,
            self.min_reward,
            None,
        )

    def reward2proxy(self, reward):
        """
        Converts a "GFlowNet reward" into a (negative) energy or values as returned by
        an oracle.
        """
        return -np.exp(
            (np.log(reward) + self.reward_beta * np.log(self.reward_norm))
            / self.reward_beta
        )

    def state2obs(self, state=None):
        """
        Converts a state into a format suitable for a machine learning model, such as a
        one-hot encoding.
        """
        if state is None:
            state = self.state
        return state

    def obs2state(self, obs):
        """
        Converts the model (e.g. one-hot encoding) version of a state given as
        argument into a state.
        """
        return obs

    def state2readable(self, state=None):
        """
        Converts a state into human-readable representation.
        """
        if state is None:
            state = self.state
        return str(state)

    def readable2state(self, readable):
        """
        Converts a human-readable representation of a state into the standard format.
        """
        return readable

    def reset(self, env_id=None):
        """
        Resets the environment.
        """
        self.state = []
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
            Representation of a state

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
        return parents, actions

    def get_paths(self, path_list, actions):
        """
        Determines all paths leading to each state in path_list, recursively.

        Args
        ----
        path_list : list
            List of paths (lists)

        actions : list
            List of actions within each path

        Returns
        -------
        path_list : list
            List of paths (lists)

        actions : list
            List of actions within each path
        """
        current_path = path_list[-1].copy()
        current_path_actions = actions[-1].copy()
        parents, parents_actions = self.parent_transitions(
            list(current_path[-1]), -1
        )
        parents = [self.obs2state(el).tolist() for el in parents]
        if parents == []:
            return path_list, actions
        for idx, (p, a) in enumerate(zip(parents, parents_actions)):
            if idx > 0:
                path_list.append(current_path)
                actions.append(current_path_actions)
            path_list[-1] += [p]
            actions[-1] += [a]
            path_list, actions = self.get_paths(path_list, actions)
        return path_list, actions

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
        if action < self.eos:
            self.done = False
            valid = True
        else:
            self.done = True
            valid = True
            self.n_actions += 1
        return self.state, action, valid

    def no_eos_mask(self, state=None):
        """
        Returns True if no eos action is allowed given state
        """
        if state is None:
            state = self.state
        return False

    def true_density(self):
        """
        Computes the reward density (reward / sum(rewards)) of the whole space

        Returns
        -------
        Tuple:
          - normalized reward for each state
          - states
          - (un-normalized) reward)
        """
        return (None, None, None)

    def make_train_set(self, ntrain, oracle=None, seed=168, output_csv=None):
        """
        Constructs a randomly sampled train set.

        Args
        ----
        """
        return None

    def make_test_set(
        self,
        ntest,
        oracle=None,
        seed=167,
        output_csv=None,
    ):
        """
        Constructs a test set.

        Args
        ----
        """
        return None

    @staticmethod
    def np2df(*args):
        """
        Args
        ----
        """
        return None


class ReplayBuffer:
    def __init__(self, capacity, env, output_csv=None):
        self.capacity = capacity
        self.env = env
        self.action_space = self.env.get_actions_space()
        self.buffer = pd.DataFrame(
            columns=["readable", "reward", "energy", "iter"]
        )

    def add(
        self,
        states,
        paths,
        rewards,
        energies,
        it,
        criterion="better",
    ):
        pass

    def _add_better(
        self,
        rewards_batch,
    ):
        rewards_buffer = self.buffer["rewards"]

    def sample(
        self,
    ):
        pass

    def __len__(self):
        return self.capacity

    @property
    def transitions(self):
        pass

    def save(
        self,
    ):
        pass

    @classmethod
    def load():
        pass

    @property
    def dummy(self):
        pass
