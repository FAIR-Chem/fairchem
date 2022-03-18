"""
Classes to represent electrocatalysts environments

Important: WIP (copied from apatamers)
"""
import itertools
import numpy as np


class AptamerSeq:
    """
    Aptamer sequence environment

    Attributes
    ----------
    max_seq_length : int
        Maximum length of the sequences

    min_seq_length : int
        Minimum length of the sequences

    nalphabet : int
        Number of letters in the alphabet

    seq : list
        Representation of a sequence (state), as a list of length max_seq_length where
        each element is the index of a letter in the alphabet, from 0 to (nalphabet -
        1).

    done : bool
        True if the sequence has reached a terminal state (maximum length, or stop
        action executed.

    func : str
        Name of the reward function

    n_actions : int
        Number of actions applied to the sequence

    proxy : lambda
        Proxy model
    """

    def __init__(
        self,
        max_seq_length=42,
        min_seq_length=1,
        nalphabet=4,
        min_word_len=1,
        max_word_len=1,
        proxy=None,
        allow_backward=False,
        debug=False,
        reward_beta=1,
        env_id=None,
        oracle_func=None,
        stats_scores=[-1.0, 0.0, 0.5, 1.0, -1.0],
        reward_norm=1.0,
    ):
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.nalphabet = nalphabet
        self.min_word_len = min_word_len
        self.max_word_len = max_word_len
        self.seq = []
        self.done = False
        self.id = env_id
        self.n_actions = 0
        self.stats_scores = stats_scores
        self.oracle = oracle_func
        if proxy:
            self.proxy = proxy
        else:
            self.proxy = self.oracle
        self.reward = (
            lambda x: [0]
            if not self.done
            else self.proxy2reward(self.proxy(self.seq2oracle(x)))
        )
        self.allow_backward = allow_backward
        self._true_density = None
        self.debug = debug
        self.reward_beta = reward_beta
        self.min_reward = 1e-8
        self.reward_norm = reward_norm
        self.action_space = self.get_actions_space(
            self.nalphabet, np.arange(self.min_word_len, self.max_word_len + 1)
        )
        self.eos = len(self.action_space)

    def get_actions_space(self, nalphabet, valid_wordlens):
        """
        Constructs with all possible actions
        """
        alphabet = [a for a in range(nalphabet)]
        actions = []
        for r in valid_wordlens:
            actions_r = [el for el in itertools.product(alphabet, repeat=r)]
            actions += actions_r
        return actions

    def reward_arbitrary_i(self, seq):
        if len(seq) > 0:
            return (seq[-1] + 1) * len(seq)
        else:
            return 0

    def seq2oracle(self, seq):
        """
        Prepares a sequence in "GFlowNet format" for the oracles.

        Args
        ----
        seq : list of lists
            List of sequences.
        """
        queries = [s + [-1] * (self.max_seq_length - len(s)) for s in seq]
        queries = np.array(queries, dtype=int)
        if queries.ndim == 1:
            queries = queries[np.newaxis, ...]
        queries += 1
        if queries.shape[1] == 1:
            import ipdb

            ipdb.set_trace()
            queries = np.column_stack((queries, np.zeros(queries.shape[0])))
        return queries

    def reward_batch(self, seq, done):
        seq = [s for s, d in zip(seq, done) if d]
        reward = np.zeros(len(done))
        reward[list(done)] = self.proxy2reward(self.proxy(self.seq2oracle(seq)))
        return reward

    def proxy2reward(self, proxy_vals):
        """
        Prepares the output of an oracle for GFlowNet.
        self.stats_scores:
            [0]: min
            [1]: max
            [2]: mean
            [3]: std
            [4]: max after norm
        """
        # Normalize
        rewards = (
            np.min(
                np.stack([np.zeros(proxy_vals.shape[0]), proxy_vals], axis=0), axis=0
            )
            - self.stats_scores[2]
        ) / self.stats_scores[3]
        # Invert and shift to the right (add maximum of normalized train distribution)
        rewards = self.stats_scores[4] - rewards
        # Re-normalize to ~[0, 1] and distort
        rewards = (rewards / self.reward_norm) ** self.reward_beta
        # Clip
        rewards = np.max(
            np.stack(
                [self.min_reward * np.ones(rewards.shape[0]), rewards], axis=0
            ),
            axis=0,
        )
        return rewards

    def reward2proxy(self, reward):
        """
        Converts a "GFlowNet reward" into energy or values as returned by an oracle.
        """
        # TODO: rewrite
        proxy_vals = np.exp((np.log(reward) + self.reward_beta * np.log(self.reward_norm)) / self.reward_beta)
        proxy_vals = self.stats_scores[4] - proxy_vals
        proxy_vals = proxy_vals * self.stats_scores[3] + self.stats_scores[2]
        return proxy_vals

    def seq2obs(self, seq=None):
        """
        Transforms the sequence (state) given as argument (or self.seq if None) into a
        one-hot encoding. The output is a list of length nalphabet * max_seq_length,
        where each n-th successive block of nalphabet elements is a one-hot encoding of
        the letter in the n-th position.

        Example:
          - Sequence: AATGC
          - State, seq: [0, 0, 1, 3, 2]
                         A, A, T, G, C
          - seq2obs(seq): [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]
                          |     A    |      A    |      T    |      G    |      C    |

        If max_seq_length > len(s), the last (max_seq_length - len(s)) blocks are all
        0s.
        """
        if seq is None:
            seq = self.seq

        z = np.zeros((self.nalphabet * self.max_seq_length), dtype=np.float32)

        if len(seq) > 0:
            if hasattr(
                seq[0], "device"
            ):  # if it has a device at all, it will be cuda (CPU numpy array has no dev
                seq = [subseq.cpu().detach().numpy() for subseq in seq]

            z[(np.arange(len(seq)) * self.nalphabet + seq)] = 1
        return z

    def obs2seq(self, obs):
        """
        Transforms the one-hot encoding version of a sequence (state) given as argument
        into a a sequence of letter indices.

        Example:
          - Sequence: AATGC
          - obs: [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]
                 |     A    |      A    |      T    |      G    |      C    |
          - seq: [0, 0, 1, 3, 2]
                  A, A, T, G, C
        """
        obs_mat = np.reshape(obs, (self.max_seq_length, self.nalphabet))
        seq = np.where(obs_mat)[1]
        return seq

    def seq2letters(self, seq, alphabet={0: "A", 1: "T", 2: "C", 3: "G"}):
        """
        Transforms a sequence given as a list of indices into a sequence of letters
        according to an alphabet.
        """
        return [alphabet[el] for el in seq]

    def letters2seq(self, letters, alphabet={0: "A", 1: "T", 2: "C", 3: "G"}):
        """
        Transforms a sequence given as a list of indices into a sequence of letters
        according to an alphabet.
        """
        alphabet = {v: k for k, v in alphabet.items()}
        return [alphabet[el] for el in letters]

    def reset(self, env_id=None):
        """
        Resets the environment
        """
        self.seq = []
        self.n_actions = 0
        self.done = False
        self.id = env_id
        return self

    def parent_transitions(self, seq, action):
        # TODO: valid parents must satisfy max_seq_length constraint!!!
        """
        Determines all parents and actions that lead to sequence (state) seq

        Args
        ----
        seq : list
            Representation of a sequence (state), as a list of length max_seq_length
            where each element is the index of a letter in the alphabet, from 0 to
            (nalphabet - 1).

        action : int
            Last action performed

        Returns
        -------
        parents : list
            List of parents as seq2obs(seq)

        actions : list
            List of actions that lead to seq for each parent in parents
        """
        if action == self.eos:
            return [self.seq2obs(seq)], [action]
        else:
            parents = []
            actions = []
            for idx, a in enumerate(self.action_space):
                if seq[-len(a) :] == list(a):
                    parents.append(self.seq2obs(seq[: -len(a)]))
                    actions.append(idx)
        return parents, actions

    def get_trajectories(self, traj_list, actions):
        """
        Determines all trajectories to sequence seq

        Args
        ----
        traj_list : list
            List of trajectories (lists)

        actions : list
            List of actions within each trajectory

        Returns
        -------
        traj_list : list
            List of trajectories (lists)

        actions : list
            List of actions within each trajectory
        """
        current_traj = traj_list[-1].copy()
        current_traj_actions = actions[-1].copy()
        parents, parents_actions = self.parent_transitions(list(current_traj[-1]), -1)
        parents = [self.obs2seq(el).tolist() for el in parents]
        if parents == []:
            return traj_list, actions
        for idx, (p, a) in enumerate(zip(parents, parents_actions)):
            if idx > 0:
                traj_list.append(current_traj)
                actions.append(current_traj_actions)
            traj_list[-1] += [p]
            actions[-1] += [a]
            traj_list, actions = self.get_trajectories(traj_list, actions)
        return traj_list, actions

    def step(self, action):
        """
        Define step given action and state.

        See: step_daug()
        See: step_chain()
        """
        if self.allow_backward:
            return self.step_chain(action)
        return self.step_dag(action)

    def step_dag(self, action):
        """
        Executes step given an action

        If action is smaller than eos (no stop), add action to next
        position.

        See: step_daug()
        See: step_chain()

        Args
        ----
        a : int
            Index of action in the action space. a == eos indicates "stop action"

        Returns
        -------
        self.seq : list
            The sequence after executing the action

        valid : bool
            False, if the action is not allowed for the current state, e.g. stop at the
            root state
        """
        if len(self.seq) == self.max_seq_length:
            self.done = True
            self.n_actions += 1
            return self.seq, True
        if action < self.eos:
            seq_next = self.seq + list(self.action_space[action])
            if len(seq_next) > self.max_seq_length:
                valid = False
            else:
                self.seq = seq_next
                valid = True
                self.n_actions += 1
        else:
            if len(self.seq) < self.min_seq_length:
                valid = False
            else:
                self.done = True
                valid = True
                self.n_actions += 1

        return self.seq, valid

    def true_density(self, max_states=1e6):
        """
        Computes the reward density (reward / sum(rewards)) of the whole space, if the
        dimensionality is smaller than specified in the arguments.

        Returns
        -------
        Tuple:
          - normalized reward for each state
          - states
          - (un-normalized) reward)
        """
        if self._true_density is not None:
            return self._true_density
        if self.nalphabet ** self.max_seq_length > max_states:
            return (None, None, None)
        seq_all = np.int32(
            list(
                itertools.product(*[list(range(self.nalphabet))] * self.max_seq_length)
            )
        )
        traj_rewards, seq_end = zip(
            *[
                (self.proxy(seq), seq)
                for seq in seq_all
                if len(self.parent_transitions(seq, 0)[0]) > 0 or sum(seq) == 0
            ]
        )
        traj_rewards = np.array(traj_rewards)
        self._true_density = (
            traj_rewards / traj_rewards.sum(),
            list(map(tuple, seq_end)),
            traj_rewards,
        )
        return self._true_density
