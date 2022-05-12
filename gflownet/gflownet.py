"""
GFlowNet
TODO:
    - Seeds
"""
import copy
import time
from argparse import ArgumentParser
from collections import defaultdict
from itertools import count
from pathlib import Path

from comet_ml import Experiment
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.distributions.categorical import Categorical
from tqdm import tqdm

from aptamers import AptamerSeq
from oracle import numbers2letters, Oracle
from utils import get_config, namespace2dict, numpy2python, add_bool_arg

# Float and Long tensors
_dev = [torch.device("cpu")]
tf = lambda x: torch.FloatTensor(x).to(_dev[0])
tl = lambda x: torch.LongTensor(x).to(_dev[0])


def add_args(parser):
    """
    Adds command-line arguments to parser

    Returns:
        argparse.Namespace: the parsed arguments
    """
    args2config = {}
    # YAML config
    parser.add_argument(
        "-y",
        "--yaml_config",
        default=None,
        type=str,
        help="YAML configuration file",
    )
    args2config.update({"yaml_config": ["yaml_config"]})
    # General
    parser.add_argument("--workdir", default=None, type=str)
    args2config.update({"workdir": ["workdir"]})
    parser.add_argument(
        "--overwrite_workdir", action="store_true", default=False
    )
    args2config.update({"overwrite_workdir": ["overwrite_workdir"]})
    parser.add_argument("--device", default="cpu", type=str)
    args2config.update({"device": ["gflownet", "device"]})
    parser.add_argument("--progress", action="store_true")
    args2config.update({"progress": ["gflownet", "progress"]})
    parser.add_argument("--debug", action="store_true")
    args2config.update({"debug": ["debug"]})
    parser.add_argument("--model_ckpt", default=None, type=str)
    args2config.update({"model_ckpt": ["gflownet", "model_ckpt"]})
    parser.add_argument("--reload_ckpt", action="store_true")
    args2config.update({"reload_ckpt": ["gflownet", "reload_ckpt"]})
    parser.add_argument("--ckpt_period", default=None, type=int)
    args2config.update({"ckpt_period": ["gflownet", "ckpt_period"]})
    parser.add_argument(
        "--rng_seed",
        type=int,
        default=0,
        help="Seed for random number generator",
    )
    args2config.update({"rng_seed": ["seeds", "gflownet"]})
    # Oracle
    parser.add_argument(
        "--oracle_seed",
        type=int,
        default=0,
        help="Seed for oracle",
    )
    args2config.update({"oracle_seed": ["oracle", "seed"]})
    parser = add_bool_arg(parser, "nupack_energy_reweighting", default=False)
    args2config.update(
        {"nupack_energy_reweighting": ["oracle", "nupack_energy_reweighting"]}
    )
    parser.add_argument(
        "--nupack_target_motif",
        type=str,
        default=".....(((((.......))))).....",
        help="if using 'nupack motif' oracle, return value is the binary distance to this fold, must be <= max sequence length",
    )
    args2config.update(
        {"nupack_target_motif": ["oracle", "nupack_target_motif"]}
    )
    # Training hyperparameters
    parser.add_argument(
        "--loss",
        default="flowmatch",
        type=str,
        help="flowmatch | trajectorybalance/tb",
    )
    args2config.update({"loss": ["gflownet", "loss"]})
    parser.add_argument(
        "--lr_z_mult",
        default=10,
        type=int,
        help="Multiplicative factor of the Z learning rate",
    )
    args2config.update({"lr_z_mult": ["gflownet", "lr_z_mult"]})
    parser.add_argument(
        "--early_stopping",
        default=0.01,
        help="Threshold loss for early stopping",
        type=float,
    )
    args2config.update({"early_stopping": ["gflownet", "early_stopping"]})
    parser.add_argument(
        "--ema_alpha",
        default=0.5,
        help="alpha coefficient for exponential moving average",
        type=float,
    )
    args2config.update({"ema_alpha": ["gflownet", "ema_alpha"]})
    parser.add_argument(
        "--learning_rate", default=1e-4, help="Learning rate", type=float
    )
    args2config.update({"learning_rate": ["gflownet", "learning_rate"]})
    parser.add_argument("--opt", default="adam", type=str)
    args2config.update({"opt": ["gflownet", "opt"]})
    parser.add_argument("--adam_beta1", default=0.9, type=float)
    args2config.update({"adam_beta1": ["gflownet", "adam_beta1"]})
    parser.add_argument("--adam_beta2", default=0.999, type=float)
    args2config.update({"adam_beta2": ["gflownet", "adam_beta2"]})
    parser.add_argument(
        "--reward_beta_init",
        default=1,
        type=float,
        help="Initial beta for exponential reward scaling",
    )
    args2config.update({"reward_beta_init": ["gflownet", "reward_beta_init"]})
    parser.add_argument(
        "--reward_max",
        default=1e6,
        type=float,
        help="Max reward to prevent numerical issues",
    )
    args2config.update({"reward_max": ["gflownet", "reward_max"]})
    parser.add_argument(
        "--reward_beta_mult",
        default=1.25,
        type=float,
        help="Multiplier for rescaling beta during training",
    )
    args2config.update({"reward_beta_mult": ["gflownet", "reward_beta_mult"]})
    parser.add_argument(
        "--reward_beta_period",
        default=-1,
        type=float,
        help="Period (number of iterations) for beta rescaling",
    )
    args2config.update(
        {"reward_beta_period": ["gflownet", "reward_beta_period"]}
    )
    parser.add_argument(
        "--reward_norm",
        default=1.0,
        type=float,
        help="Factor for the reward normalization",
    )
    args2config.update({"reward_norm": ["gflownet", "reward_norm"]})
    parser.add_argument("--momentum", default=0.9, type=float)
    args2config.update({"momentum": ["gflownet", "momentum"]})
    parser.add_argument(
        "--mbsize", default=16, help="Minibatch size", type=int
    )
    args2config.update({"mbsize": ["gflownet", "mbsize"]})
    parser.add_argument("--train_to_sample_ratio", default=1, type=float)
    args2config.update(
        {"train_to_sample_ratio": ["gflownet", "train_to_sample_ratio"]}
    )
    parser.add_argument("--n_hid", default=256, type=int)
    args2config.update({"n_hid": ["gflownet", "n_hid"]})
    parser.add_argument("--n_layers", default=2, type=int)
    args2config.update({"n_layers": ["gflownet", "n_layers"]})
    parser.add_argument("--n_train_steps", default=20000, type=int)
    args2config.update({"n_train_steps": ["gflownet", "n_iter"]})
    parser.add_argument(
        "--num_empirical_loss",
        default=200000,
        type=int,
        help="Number of samples used to compute the empirical distribution loss",
    )
    args2config.update(
        {"num_empirical_loss": ["gflownet", "num_empirical_loss"]}
    )
    parser.add_argument("--clip_grad_norm", default=0.0, type=float)
    args2config.update({"clip_grad_norm": ["gflownet", "clip_grad_norm"]})
    parser.add_argument("--random_action_prob", default=0.0, type=float)
    args2config.update(
        {"random_action_prob": ["gflownet", "random_action_prob"]}
    )
    parser.add_argument("--pct_batch_empirical", default=0.0, type=float)
    args2config.update(
        {"pct_batch_empirical": ["gflownet", "pct_batch_empirical"]}
    )
    # Environment
    parser.add_argument("--func", default="arbitrary_i")
    args2config.update({"func": ["gflownet", "func"]})
    parser.add_argument(
        "--max_seq_length",
        default=40,
        help="Maximum number of episodes; maximum sequence length",
        type=int,
    )
    args2config.update({"max_seq_length": ["gflownet", "max_seq_length"]})
    parser.add_argument(
        "--min_seq_length",
        default=1,
        help="Minimum sequence length",
        type=int,
    )
    args2config.update({"min_seq_length": ["gflownet", "min_seq_length"]})
    parser.add_argument("--nalphabet", default=4, type=int)
    args2config.update({"nalphabet": ["gflownet", "nalphabet"]})
    parser.add_argument("--min_word_len", default=1, type=int)
    args2config.update({"min_word_len": ["gflownet", "min_word_len"]})
    parser.add_argument("--max_word_len", default=1, type=int)
    args2config.update({"max_word_len": ["gflownet", "max_word_len"]})
    args2config.update({"learning_rate": ["gflownet", "learning_rate"]})
    # Sampling
    parser.add_argument("--bootstrap_tau", default=0.0, type=float)
    args2config.update({"bootstrap_tau": ["gflownet", "bootstrap_tau"]})
    parser.add_argument("--batch_reward", action="store_true")
    args2config.update({"batch_reward": ["gflownet", "batch_reward"]})
    # Train set
    parser.add_argument("--train_set_path", default=None, type=str)
    args2config.update({"train_set_path": ["gflownet", "train", "path"]})
    parser.add_argument("--ntrain", default=10000, type=int)
    args2config.update({"ntrain": ["gflownet", "train", "n"]})
    parser.add_argument("--train_set_seed", default=167, type=int)
    args2config.update({"train_set_seed": ["gflownet", "train", "seed"]})
    parser.add_argument("--train_output", default=None, type=str)
    args2config.update({"train_output": ["gflownet", "train", "output"]})
    # Test set
    parser.add_argument("--test_set_path", default=None, type=str)
    args2config.update({"test_set_path": ["gflownet", "test", "path"]})
    parser.add_argument("--test_set_base", default=None, type=str)
    args2config.update({"test_set_base": ["gflownet", "test", "base"]})
    parser.add_argument("--test_set_seed", default=167, type=int)
    args2config.update({"test_set_seed": ["gflownet", "test", "seed"]})
    parser.add_argument("--ntest", default=10000, type=int)
    args2config.update({"ntest": ["gflownet", "test", "n"]})
    parser.add_argument("--min_length", default=1, type=int)
    args2config.update({"min_length": ["gflownet", "test", "min_length"]})
    parser.add_argument("--test_output", default=None, type=str)
    args2config.update({"test_output": ["gflownet", "test", "output"]})
    parser.add_argument("--test_period", default=500, type=int)
    args2config.update({"test_period": ["gflownet", "test", "period"]})
    # Oracle metrics
    parser.add_argument("--oracle_period", default=500, type=int)
    args2config.update({"oracle_period": ["gflownet", "oracle", "period"]})
    parser.add_argument("--oracle_nsamples", default=500, type=int)
    args2config.update({"oracle_nsamples": ["gflownet", "oracle", "nsamples"]})
    parser.add_argument(
        "--oracle_k",
        default=[1, 10, 100],
        nargs="*",
        type=int,
        help="List of K, for Top-K",
    )
    args2config.update({"oracle_k": ["gflownet", "oracle", "k"]})
    # Comet
    parser.add_argument("--comet_project", default=None, type=str)
    args2config.update({"comet_project": ["gflownet", "comet", "project"]})
    parser.add_argument(
        "-t", "--tags", nargs="*", help="Comet.ml tags", default=[], type=str
    )
    args2config.update({"tags": ["gflownet", "comet", "tags"]})
    parser.add_argument("--no_comet", action="store_true")
    args2config.update({"no_comet": ["gflownet", "comet", "skip"]})
    parser.add_argument("--no_log_times", action="store_true")
    args2config.update({"no_log_times": ["gflownet", "no_log_times"]})
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10000,
        help="Sequences to sample",
    )
    args2config.update({"n_samples": ["gflownet", "n_samples"]})

    return parser, args2config


def process_config(config):
    if (
        not config.gflownet.test.score
        or "nupack" in config.gflownet.test.score
    ):
        config.gflownet.test.score = config.gflownet.func.replace(
            "nupack ", ""
        )
    return config


def set_device(dev):
    _dev[0] = dev


class GFlowNetAgent:
    def __init__(
        self, args, comet=None, proxy=None, al_iter=-1, data_path=None
    ):
        # Misc
        self.rng = np.random.default_rng(args.seeds.gflownet)
        self.debug = args.debug
        self.device_torch = torch.device(args.gflownet.device)
        self.device = self.device_torch
        set_device(self.device_torch)
        if args.gflownet.loss in ["flowmatch"]:
            self.loss = "flowmatch"
            self.Z = None
        elif args.gflownet.loss in ["trajectorybalance", "tb"]:
            self.loss = "trajectorybalance"
            self.Z = nn.Parameter(torch.ones(64) * 150.0 / 64)
        else:
            print("Unkown loss. Using flowmatch as default")
            self.loss == "flowmatch"
            self.Z = None
        self.loss_eps = torch.tensor(float(1e-5)).to(self.device)
        self.lightweight = True
        self.tau = args.gflownet.bootstrap_tau
        self.ema_alpha = args.gflownet.ema_alpha
        self.early_stopping = args.gflownet.early_stopping
        self.reward_norm = reward_norm
        self.reward_beta = args.gflownet.reward_beta_init
        self.reward_beta_mult = args.gflownet.reward_beta_mult
        self.reward_beta_period = args.gflownet.reward_beta_period
        if self.reward_beta_period in [None, -1]:
            self.reward_beta_period = np.inf
        self.reward_max = args.gflownet.reward_max
        if al_iter >= 0:
            self.al_iter = "_iter{}".format(al_iter)
        else:
            self.al_iter = ""
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        # Oracle
        self.oracle = Oracle(
            seed=args.oracle.seed,
            seq_len=args.gflownet.max_seq_length,
            dict_size=args.gflownet.nalphabet,
            min_len=args.gflownet.min_seq_length,
            max_len=args.gflownet.max_seq_length,
            oracle=args.gflownet.func,
            energy_weight=args.oracle.nupack_energy_reweighting,
            nupack_target_motif=args.oracle.nupack_target_motif,
        )
        # Comet
        if args.gflownet.comet.project and not args.gflownet.comet.skip:
            self.comet = Experiment(
                project_name=args.gflownet.comet.project,
                display_summary_level=0,
            )
            if args.gflownet.comet.tags:
                if isinstance(args.gflownet.comet.tags, list):
                    self.comet.add_tags(args.gflownet.comet.tags)
                else:
                    self.comet.add_tag(args.gflownet.comet.tags)
            self.comet.log_parameters(vars(args))
            if "workdir" in args and Path(args.workdir).exists():
                with open(Path(args.workdir) / "comet.url", "w") as f:
                    f.write(self.comet.url + "\n")
        else:
            if isinstance(comet, Experiment):
                self.comet = comet
            else:
                self.comet = None
        self.no_log_times = args.gflownet.no_log_times
        # Train set or empirical data from active learning
        if data_path:
            self.data_path = Path(data_path)
            self.al_init_length = args.dataset.init_length
            self.al_queries_per_iter = args.al.queries_per_iter
            self.pct_test = args.gflownet.test.pct_test
            self.data_seed = args.seeds.dataset
            if self.data_path.suffix == ".npy":
                self.df_data = np2df(
                    self.data_path,
                    args.gflownet.test.score,
                    self.al_init_length,
                    self.al_queries_per_iter,
                    self.pct_test,
                    self.data_seed,
                )
                self.df_train = self.df_data.loc[self.df_data.train]
            else:
                self.df_data = None
                self.df_train = None
        elif args.gflownet.train.path:
            self.df_data = None
            self.df_train = pd.read_csv(args.gflownet.train.path, index_col=0)
        else:
            self.df_data = None
            self.df_train = make_train_set(
                self.oracle,
                args.gflownet.train.n,
                args.gflownet.train.seed,
                args.gflownet.train.output,
            )
        if self.df_train is not None:
            min_scores_tr = self.df_train["scores"].min()
            max_scores_tr = self.df_train["scores"].max()
            mean_scores_tr = self.df_train["scores"].mean()
            std_scores_tr = self.df_train["scores"].std()
            scores_tr_norm = (
                self.df_train["scores"].values - mean_scores_tr
            ) / std_scores_tr
            max_norm_scores_tr = np.max(scores_tr_norm)
            self.stats_scores_tr = [
                min_scores_tr,
                max_scores_tr,
                mean_scores_tr,
                std_scores_tr,
                max_norm_scores_tr,
            ]
        else:
            self.stats_scores_tr = None
        # Test set
        self.test_period = args.gflownet.test.period
        if self.test_period in [None, -1]:
            self.test_period = np.inf
            self.df_test = None
        else:
            self.test_score = args.gflownet.test.score
            if self.df_data is not None:
                self.df_test = self.df_data.loc[self.df_data.test]
            elif args.gflownet.test.path:
                self.df_test = pd.read_csv(
                    args.gflownet.test.path, index_col=0
                )
            else:
                self.df_test, test_set_times = make_approx_uniform_test_set(
                    path_base_dataset=args.gflownet.test.base,
                    score=self.test_score,
                    ntest=args.gflownet.test.n,
                    min_length=args.gflownet.test.min_length,
                    max_length=args.gflownet.max_seq_length,
                    seed=args.gflownet.test.seed,
                    output_csv=args.gflownet.test.output,
                )
        if self.df_test is not None:
            print("\nTest data")
            print(f"\tAverage score: {self.df_test[self.test_score].mean()}")
            print(f"\tStd score: {self.df_test[self.test_score].std()}")
            print(f"\tMin score: {self.df_test[self.test_score].min()}")
            print(f"\tMax score: {self.df_test[self.test_score].max()}")
        # Environment
        self.env = AptamerSeq(
            args.gflownet.max_seq_length,
            args.gflownet.min_seq_length,
            args.gflownet.nalphabet,
            args.gflownet.min_word_len,
            args.gflownet.max_word_len,
            proxy=proxy,
            allow_backward=False,
            debug=self.debug,
            reward_beta=self.reward_beta,
            oracle_func=self.oracle.score,
            stats_scores=self.stats_scores_tr,
            reward_norm=self.reward_norm,
        )
        self.envs = [
            AptamerSeq(
                args.gflownet.max_seq_length,
                args.gflownet.min_seq_length,
                args.gflownet.nalphabet,
                args.gflownet.min_word_len,
                args.gflownet.max_word_len,
                proxy=proxy,
                allow_backward=False,
                debug=self.debug,
                reward_beta=self.reward_beta,
                oracle_func=self.oracle.score,
                stats_scores=self.stats_scores_tr,
                reward_norm=self.reward_norm,
            )
            for _ in range(args.gflownet.mbsize)
        ]
        self.batch_reward = args.gflownet.batch_reward
        # Model
        self.model = make_mlp(
            [args.gflownet.max_seq_length * args.gflownet.nalphabet]
            + [args.gflownet.n_hid] * args.gflownet.n_layers
            + [len(self.env.action_space) + 1]
        )
        self.reload_ckpt = args.gflownet.reload_ckpt
        if args.gflownet.model_ckpt:
            if "workdir" in args and Path(args.workdir).exists():
                if (Path(args.workdir) / "ckpts").exists():
                    self.model_path = (
                        Path(args.workdir) / "ckpts" / args.gflownet.model_ckpt
                    )
                else:
                    self.model_path = (
                        Path(args.workdir) / args.gflownet.model_ckpt
                    )
            else:
                self.model_path = args.gflownet.model_ckpt
            if self.model_path.exists() and self.reload_ckpt:
                self.model.load_state_dict(torch.load(self.model_path))
                print("Reloaded GFN Model Checkpoint")
        else:
            self.model_path = None
        self.ckpt_period = args.gflownet.ckpt_period
        if self.ckpt_period in [None, -1]:
            self.ckpt_period = np.inf
        self.model.to(self.device_torch)
        self.target = copy.deepcopy(self.model)
        # Training
        self.opt = make_opt(self.parameters(), self.Z, args)
        self.n_train_steps = args.gflownet.n_iter
        self.mbsize = args.gflownet.mbsize
        self.progress = args.gflownet.progress
        self.clip_grad_norm = args.gflownet.clip_grad_norm
        self.num_empirical_loss = args.gflownet.num_empirical_loss
        self.ttsr = max(int(args.gflownet.train_to_sample_ratio), 1)
        self.sttr = max(int(1 / args.gflownet.train_to_sample_ratio), 1)
        self.random_action_prob = args.gflownet.random_action_prob
        self.pct_batch_empirical = args.gflownet.pct_batch_empirical
        # Oracle metrics
        self.oracle_period = args.gflownet.oracle.period
        self.oracle_nsamples = args.gflownet.oracle.nsamples
        self.oracle_k = args.gflownet.oracle.k

    def parameters(self):
        return self.model.parameters()

    def sample_batch(self, mask_eos=True):
        """
        Builds a mini-batch of data

        Each item in the batch is a list of 5 elements:
            - all parents of the state
            - actions that lead to the state from each parent
            - reward of the state
            - the state, as seq2obs(seq)
            - done
            - trajectory id: identifies each trajectory
            - seq id: identifies each sequence within a trajectory

        Args
        ----
        mbsize : int
            Mini-batch size
        """
        times = {
            "all": 0.0,
            "actions_model": 0.0,
            "actions_envs": 0.0,
            "rewards": 0.0,
        }
        t0_all = time.time()
        batch = []
        envs = [env.reset(idx) for idx, env in enumerate(self.envs)]
        # Sequences from empirical distribution
        n_empirical = int(self.pct_batch_empirical * len(envs))
        for env in envs[:n_empirical]:
            env.done = True
            seq_letters = self.rng.permutation(self.df_train.letters.values)[0]
            seq = env.letters2seq(seq_letters)
            done = True
            action = env.eos
            while len(seq) > 0:
                parents, parents_a = env.parent_transitions(seq, action)
                batch.append(
                    [
                        tf(parents),
                        tl(parents_a),
                        seq,
                        tf([env.seq2obs(seq)]),
                        done,
                        tl([env.id]),
                        tl([env.n_actions]),
                    ]
                )
                seq = env.obs2seq(self.rng.permutation(parents)[0])
                done = False
                action = -1
        envs = [env for env in envs if not env.done]
        # Rest of batch
        while envs:
            seqs = [env.seq2obs() for env in envs]
            mask = [len(env.seq) < env.min_seq_length for env in envs]
            random_action = self.rng.uniform()
            if random_action > self.random_action_prob:
                with torch.no_grad():
                    t0_a_model = time.time()
                    action_probs = self.model(tf(seqs))
                    if mask_eos:
                        action_probs[mask, -1] = -1000
                    t1_a_model = time.time()
                    times["actions_model"] += t1_a_model - t0_a_model
                    if all(torch.isfinite(action_probs).flatten()):
                        actions = Categorical(logits=action_probs).sample()
                    else:
                        random_action = -1
                        if self.debug:
                            print("Action could not be sampled from model!")
            if random_action < self.random_action_prob:
                high = (len(self.env.action_space) + 1) * np.ones(
                    len(envs), dtype=int
                )
                if mask_eos:
                    high[mask] -= 1
                actions = self.rng.integers(low=0, high=high, size=len(envs))
            t0_a_envs = time.time()
            assert len(envs) == actions.shape[0]
            for env, action in zip(envs, actions):
                if len(env.seq) == env.max_seq_length:
                    action = env.eos
                seq, valid = env.step(action)
                if valid:
                    parents, parents_a = env.parent_transitions(seq, action)
                    batch.append(
                        [
                            tf(parents),
                            tl(parents_a),
                            seq,
                            tf([env.seq2obs()]),
                            env.done,
                            tl([env.id]),
                            tl([env.n_actions]),
                        ]
                    )
            envs = [env for env in envs if not env.done]
            t1_a_envs = time.time()
            times["actions_envs"] += t1_a_envs - t0_a_envs
        parents, parents_a, seqs, obs, done, traj_id, seq_id = zip(*batch)
        t0_rewards = time.time()
        rewards = env.reward_batch(seqs, done)
        t1_rewards = time.time()
        times["rewards"] += t1_rewards - t0_rewards
        rewards = [tf([r]) for r in rewards]
        done = [tl([d]) for d in done]
        batch = list(
            zip(parents, parents_a, rewards, obs, done, traj_id, seq_id)
        )
        t1_all = time.time()
        times["all"] += t1_all - t0_all
        return batch, times

    def flowmatch_loss(self, it, batch):
        """
        Computes the loss of a batch

        Args
        ----
        it : int
            Iteration

        batch : ndarray
            A batch of data: every row is a state (list), corresponding to all states
            visited in each sequence in the batch.

        Returns
        -------
        loss : float
            Loss, as per Equation 12 of https://arxiv.org/abs/2106.04399v1

        term_loss : float
            Loss of the terminal nodes only

        flow_loss : float
            Loss of the intermediate nodes only
        """
        loginf = tf([1000])
        batch_idxs = tl(
            sum(
                [
                    [i] * len(parents)
                    for i, (parents, _, _, _, _, _, _) in enumerate(batch)
                ],
                [],
            )
        )
        parents, actions, r, sp, done, _, _ = map(torch.cat, zip(*batch))

        # Sanity check if negative rewards
        if self.debug and torch.any(r < 0):
            neg_r_idx = torch.where(r < 0)[0].tolist()
            for idx in neg_r_idx:
                obs = sp[idx].tolist()
                seq = list(self.env.obs2seq(obs))
                seq_oracle = self.env.seq2oracle([seq])
                output_proxy = self.env.proxy(seq_oracle)
                reward = self.env.proxy2reward(output_proxy)
                print(idx, output_proxy, reward)
                import ipdb

                ipdb.set_trace()

        # Q(s,a)
        parents_Qsa = self.model(parents)[
            torch.arange(parents.shape[0]), actions
        ]

        # log(eps + exp(log(Q(s,a)))) : qsa
        in_flow = torch.log(
            tf(torch.zeros((sp.shape[0],))).index_add_(
                0, batch_idxs, torch.exp(parents_Qsa)
            )
        )
        # the following with work if autoregressive
        #         in_flow = torch.logaddexp(parents_Qsa[batch_idxs], torch.log(self.loss_eps))
        if self.tau > 0:
            with torch.no_grad():
                next_q = self.target(sp)
        else:
            next_q = self.model(sp)
        qsp = torch.logsumexp(next_q, 1)
        # qsp: qsp if not done; -loginf if done
        qsp = qsp * (1 - done) - loginf * done
        out_flow = torch.logaddexp(torch.log(r + self.loss_eps), qsp)
        loss = (in_flow - out_flow).pow(2).mean()

        with torch.no_grad():
            term_loss = ((in_flow - out_flow) * done).pow(2).sum() / (
                done.sum() + 1e-20
            )
            flow_loss = ((in_flow - out_flow) * (1 - done)).pow(2).sum() / (
                (1 - done).sum() + 1e-20
            )

        if self.tau > 0:
            for a, b in zip(self.model.parameters(), self.target.parameters()):
                b.data.mul_(1 - self.tau).add_(self.tau * a)

        return loss, term_loss, flow_loss

    def trajectorybalance_loss(self, it, batch):
        """
        Computes the trajectory balance loss of a batch

        Args
        ----
        it : int
            Iteration

        batch : ndarray
            A batch of data: every row is a state (list), corresponding to all states
            visited in each sequence in the batch.

        Returns
        -------
        loss : float

        term_loss : float
            Loss of the terminal nodes only

        flow_loss : float
            Loss of the intermediate nodes only
        """
        # Unpack batch
        parents, actions, rewards, _, done, traj_id, _ = map(
            torch.cat, zip(*batch)
        )
        # Log probs of each (s, a)
        logprobs = self.logsoftmax(self.model(parents))[
            torch.arange(parents.shape[0]), actions
        ]
        # Sum of log probs
        sumlogprobs = tf(
            torch.zeros(len(torch.unique(traj_id, sorted=True)))
        ).index_add_(0, traj_id, logprobs)
        # Sort rewards of done sequences by ascending traj id
        rewards = rewards[done.eq(1)][torch.argsort(traj_id[done.eq(1)])]
        # Trajectory balance loss
        loss = (
            (self.Z.sum() + sumlogprobs - torch.log((rewards))).pow(2).mean()
        )
        return loss, loss, loss

    def train(self):

        # Metrics
        all_losses = []
        all_visited = []
        empirical_distrib_losses = []
        loss_term_ema = None
        loss_flow_ema = None

        # Train loop
        for i in tqdm(
            range(self.n_train_steps + 1)
        ):  # , disable=not self.progress):
            t0_iter = time.time()
            data = []
            for j in range(self.sttr):
                batch, times = self.sample_batch()
                data += batch
            rewards = [d[2][0].item() for d in data if bool(d[4].item())]
            proxy_vals = self.env.reward2proxy(rewards)
            for j in range(self.ttsr):
                if self.loss == "flowmatch":
                    losses = self.flowmatch_loss(
                        i * self.ttsr + j, data
                    )  # returns (opt loss, *metrics)
                elif self.loss == "trajectorybalance":
                    losses = self.trajectorybalance_loss(
                        i * self.ttsr + j, data
                    )  # returns (opt loss, *metrics)
                else:
                    print("Unknown loss!")
                if (
                    not all([torch.isfinite(loss) for loss in losses])
                    or np.max(rewards) > self.reward_max
                ):
                    if self.debug:
                        print(
                            "Too large rewards: Skipping backward pass, increasing "
                            "reward temperature from -{:.4f} to -{:.4f} and cancelling "
                            "beta scheduling".format(
                                self.reward_beta,
                                self.reward_beta / self.reward_beta_mult,
                            )
                        )
                    self.reward_beta /= self.reward_beta_mult
                    self.reward_beta_period = np.inf
                    for env in [self.env] + self.envs:
                        env.reward_beta = self.reward_beta
                    if len(all_losses) > 0:
                        all_losses.append([loss for loss in all_losses[-1]])
                else:
                    losses[0].backward()
                    if self.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.parameters(), self.clip_grad_norm
                        )
                    self.opt.step()
                    self.opt.zero_grad()
                    all_losses.append([i.item() for i in losses])
            # Reward beta scaling
            if not i % self.reward_beta_period and i > 0:
                if self.debug:
                    print(
                        "\tDecreasing reward temperature from "
                        "-{:.4f} to -{:.4f}".format(
                            self.reward_beta,
                            self.reward_beta * self.reward_beta_mult,
                        )
                    )
                self.reward_beta *= self.reward_beta_mult
                for env in [self.env] + self.envs:
                    env.reward_beta = self.reward_beta
            # Log
            seqs_batch = [
                tuple(self.env.obs2seq(d[3][0].tolist()))
                for d in data
                if bool(d[4].item())
            ]
            idx_best = np.argmax(rewards)
            seq_best = "".join(self.env.seq2letters(seqs_batch[idx_best]))
            if self.lightweight:
                all_losses = all_losses[-100:]
                all_visited = seqs_batch

            else:
                all_visited.extend(seqs_batch)
            if self.comet:
                self.comet.log_text(
                    seq_best + " / proxy: {}".format(proxy_vals[idx_best]),
                    step=i,
                )
                self.comet.log_metrics(
                    dict(
                        zip(
                            [
                                "mean_reward{}".format(self.al_iter),
                                "max_reward{}".format(self.al_iter),
                                "mean_proxy{}".format(self.al_iter),
                                "min_proxy{}".format(self.al_iter),
                                "max_proxy{}".format(self.al_iter),
                                "mean_seq_length{}".format(self.al_iter),
                                "batch_size{}".format(self.al_iter),
                                "reward_beta{}".format(self.al_iter),
                            ],
                            [
                                np.mean(rewards),
                                np.max(rewards),
                                np.mean(proxy_vals),
                                np.min(proxy_vals),
                                np.max(proxy_vals),
                                np.mean([len(seq) for seq in seqs_batch]),
                                len(data),
                                self.reward_beta,
                            ],
                        )
                    ),
                    step=i,
                )
            # Test set metrics
            if not i % self.test_period and self.df_test is not None:
                data_logq = []
                times.update(
                    {
                        "test_traj": 0.0,
                        "test_logq": 0.0,
                    }
                )
                # TODO: this could be done just once and store it
                for seqstr, score in tqdm(
                    zip(self.df_test.letters, self.df_test[self.test_score])
                ):
                    t0_test_traj = time.time()
                    traj_list, actions = self.env.get_trajectories(
                        [[self.env.letters2seq(seqstr)]],
                        [[self.env.eos]],
                    )
                    t1_test_traj = time.time()
                    times["test_traj"] += t1_test_traj - t0_test_traj
                    t0_test_logq = time.time()
                    data_logq.append(
                        logq(traj_list, actions, self.model, self.env)
                    )
                    t1_test_logq = time.time()
                    times["test_logq"] += t1_test_logq - t0_test_logq
                corr = np.corrcoef(data_logq, self.df_test[self.test_score])
                if self.comet:
                    self.comet.log_metrics(
                        dict(
                            zip(
                                [
                                    "test_corr_logq_score{}".format(
                                        self.al_iter
                                    ),
                                    "test_mean_logq{}".format(self.al_iter),
                                ],
                                [
                                    corr[0, 1],
                                    np.mean(data_logq),
                                ],
                            )
                        ),
                        step=i,
                    )
            # Oracle metrics (for monitoring)
            if not i % self.oracle_period and self.debug:
                oracle_dict, oracle_times = self.sample(
                    self.oracle_nsamples,
                    self.env.max_seq_length,
                    self.env.min_seq_length,
                    self.env.nalphabet,
                    self.env.min_word_len,
                    self.env.max_word_len,
                    self.env.oracle,
                    get_uncertainties=False,
                )
                scores = oracle_dict["energies"]
                scores_sorted = np.sort(scores)
                dict_topk = {}
                for k in self.oracle_k:
                    mean_topk = np.mean(scores_sorted[:k])
                    dict_topk.update(
                        {
                            "oracle_mean_top{}{}".format(
                                k, self.al_iter
                            ): mean_topk
                        }
                    )
                    if self.comet:
                        self.comet.log_metrics(dict_topk)
            if not i % 100:
                if not self.lightweight:
                    empirical_distrib_losses.append(
                        compute_empirical_distribution_error(
                            self.env, all_visited[-self.num_empirical_loss :]
                        )
                    )
                else:
                    empirical_distrib_losses.append((None, None))
                if self.progress:
                    k1, kl = empirical_distrib_losses[-1]
                    if self.debug:
                        print("Empirical L1 distance", k1, "KL", kl)
                        if len(all_losses):
                            print(
                                *[
                                    f"{np.mean([i[j] for i in all_losses[-100:]]):.5f}"
                                    for j in range(len(all_losses[0]))
                                ]
                            )
                if self.comet:
                    self.comet.log_metrics(
                        dict(
                            zip(
                                [
                                    "loss{}".format(self.al_iter),
                                    "term_loss{}".format(self.al_iter),
                                    "flow_loss{}".format(self.al_iter),
                                ],
                                [loss.item() for loss in losses],
                            )
                        ),
                        step=i,
                    )
                    if not self.lightweight:
                        self.comet.log_metric(
                            "unique_states{}".format(self.al_iter),
                            np.unique(all_visited).shape[0],
                            step=i,
                        )
            # Save intermediate model
            if not i % self.ckpt_period and self.model_path:
                path = self.model_path.parent / Path(
                    self.model_path.stem
                    + "{}_iter{:06d}".format(self.al_iter, i)
                    + self.model_path.suffix
                )
                torch.save(self.model.state_dict(), path)
            # Moving average of the loss for early stopping
            if loss_term_ema and loss_flow_ema:
                loss_term_ema = (
                    self.ema_alpha * losses[1]
                    + (1.0 - self.ema_alpha) * loss_term_ema
                )
                loss_flow_ema = (
                    self.ema_alpha * losses[2]
                    + (1.0 - self.ema_alpha) * loss_flow_ema
                )
                if (
                    loss_term_ema < self.early_stopping
                    and loss_flow_ema < self.early_stopping
                ):
                    break
            else:
                loss_term_ema = losses[1]
                loss_flow_ema = losses[2]

            # Log times
            t1_iter = time.time()
            times.update({"iter": t1_iter - t0_iter})
            times = {
                "time_{}{}".format(k, self.al_iter): v
                for k, v in times.items()
            }
            if self.comet and not self.no_log_times:
                self.comet.log_metrics(times, step=i)
        # Save final model
        if self.model_path:
            path = self.model_path.parent / Path(
                self.model_path.stem + "_final" + self.model_path.suffix
            )
            torch.save(self.model.state_dict(), path)
            torch.save(self.model.state_dict(), self.model_path)

        # Close comet
        if self.comet and self.al_iter == -1:
            self.comet.end()

    def sample(
        self,
        n_samples,
        max_seq_length,
        min_seq_length,
        nalphabet,
        min_word_len,
        max_word_len,
        proxy,
        mask_eos=True,
        get_uncertainties=True,
        al_query_function=None,
    ):
        times = {
            "all": 0.0,
            "actions_model": 0.0,
            "actions_envs": 0.0,
            "proxy": 0.0,
            "sanitycheck": 0.0,
        }
        t0_all = time.time()
        batch = []
        envs = [
            AptamerSeq(
                max_seq_length=max_seq_length,
                min_seq_length=min_seq_length,
                nalphabet=nalphabet,
                min_word_len=min_word_len,
                max_word_len=max_word_len,
                proxy=proxy,
                stats_scores=self.stats_scores_tr,
            )
            for i in range(n_samples)
        ]
        envs = [env.reset() for env in envs]
        while envs:
            seqs = [env.seq2obs() for env in envs]
            mask = [len(env.seq) < env.min_seq_length for env in envs]
            with torch.no_grad():
                t0_a_model = time.time()
                action_probs = self.model(tf(seqs))
                if mask_eos:
                    action_probs[mask, -1] = -1000
                t1_a_model = time.time()
                times["actions_model"] += t1_a_model - t0_a_model
                if all(torch.isfinite(action_probs).flatten()):
                    actions = Categorical(logits=action_probs).sample()
                else:
                    actions = np.random.randint(
                        low=0,
                        high=action_probs.shape[1],
                        size=action_probs.shape[0],
                    )
                    if self.debug:
                        print("Action could not be sampled from model!")
            t0_a_envs = time.time()
            assert len(envs) == actions.shape[0]
            for env, action in zip(envs, actions):
                seq, valid = env.step(action)
                if valid and env.done:
                    batch.append(env.seq2oracle([seq])[0])
            envs = [env for env in envs if not env.done]
            t1_a_envs = time.time()
            times["actions_envs"] += t1_a_envs - t0_a_envs
        t0_proxy = time.time()
        batch = np.asarray(batch)

        if get_uncertainties:
            if self.al_query_function == "fancy_acquisition":
                scores, proxy_vals, uncertainties = env.proxy(
                    batch, "fancy_acquisition"
                )
            else:
                proxy_vals, uncertainties = env.proxy(batch, "Both")
                scores = proxy_vals
        else:
            proxy_vals = env.proxy(batch)
            uncertainties = None
            scores = proxy_vals
        t1_proxy = time.time()
        times["proxy"] += t1_proxy - t0_proxy
        samples = {
            "samples": batch.astype(np.int64),
            "scores": scores,
            "energies": proxy_vals,
            "uncertainties": uncertainties,
        }

        # Sanity-check: absolute zero pad
        t0_sanitycheck = time.time()
        zeros = np.where(batch == 0)
        row_unique, row_unique_idx = np.unique(zeros[0], return_index=True)
        for row, idx in zip(row_unique, row_unique_idx):
            if np.sum(batch[row, zeros[1][idx] :]):
                print(
                    f"Found sequence with positive values after last 0, row {row}"
                )
                import ipdb

                ipdb.set_trace()
        t1_sanitycheck = time.time()
        times["sanitycheck"] += t1_sanitycheck - t0_sanitycheck
        t1_all = time.time()
        times["all"] += t1_all - t0_all
        return samples, times


def sample(
    model,
    n_samples,
    max_seq_length,
    min_seq_length,
    nalphabet,
    min_word_len,
    max_word_len,
    func,
    mask_eos=True,
    stats_scores_tr=None,
):
    times = {
        "all": 0.0,
        "actions_model": 0.0,
        "actions_envs": 0.0,
        "proxy": 0.0,
        "sanitycheck": 0.0,
    }
    t0_all = time.time()
    batch = []
    envs = [
        AptamerSeq(
            max_seq_length=max_seq_length,
            min_seq_length=min_seq_length,
            nalphabet=nalphabet,
            min_word_len=min_word_len,
            max_word_len=max_word_len,
            func=func,
            stats_scores=stats_scores_tr,
        )
        for i in range(n_samples)
    ]
    envs = [env.reset() for env in envs]
    while envs:
        seqs = [env.seq2obs() for env in envs]
        mask = [len(env.seq) < env.min_seq_length for env in envs]
        with torch.no_grad():
            t0_a_model = time.time()
            action_probs = model(tf(seqs))
            if mask_eos:
                action_probs[mask, -1] = -1000
            t1_a_model = time.time()
            times["actions_model"] += t1_a_model - t0_a_model
            if all(torch.isfinite(action_probs).flatten()):
                actions = Categorical(logits=action_probs).sample()
            else:
                actions = np.random.randint(
                    low=0,
                    high=action_probs.shape[1],
                    size=action_probs.shape[0],
                )
        t0_a_envs = time.time()
        assert len(envs) == actions.shape[0]
        for env, action in zip(envs, actions):
            seq, valid = env.step(action)
            if valid and env.done:
                batch.append(env.seq2oracle([seq])[0])
        envs = [env for env in envs if not env.done]
        t1_a_envs = time.time()
        times["actions_envs"] += t1_a_envs - t0_a_envs
    t0_proxy = time.time()
    batch = np.asarray(batch)
    proxy_vals = env.proxy(batch)
    t1_proxy = time.time()
    times["proxy"] += t1_proxy - t0_proxy
    samples = {
        "samples": batch.astype(np.int64),
        "scores": proxy_vals,
    }
    # Sanity-check: absolute zero pad
    t0_sanitycheck = time.time()
    zeros = np.where(batch == 0)
    row_unique, row_unique_idx = np.unique(zeros[0], return_index=True)
    for row, idx in zip(row_unique, row_unique_idx):
        if np.sum(batch[row, zeros[1][idx] :]):
            print(
                f"Found sequence with positive values after last 0, row {row}"
            )
            import ipdb

            ipdb.set_trace()
    t1_sanitycheck = time.time()
    times["sanitycheck"] += t1_sanitycheck - t0_sanitycheck
    t1_all = time.time()
    times["all"] += t1_all - t0_all
    return samples, times


class RandomTrajAgent:
    def __init__(self, args, envs):
        self.mbsize = args.gflownet.mbsize  # mini-batch size
        self.envs = envs
        self.nact = args.ndim + 1
        self.model = None

    def parameters(self):
        return []

    def sample_batch(self, mbsize, all_visited):
        batch = []
        [i.reset()[0] for i in self.envs]  # reset envs
        done = [False] * mbsize
        while not all(done):
            acts = np.random.randint(0, self.nact, mbsize)  # actions (?)
            # step : list
            # - For each e in envs, if corresponding done is False
            #   - For each element i in env, and a in acts
            #     - i.step(a)
            step = [
                i.step(a)
                for i, a in zip(
                    [e for d, e in zip(done, self.envs) if not d], acts
                )
            ]
            c = count(0)
            m = {j: next(c) for j in range(mbsize) if not done[j]}
            done = [bool(d or step[m[i]][2]) for i, d in enumerate(done)]
            for (_, r, d, sp) in step:
                if d:
                    all_visited.append(tuple(sp))
        return []  # agent is stateful, no need to return minibatch data

    def flowmatch_loss(self, it, batch):
        return None


def make_mlp(layers_dim, act=nn.LeakyReLU(), tail=[]):
    """
    Defines an MLP with no top layer activation

    Args
    ----
    layers_dim : list
        Dimensionality of each layer

    act : Activation
        Activation function
    """
    return nn.Sequential(
        *(
            sum(
                [
                    [nn.Linear(idim, odim)]
                    + ([act] if n < len(layers_dim) - 2 else [])
                    for n, (idim, odim) in enumerate(
                        zip(layers_dim, layers_dim[1:])
                    )
                ],
                [],
            )
            + tail
        )
    )


def make_opt(params, Z, args):
    """
    Set up the optimizer
    """
    params = list(params)
    if not len(params):
        return None
    if args.gflownet.opt == "adam":
        opt = torch.optim.Adam(
            params,
            args.gflownet.learning_rate,
            betas=(args.gflownet.adam_beta1, args.gflownet.adam_beta2),
        )
        if Z is not None:
            opt.add_param_group(
                {
                    "params": Z,
                    "lr": args.gflownet.learning_rate
                    * args.gflownet.lr_z_mult,
                }
            )
    elif args.gflownet.opt == "msgd":
        opt = torch.optim.SGD(
            params,
            args.gflownet.learning_rate,
            momentum=args.gflownet.momentum,
        )
    return opt


def compute_empirical_distribution_error(env, visited):
    """
    Computes the empirical distribution errors, as the mean L1 error and the KL
    divergence between the true density of the space and the estimated density from all
    states visited.
    """
    td, end_states, true_r = env.true_density()
    if td is None:
        return None, None
    true_density = tf(td)
    if not len(visited):
        return 1, 100
    hist = defaultdict(int)
    for i in visited:
        hist[i] += 1
    Z = sum([hist[i] for i in end_states])
    estimated_density = tf([hist[i] / Z for i in end_states])
    k1 = abs(estimated_density - true_density).mean().item()
    # KL divergence
    kl = (
        (true_density * torch.log(estimated_density / true_density))
        .sum()
        .item()
    )
    return k1, kl


# TODO: improve approximation of uniform
def make_approx_uniform_test_set(
    path_base_dataset,
    score,
    ntest,
    min_length=0,
    max_length=np.inf,
    seed=167,
    output_csv=None,
):
    """
    Constructs an approximately uniformly distributed (on the score) set, by
    selecting samples from a larger base set.

    Args
    ----
    path_base_dataset : str
        Path to a CSV file containing the base data set.

    score : str
        Column in the CSV file containing the score.

    ntest : int
        Number of test samples.

    seed : int
        Random seed.

    dask : bool
        If True, use dask to efficiently read a large base file.

    output_csv: str
        Optional path to store the test set as CSV.
    """
    if path_base_dataset is None:
        return None
    times = {
        "all": 0.0,
        "indices": 0.0,
    }
    t0_all = time.time()
    if seed:
        np.random.seed(seed)
    df_base = pd.read_csv(path_base_dataset, index_col=0)
    df_base = df_base.loc[
        (df_base["letters"].map(len) >= min_length)
        & (df_base["letters"].map(len) <= max_length)
    ]
    scores_base = df_base[score].values
    min_base = scores_base.min()
    max_base = scores_base.max()
    distr_unif = np.random.uniform(low=min_base, high=max_base, size=ntest)
    # Get minimum distance samples without duplicates
    t0_indices = time.time()
    idx_samples = []
    for idx in tqdm(range(ntest)):
        dist = np.abs(scores_base - distr_unif[idx])
        idx_min = np.argmin(dist)
        if idx_min in idx_samples:
            idx_sort = np.argsort(dist)
            for idx_next in idx_sort:
                if idx_next not in idx_samples:
                    idx_samples.append(idx_next)
                    break
        else:
            idx_samples.append(idx_min)
    t1_indices = time.time()
    times["indices"] += t1_indices - t0_indices
    # Make test set
    df_test = df_base.iloc[idx_samples]
    if output_csv:
        df_test.to_csv(output_csv)
    t1_all = time.time()
    times["all"] += t1_all - t0_all
    return df_test, times


def make_train_set(
    oracle,
    ntrain,
    seed=168,
    output_csv=None,
):
    """
    Constructs a randomly sampled train set.

    Args
    ----
    ntest : int
        Number of test samples.

    seed : int
        Random seed.

    output_csv: str
        Optional path to store the test set as CSV.
    """
    samples_dict = oracle.initializeDataset(
        save=False, returnData=True, customSize=ntrain, custom_seed=seed
    )
    energies = samples_dict["energies"]
    samples_mat = samples_dict["samples"]
    seq_letters = oracle.numbers2letters(samples_mat)
    seq_ints = [
        "".join([str(el) for el in seq if el > 0]) for seq in samples_mat
    ]
    if isinstance(energies, dict):
        energies.update({"letters": seq_letters, "indices": seq_ints})
        df_train = pd.DataFrame(energies)
    else:
        df_train = pd.DataFrame(
            {"letters": seq_letters, "indices": seq_ints, "scores": energies}
        )
    if output_csv:
        df_train.to_csv(output_csv)
    return df_train


def np2df(
    test_path, score, al_init_length, al_queries_per_iter, pct_test, data_seed
):
    data_dict = np.load(test_path, allow_pickle=True).item()
    letters = numbers2letters(data_dict["samples"])
    df = pd.DataFrame(
        {
            "letters": letters,
            score: data_dict["energies"],
            "train": [False] * len(letters),
            "test": [False] * len(letters),
        }
    )
    # Split train and test section of init data set
    rng = np.random.default_rng(data_seed)
    indices = rng.permutation(al_init_length)
    n_tt = int(pct_test * len(indices))
    indices_tt = indices[:n_tt]
    indices_tr = indices[n_tt:]
    df.loc[indices_tt, "test"] = True
    df.loc[indices_tr, "train"] = True
    # Split train and test the section of each iteration to preserve splits
    idx = al_init_length
    iters_remaining = (len(df) - al_init_length) // al_queries_per_iter
    indices = rng.permutation(al_queries_per_iter)
    n_tt = int(pct_test * len(indices))
    for it in range(iters_remaining):
        indices_tt = indices[:n_tt] + idx
        indices_tr = indices[n_tt:] + idx
        df.loc[indices_tt, "test"] = True
        df.loc[indices_tr, "train"] = True
        idx += al_queries_per_iter
    return df


def logq(traj_list, actions_list, model, env):
    # TODO: this method is probably suboptimal, since it may repeat forward calls for
    # the same nodes.
    log_q = torch.tensor(1.0)
    for traj, actions in zip(traj_list, actions_list):
        traj = traj[::-1]
        actions = actions[::-1]
        traj_obs = np.asarray([env.seq2obs(seq) for seq in traj])
        with torch.no_grad():
            logits_traj = model(tf(traj_obs))
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        logprobs_traj = logsoftmax(logits_traj)
        log_q_traj = torch.tensor(0.0)
        for s, a, logprobs in zip(*[traj, actions, logprobs_traj]):
            log_q_traj = log_q_traj + logprobs[a]
        # Accumulate log prob of trajectory
        if torch.le(log_q, 0.0):
            log_q = torch.logaddexp(log_q, log_q_traj)
        else:
            log_q = log_q_traj
    return log_q.item()


def main(args):
    gflownet_agent = GFlowNetAgent(args)
    gflownet_agent.train()

    # sample from the oracle, not from a proxy model
    samples, times = gflownet_agent.sample(
        args.gflownet.n_samples,
        args.gflownet.max_seq_length,
        args.gflownet.min_seq_length,
        args.gflownet.nalphabet,
        args.gflownet.min_word_len,
        args.gflownet.max_word_len,
        gflownet_agent.env.oracle,
        mask_eos=True,
        get_uncertainties=False,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    _, override_args = parser.parse_known_args()
    parser, args2config = add_args(parser)
    args = parser.parse_args()
    config = get_config(args, override_args, args2config)
    config = process_config(config)
    print("Config file: " + config.yaml_config)
    print("Working dir: " + config.workdir)
    print(
        "Config:\n"
        + "\n".join(
            [f"    {k:20}: {v}" for k, v in vars(config.gflownet).items()]
        )
    )
    if "workdir" in config:
        if not Path(config.workdir).exists() or config.overwrite_workdir:
            Path(config.workdir).mkdir(parents=True, exist_ok=True)
            with open(config.workdir + "/config.yml", "w") as f:
                yaml.dump(
                    numpy2python(namespace2dict(config)),
                    f,
                    default_flow_style=False,
                )
            torch.set_num_threads(1)
            main(config)
        else:
            print(f"workdir {config.workdir} already exists! - Ending run...")
    else:
        print(f"workdir not defined - Ending run...")
