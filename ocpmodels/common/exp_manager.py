from orion.client import get_experiment
from pathlib import Path
from collections import defaultdict, Counter
import wandb
from textwrap import dedent
from minydra import resolved_args
import os
import sys

rundir = Path(os.environ["SCRATCH"]) / "ocp" / "runs"


class Manager:
    def __init__(
        self,
        orion_db_path="",
        name="",
        wandb_path="mila-ocp/ocp-qm",
    ):
        self.api = wandb.Api()
        self.wandb_path = wandb_path
        self.wandb_runs = [
            r
            for r in self.api.runs(wandb_path)
            if "orion_hash_params" in r.config
            and name in r.config.get("orion_exp_config_path", "")
        ]
        self.name = name
        self.trial_hparams_to_rundirs = defaultdict(list)
        self.exp = get_experiment(
            name=name,
            storage={
                "database": {
                    "host": str(orion_db_path),
                    "type": "pickleddb",
                }
            },
        )
        self.trials = self.exp.fetch_trials()
        self.budgets = self.exp.algorithms.algorithm.budgets
        self.total_budgets = sum(
            b.n_trials for bracket in self.budgets for b in bracket
        )
        self.id_to_trial = {t.id: t for t in self.trials}
        self.id_to_wandb_runs = {
            t.id: sorted(
                [
                    r
                    for r in self.wandb_runs
                    if r.config["orion_hash_params"] == t.hash_params
                ],
                key=lambda r: r.config["job_id"],
            )
            for t in self.trials
        }
        self.hash_to_trials = defaultdict(list)
        for t in self.trials:
            self.hash_to_trials[t.hash_params].append(t)
        self.discover_run_dirs()
        self.job_ids = sorted(
            [p.name for runs in self.trial_hparams_to_rundirs.values() for p in runs]
        )
        print("\n")
        self.print_status()

    def print_status(self):
        print("{:32} : {:4} ".format("Trials in experiment", len(self.trials)))
        print("{:32} : {:4}".format("Total expected trials", self.total_budgets))
        print(
            "{:32} : {:4} ".format(
                "Trials status",
                " ".join(
                    [
                        f"{k}->{v}"
                        for k, v in Counter([t.status for t in self.trials]).items()
                    ]
                ),
            )
        )
        print(
            "{:32} : {}".format(
                "Trial level(=rung) distribution",
                " ".join(
                    [
                        f"{k}->{v}"
                        for k, v in Counter(
                            map(len, self.hash_to_trials.values())
                        ).items()
                    ]
                ),
            )
        )
        print(
            "{:32} : {:4}".format(
                "Existing unique HP sets executed", len(self.trial_hparams_to_rundirs)
            )
        )
        print(
            "{:32} : {:4}".format(
                "Total existing trial run dirs",
                sum(len(v) for v in self.trial_hparams_to_rundirs.values()),
            )
        )
        print("{:32} : {:4}".format("Existing wandb runs", len(self.wandb_runs)))
        print("{:32} : {}".format("Algorithm's budgets", str(self.budgets)))
        sq = set(
            [
                j.strip()
                for j in os.popen("/opt/slurm/bin/squeue -u $USER -o '%12i'")
                .read()
                .splitlines()[1:]
            ]
        )
        running = set(self.job_ids) & sq
        waiting = (
            set([j.parent.name for j in rundir.glob(f"*/{self.name}.exp")]) & sq
        ) - running
        print(
            "{:32} : {}".format(
                "Jobs currently running:",
                f"{len(running)} " + " ".join(running),
            )
        )
        print(
            "{:32} : {}".format(
                "Jobs currently waiting:",
                f"{len(waiting)} " + " ".join(waiting),
            )
        )

    def discover_run_dirs(self):
        for unique in rundir.glob(f"*/{self.name}--*.unique"):
            self.trial_hparams_to_rundirs[unique.stem.split("--")[-1]].append(
                unique.parent
            )

    def get_dirs_for_trial(self, trial):
        if trial.hash_params in self.trial_hparams_to_rundirs:
            return self.trial_hparams_to_rundirs[trial.hash_params]
        else:
            print(f"No run dir for this trial with hparams {trial.hash_params}.")

    def get_trial_for_id(self, id):
        if id in self.id_to_trial:
            return self.id_to_trial[id]
        else:
            print("No trial for this id.")

    def get_dirs_for_id(self, id):
        return self.get_dirs_for_trial(self.get_trial_for_id(id))

    def get_reserved_wandb_runs(self):
        reserved = {}
        for trial_id, wandb_runs in self.id_to_wandb_runs.items():
            trial = self.get_trial_for_id(trial_id)
            if trial.status == "reserved":
                reserved[trial_id] = {"wandb_runs": wandb_runs, "trial": trial}
        return reserved

    def print_wandb_query(self):
        print(f"{'WandB runs query:':32}\n" + "(" + "|".join(self.job_ids) + ")")

    @classmethod
    def help(self):
        return dedent(
            """\
        --------------
        Manager init()
        --------------

        orion_db_path -> (str or pathlib.Path) pointing to the orion db pickle file
        name          -> (str) unique orion experiment name in the db
        wandb_path    -> (str) path to the wandb project like "{entity}/{project}"

        ----------
        Attributes
        ----------

        manager.trial_hparams_to_rundirs  -> dict {trial.params_hash: [list of run dirs]}
        manager.exp                       -> Orion experiment object
        manager.trials                    -> list of Orion trial objects for this exp
        manager.budgets                   -> list of budget of the exp's algorithm: n_trials and resources associated
        manager.total_budgets             -> total number of trials expected for this exp
        manager.id_to_trial               -> dict {trial_id: trial}
        manager.id_to_wandb_runs          -> dict {trial_id: [list of wandb Run objects]}
        manager.hash_to_trials             -> dict {hash_params: [list Orion trial objects]}

        -------
        Methods
        -------

        manager.get_dirs_for_trial(trial_obj: orion.Trial) -> list of run dirs for this trial
        manager.get_trial_for_id(trial_id: str)            -> trial object for this trial_id (wrapper around manager.id_to_trial[trial_id])
        manager.get_dirs_for_id(trial_id: str)             -> list of run dirs for this trial_id
        manager.get_reserved_wandb_runs()                  -> dict {trial_id: {"wandb_runs": [list of wandb Run objects], "trial": trial}}
                                                              get the currently reserved trials and their wandb runs

        --------
        Examples
        --------

        m = Manager(orion_db_path="./data/orion/storage/orion_db.pkl", name="ocp-qm9-orion-debug-v1.0.0", wandb_path="mila-ocp/ocp-qm")
        exp_df = m.exp.to_pandas()
        reserved_wandbs = m.get_reserved_wandb_runs()
        print(list(reserved_wandbs.values())[0]["wandb_runs"][0].config["run_dir"])
        """
        )


if __name__ == "__main__":
    defaults = {
        "help": False,
        "name": None,
        "wandb_path": None,
        "orion_db_path": str(
            Path(__file__).resolve().parent.parent.parent
            / "data/orion/storage/orion_db.pkl"
        ),
    }
    args = resolved_args(defaults=defaults)
    if args.help:
        print("🖱 Command-line (default) parameters:")
        print("\n".join("  {:15} : {}".format(k, v) for k, v in defaults.items()))
        print("\n\n🐍 Example command-line in IPython:")
        print(
            "In [1]: run ocpmodels/common/exp_manager.py",
            "name='ocp-qm9-orion-debug-v1.0.0' wandb_path='mila-ocp/ocp-3'",
        )
        print("\n\n🧞 Manager help:")
        print(Manager.help())
        sys.exit(0)

    if not args.name:
        raise ValueError(
            "Please provide `name=` for the experiment."
            + " See `$ python exp_manager.py help`"
        )
    if not args.wandb_path:
        raise ValueError(
            "Please provide `wandb_path='{entity}/{project}}'`."
            + " See `$ python exp_manager.py help`"
        )

    print(
        "💃 Status of experiment",
        f"'{args.name}' and wandb entity/project '{args.wandb_path}':",
    )
    m = Manager(
        name=args.name,
        wandb_path=args.wandb_path,
        orion_db_path=args.orion_db_path,
    )

    m.print_wandb_query()
    exp_df = m.exp.to_pandas()
    reserved_wandbs = m.get_reserved_wandb_runs()
