import wandb
from orion.client import get_experiment
from pathlib import Path
from collections import defaultdict, Counter
from textwrap import dedent
from minydra import resolved_args
import os
import sys
import time
from datetime import datetime
import yaml
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from ocpmodels.common.utils import ROOT, RUNS_DIR
from ocpmodels.common.orion_utils import get_and_move_orion_db_path

EXP_OUT_DIR = ROOT / "data" / "exp_outputs"
MANAGER_CACHE = ROOT / "data" / "exp_manager_cache"


class Manager:
    def __init__(
        self,
        orion_db_path="",
        name="",
        wandb_path="mila-ocp/ocp-qm",
        rebuild_cache=False,
        print_tracebacks=True,
    ):
        self.api = wandb.Api()
        self.wandb_path = wandb_path
        self.rebuild_cache = rebuild_cache
        self.print_tracebacks = print_tracebacks
        self.wandb_runs = [
            r
            for r in self.api.runs(wandb_path)
            if "orion_hash_params" in r.config
            and name in r.config.get("orion_exp_config_path", "")
        ]
        self.name = name
        self.cache_path = MANAGER_CACHE / f"{self.name}.yaml"
        self.cache = (
            yaml.safe_load(self.cache_path.read_text())
            if self.cache_path.exists()
            else {}
        )
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
        sq_cmd = (
            "/opt/slurm/bin/squeue"
            if "CC_CLUSTER" not in os.environ
            else "/opt/software/slurm/bin/squeue"
        )
        sq = set(
            [
                j.strip()
                for j in os.popen(f"{sq_cmd} -u $USER -o '%12i'")
                .read()
                .splitlines()[1:]
            ]
        )
        self.running_jobs = set(self.job_ids) & sq
        self.waiting_jobs = (
            set([j.parent.name for j in RUNS_DIR.glob(f"*/{self.name}.exp")]) & sq
        ) - self.running_jobs
        print("\n")
        self.discover_yamls()
        self.discover_job_ids_from_yaml()
        self.parse_output_files()
        self.print_status()
        print("\n")
        self.print_output_files_stats()

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

        print(
            "{:32} : {}".format(
                "Jobs currently running:",
                f"{len(self.running_jobs)} " + " ".join(sorted(self.running_jobs)),
            )
        )
        print(
            "{:32} : {}".format(
                "Jobs currently waiting:",
                f"{len(self.waiting_jobs)} " + " ".join(sorted(self.waiting_jobs)),
            )
        )

    def discover_run_dirs(self):
        for unique in RUNS_DIR.glob(f"*/{self.name}--*.unique"):
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

    def parse_output_files(self):
        if "job_state" not in self.cache:
            self.cache["job_state"] = {}
        for j in tqdm(self.cache["all_job_ids"], desc="Parsing output files"):
            if j in self.cache["job_state"] and not self.rebuild_cache:
                continue
            if j in self.waiting_jobs:
                self.cache["job_state"][j] = "Waiting"
                continue
            if j in self.running_jobs:
                self.cache["job_state"][j] = "Running"
                continue
            out_file = RUNS_DIR / j / "output-0.txt"

            if not out_file.exists():
                self.cache["job_state"][j] = "No output file (RaceCondition)"
                continue

            out_txt = out_file.read_text()
            if "eval_all_splits" in out_txt and "Final results" in out_txt:
                self.cache["job_state"][j] = "Finished"
            elif "DUE TO TIME LIMIT" in out_txt:
                self.cache["job_state"][j] = "TimeLimit"
            elif "RaceCondition" in out_txt:
                self.cache["job_state"][j] = "RaceCondition"
            elif "DatabaseTimeout: Could not acquire lock for PickledDB" in out_txt:
                self.cache["job_state"][j] = "DatabaseTimeout"
            elif (
                "Algo does not have more trials to sample.Waiting for current trials to finish"  # noqa: E501
                in out_txt
            ):
                self.cache["job_state"][j] = "WaitingForTrials"
            elif (
                "RuntimeError: Trying to create tensor with negative dimension"
                in out_txt
            ):
                self.cache["job_state"][j] = "NegativeEmbeddingDimension"
            elif "Traceback" in out_txt:
                self.cache["job_state"][j] = (
                    "Traceback: " + out_txt.split("Traceback")[1]
                )
            elif "srun: Job step aborted" in out_txt:
                if "slurmstepd" in out_txt and " CANCELLED AT " in out_txt:
                    self.cache["job_state"][j] = "Cancelled"
            elif "Loss is NaN. Stopping training." in out_txt:
                self.cache["job_state"][j] = "NaN loss"
            else:
                self.cache["job_state"][j] = "Unknown"
        self.commit_cache()

    def print_output_files_stats(self):
        print("Job status from output files:\n" + "-" * 29 + "\n")
        stats = {}
        for j, o in self.cache["job_state"].items():
            if "Traceback" in o:
                if "Traceback" not in stats:
                    stats["Traceback"] = {"n": 0, "ids": [], "contents": []}
                stats["Traceback"]["n"] += 1
                stats["Traceback"]["ids"].append(j)
                stats["Traceback"]["contents"].append(o)
            else:
                if o not in stats:
                    stats[o] = {"n": 0, "ids": []}
                stats[o]["n"] += 1
                stats[o]["ids"].append(j)
        for s, v in stats.items():
            print(f"\n• {s:31}" + f": {v['n']}\n    " + " ".join(v["ids"]))
        if stats["Traceback"]["n"] > 0 and self.print_tracebacks:
            print("\nTraceback contents:\n" + "-" * 19 + "\n")
            print(
                f"\n\n{'|' * 50}\n{'|' * 50}\n{'|' * 50}\n".join(
                    f"{j}:\n{o}"
                    for j, o in zip(
                        stats["Traceback"]["ids"], stats["Traceback"]["contents"]
                    )
                )
            )

    def discover_job_ids_from_yaml(self):
        all_jobs = (
            set(self.cache.get("all_job_ids", [])) if not self.rebuild_cache else set()
        )
        for yaml_path in self.cache["exp_yamls"]:
            lines = Path(yaml_path).read_text().splitlines()
            jobs_line = [line for line in lines if "All jobs launched" in line][0]
            jobs = [
                j.strip()
                for j in jobs_line.split("All jobs launched: ")[-1].strip().split(", ")
            ]
            all_jobs |= set(jobs)
        self.cache["all_job_ids"] = sorted(all_jobs)
        self.commit_cache()

    def discover_yamls(self):
        yamls = set()
        if self.cache and not self.rebuild_cache:
            cache_yamls = self.cache.get("exp_yamls") or []
            yamls |= set(cache_yamls)
        for yaml_conf in EXP_OUT_DIR.glob("**/*.yaml"):
            if str(yaml_conf) not in yamls:
                yaml_txt = yaml_conf.read_text()
                if self.name in yaml_txt:
                    y = yaml.safe_load(yaml_txt)
                    if y.get("orion", {}).get("unique_exp_name") == self.name:
                        yamls.add(str(yaml_conf))
        yamls = sorted(yamls)
        self.cache["exp_yamls"] = yamls
        self.commit_cache()

    def commit_cache(self):
        if not self.cache_path.parent.exists():
            self.cache_path.parent.mkdir(parents=True)
        self.cache_path.write_text(yaml.safe_dump(self.cache))

    @classmethod
    def help(self):
        return dedent(
            """\
        --------------
        Manager init()
        --------------

        orion_db_path    -> (str or pathlib.Path) pointing to the orion db pickle file
        name             -> (str) unique orion experiment name in the db
        wandb_path       -> (str) path to the wandb project like "{entity}/{project}"
        rebuild_cache    -> (bool, default: False) if True, will rebuild the output file cache from scratch
        print_tracebacks -> (bool, default: False) if True, will print the Traceback contents in the output files

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
        "watch": -1,
        "rebuild_cache": False,
        "print_tracebacks": False,
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
        print(
            "In [1]: run ocpmodels/common/exp_manager.py",
            "name='ocp-qm9-orion-debug-v1.0.0' wandb_path='mila-ocp/ocp-3'",
            "print_tracebacks",
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
    orion_db_path = get_and_move_orion_db_path(args.name)
    m = Manager(
        name=args.name,
        wandb_path=args.wandb_path,
        orion_db_path=orion_db_path,
        rebuild_cache=args.rebuild_cache,
        print_tracebacks=args.print_tracebacks,
    )

    # m.print_wandb_query()
    # exp_df = m.exp.to_pandas()
    # reserved_wandbs = m.get_reserved_wandb_runs()

    if args.watch and args.watch > 0:
        if args.watch < 15:
            print("Cannot watch to often, setting to 15 seconds.")
            args.watch = 15
        try:
            print("👀 Watching for exp status every every", args.watch, "seconds.")
            while True:
                time.sleep(args.watch)
                print()
                print("=" * 30)
                print("=" * 30)
                print()
                print(
                    "💃 Status of experiment",
                    f"'{args.name}' and wandb entity/project '{args.wandb_path}' @",
                    str(datetime.now()).split(".")[0],
                )
                print()
                m = Manager(
                    name=args.name,
                    wandb_path=args.wandb_path,
                    orion_db_path=args.orion_db_path,
                )
        except KeyboardInterrupt:
            print("👋 Exiting.")
            sys.exit(0)
