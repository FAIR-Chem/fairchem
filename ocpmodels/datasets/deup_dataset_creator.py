import sys
from pathlib import Path
from typing import List
import pickle
import lmdb
import torch
from yaml import dump
from tqdm import tqdm

from ocpmodels.common.utils import (
    make_trainer_from_dir,
    resolve,
    JOB_ID,
    RUNS_DIR,
    set_cpus_to_workers,
)
from ocpmodels.datasets.lmdb_dataset import DeupDataset


def clean_conf(config):
    """
    Recursively convert Path objects to strings in a config dict.

    Args:
        config (dict): Config dict to clean.

    Returns:
        dict: Cleaned config dict.
    """
    for k, v in config.items():
        if isinstance(v, Path):
            config[k] = str(v)
        elif isinstance(v, dict):
            config[k] = clean_conf(v)
    return config


class DeupDatasetCreator:
    def __init__(self, trainers_conf={}, overrides={}, **kwargs):
        """
        Create an ensemble of trainers to create a dataset from those trainers' models.

        trainers_conf:
        {
            "checkpoints": Single run dir or list of run dirs, or path to specific ckpts
                and it will be assumed that chekpoints are in run_dir/checkpoints/*.pt.
            "dropout": float, optional, otherwise 0.75
        }
        overrides (dict): dict to override all the trainers' configs.

        If the provided ``checkpoints`` are not a list, or if they are
        a list with a single item, it is assumed that this should be
        an MC-Dropout ensemble.


        Args:
            trainers_conf (dict): Ensemble config as a dict.
                Must contain the checkpoints to load.
            overrides: dict of overrides for this trainer
        """
        self.config = {
            "trainers_conf": trainers_conf,
            "overrides": overrides,
            "kwargs": kwargs,
        }
        self.trainers_conf = trainers_conf
        self.checkpoints = self.trainers_conf.get("checkpoints") or kwargs.get(
            "checkpoints"
        )
        self.dropout = self.trainers_conf.get("dropout", 0.7)
        assert (
            self.checkpoints is not None
        ), "checkpoints must be provided in the yaml config or the kwargs"

        if not isinstance(self.checkpoints, list):
            self.checkpoints = [self.checkpoints]

        self.mc_dropout = len(self.checkpoints) == 1

        self.trainers = []
        self.load_trainers(overrides)
        # print("Loading self with shared config", shared_config, "...")
        print("Ready.")

    @classmethod
    def find_checkpoint(cls, ckpt):
        """
        Discover the best checkpoint from a run directory.
        If ``ckpt`` is a file, it is immediately returned.
        If it is a directory, it is assumed that the checkpoints are in
        ``ckpt/checkpoints/*.pt`` and the best checkpoint is returned if
        it exists. If it does not exist, the last checkpoint is returned.

        Args:
            ckpt (pathlike): Where to look for a checkpoint.

        Raises:
            FileNotFoundError: If no checkpoints are found in the
                ``ckpt`` directory.

        Returns:
            pathlib.Path: A path to a checkpoint.
        """
        path = resolve(ckpt)
        assert path.exists(), f"Checkpoint {str(path)} does not exist."

        # return path if it is a file with a warning if it is not a .pt file
        if path.is_file():
            if path.suffix != ".pt":
                print("Checkpoint should be a .pt file, received: ", path.name)
            return path

        # checkpoints should be in the ``ckpt/checkpoints/`` folder
        ckpt_dir = path / "checkpoints"
        ckpts = list(ckpt_dir.glob("*.pt"))

        if not ckpts:
            raise FileNotFoundError("No checkpoints found in: " + str(ckpt_dir))

        # tries to get the best checkpoint
        best = ckpt_dir / "best_checkpoint.pt"
        if best.exists():
            return best

        # returns the last checkpoint because no best checkpoint was found
        print("  >> ⁉️ Warning: no best checkpoint found, using last checkpoint.")
        return sorted(ckpts, key=lambda x: x.stat().st_mtime)[-1]

    def load_trainers(self, overrides={}):
        print("Loading checkpoints...")
        for c, ckpt_path in enumerate(self.checkpoints):
            # find actual checkpoint if a directory is provided
            ckpt_path = self.find_checkpoint(ckpt_path)

            print(f"  🚜 Loading trainer from: {str(ckpt_path)}")

            trainer = make_trainer_from_dir(
                ckpt_path,
                "continue",
                overrides={**{"load": False, "silent": True}, **overrides},
                silent=True,
            )
            trainer.config = set_cpus_to_workers(trainer.config, False)
            # trainer.load_seed_from_config()
            # trainer.load_logger()
            trainer.load_datasets()
            trainer.load_task()
            trainer.load_model()
            trainer.load_loss()
            # trainer.load_optimizer()
            # trainer.load_extras()
            trainer.load_checkpoint(ckpt_path, silent=True)

            # load checkpoint
            if self.mc_dropout:
                print("    Setting model dropouts to: ", self.dropout)
                trainer.model.module.set_dropouts(self.dropout)
            # store model in ``self.models`` list
            self.trainers.append(trainer)

        assert all(
            [
                t.config["graph_rewiring"] == self.trainers[0].config["graph_rewiring"]
                for t in self.trainers
            ]
        ), "All models must have the same graph rewiring setting."

        shared_config = {}
        shared_config["graph_rewiring"] = self.trainers[0].config["graph_rewiring"]
        shared_config["fa_frames"] = self.trainers[0].config["fa_frames"]
        shared_config["frame_averaging"] = self.trainers[0].config["frame_averaging"]

        # Done!
        print("Loaded all checkpoints.")
        return shared_config

    def forward(self, batch_list, n_samples=-1, shared_encoder=True):
        """
        Passes a batch_list through the ensemble.
        Returns a tensor of shape (batch_size, n_models).
        Assumes we are interested in ``"energy"`` predictions.

        ``n_samples`` is the number of models to use for inference.
        * In the case of a Deep ensemble, ``n_samples`` must be less than the number
          of underlying models. It can be set to -1 to use all models. If
          ``0 < n_samples < n_models`` then the models to use are randomly sampled.
        * In the case of an MC-Dropout ensemble, ``n_samples`` must be > 0.

        Args:
            batch_list (List[torch_geometric.Data]): batch list to forward through
                models
            n_samples (int, optional): Number of inferences requested. Defaults to -1.

        Raises:
            ValueError: If ``n_samples`` is larger than the number of models in the
                Deep ensemble.
            ValueError: If ``n_samples`` is not > 0 in the case of an MC-Dropout
                ensemble.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, n_samples) containing
        """
        # MC-Dropout

        def _structure(preds):
            return {
                "energies": torch.cat([p["energy"][:, None] for p in preds], dim=-1),
                "q": preds[0]["q"],
            }

        preds = []
        q = None
        trainers_idx = []
        trainer_idx = 0

        if self.mc_dropout:
            shared_encoder = True
        else:
            trainers_idx = torch.randperm(len(self.trainers))[
                : n_samples - len(preds)
            ].tolist()
            trainer_idx = trainers_idx[0]

        if shared_encoder:
            preds = [self.trainers[trainer_idx].model_forward(batch_list, mode="deup")]
            q = preds[0]["q"]

        if self.mc_dropout:
            if n_samples <= 0:
                raise ValueError("n_samples must be > 0 for MC-Dropout ensembles.")
            preds += [
                self.trainers[0].model_forward(batch_list, mode="deup", q=q)
                for _ in range(n_samples - len(preds))
            ]
            return _structure(preds)

        # Deep ensemble
        if n_samples > len(self.trainers):
            raise ValueError(
                f"n_samples must be <= {len(self.trainers)}. Received {n_samples}."
            )

        if n_samples > 0:
            # concatenate random models' outputs for the Energy
            preds += [
                self.trainers[i].model_forward(batch_list, mode="deup", q=q)
                for i in trainers_idx[1:]
            ]
            return _structure(preds)

        if n_samples != -1:
            print("Warning: n_samples must be -1 to use all models. Using all models.")

        # concatenate all model outputs for the Energy
        preds += [
            trainer.model_forward(batch_list, mode="deup")
            for trainer in self.trainers[1:]
        ]
        return _structure(preds)

    @torch.no_grad()
    def create_deup_dataset(
        self,
        output_path: str,
        dataset_strs: List[str],
        n_samples: int = 10,
        max_samples: int = -1,
        batch_size: int = None,
        shared_encoder: bool = True,
    ):
        """
        Checkpoints  : ["/network/.../"]
        dataset_strs : ["train", "val_id", "val_ood_cat"]

        Args:
            dataset_strs (List[str]): List of dataset strings to use for creating the
                deup dataset.
        """

        assert output_path
        output_path = resolve(output_path)
        if batch_size is not None:
            assert isinstance(batch_size, int)
            self.trainers[0].config["optim"]["batch_size"] = batch_size
            self.trainers[0].load_datasets()
            print(
                "Updated sub-trainer dataset batch_size to",
                self.trainers[0].loaders["train"].batch_sampler.batch_size,
            )
        output_path.mkdir(exist_ok=True)
        (output_path / "deup_config.yaml").write_text(
            dump(
                {
                    "dataset_trainer_config": clean_conf(self.config),
                    "n_samples": n_samples,
                    "datasets": dataset_strs,
                    "output_path": str(output_path),
                }
            )
        )

        self.trainers[0].load_loss(reduction="none")

        for trainer in self.trainers:
            trainer.model.module.set_deup_inference(True)

        stats = {d: {} for d in dataset_strs}

        for dataset_name in dataset_strs:
            deup_samples = []
            deup_ds_size = 0
            for batch_list in tqdm(
                self.trainers[0].loaders[dataset_name],
                desc=f"Infering on dataset: {dataset_name}",
            ):
                batch = batch_list[0]

                # {"energies": Batch x n, "q": Batch x hidden_dim}
                preds = self.forward(
                    batch_list, n_samples=n_samples, shared_encoder=True
                )

                pred_mean = preds["energies"].mean(dim=1)  # Batch
                pred_std = preds["energies"].std(dim=1)  # Batch
                loss = self.trainers[0].loss_fn["energy"](
                    pred_mean, batch.y_relaxed.to(pred_mean.device)
                )
                deup_samples += [
                    {
                        "energy_target": batch.y_relaxed.clone(),
                        "energy_pred_mean": pred_mean.clone(),
                        "energy_pred_std": pred_std.clone(),
                        "loss": loss.clone(),
                        "s": torch.full_like(
                            loss, bool(dataset_name == "train")
                        ).bool(),
                        "ds": [dataset_name for _ in range(len(loss))],
                        "idx_in_dataset": batch.idx_in_dataset.clone(),
                        "q": preds["q"].clone(),
                        "batch": batch.batch.clone(),
                    }
                ]
                deup_ds_size += len(loss)

                if max_samples > 0 and deup_ds_size >= max_samples:
                    break

            epm = torch.cat([s["energy_pred_mean"] for s in deup_samples], dim=0)
            stats[dataset_name]["mean"] = epm.mean().cpu().item()
            stats[dataset_name]["std"] = epm.std().cpu().item()
            (output_path / "deup_config.yaml").write_text(
                dump(
                    {
                        "dataset_trainer_config": clean_conf(self.config),
                        "n_samples": n_samples,
                        "datasets": dataset_strs,
                        "output_path": str(output_path),
                        "stats": stats,
                    }
                )
            )

            self.write_lmdb(
                deup_samples,
                output_path / f"{dataset_name}_deup_samples.lmdb",
                total_size=deup_ds_size,
                max_samples=max_samples,
            )
        return output_path

    def write_lmdb(self, samples, path, total_size=-1, max_samples=-1):
        env = lmdb.open(
            str(path),
            map_size=1099511627776 * 2,
            subdir=False,
            readonly=False,
            meminit=False,
            map_async=True,
            writemap=True,
            max_readers=100,
            max_dbs=1,
        )
        txn = env.begin(write=True)
        k = 0
        pbar = tqdm(
            total=total_size,
            desc="Writing LMDB DB in {}".format("/".join(path.parts[-3:])),
        )
        for i, sample in enumerate(samples):
            n = len(sample["energy_target"])
            sample = {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in sample.items()
            }
            batch = sample.pop("batch")
            batched_q = [sample["q"][batch == i, :] for i in range(batch.max() + 1)]
            assert len(batched_q) == n
            assert sum([len(q) for q in batched_q]) == len(sample["q"])
            for s in range(n):
                deup_sample = {
                    k: v[s].item()
                    if isinstance(v, torch.Tensor) and v[s].ndim == 0
                    else v[s]
                    for k, v in sample.items()
                    if k != "q"
                }
                deup_sample["q"] = batched_q[s]
                # remember: if storing raw data, need to clone() tensor before
                # pickling. It's fine for batched_q.
                txn.put(
                    key=str(k).encode("ascii"),
                    value=pickle.dumps(deup_sample, protocol=pickle.HIGHEST_PROTOCOL),
                )
                pbar.update(1)
                k += 1
                if max_samples > 0 and k == max_samples - 1:
                    break

        pbar.close()

        txn.commit()
        env.sync()
        env.close()


if __name__ == "__main__":
    # python -m ocpmodels.datasets.deup_dataset_creator

    from ocpmodels.datasets.data_transforms import get_transforms
    from ocpmodels.datasets.deup_dataset_creator import DeupDatasetCreator
    from ocpmodels.datasets.lmdb_dataset import DeupDataset
    from ocpmodels.common.utils import JOB_ID, RUNS_DIR, make_config_from_conf_str

    base_trainer_path = "/network/scratch/s/schmidtv/ocp/runs/3298908"

    # what models to load for inference
    trainers_conf = {
        "checkpoints": [base_trainer_path],
        "dropout": 0.7,
    }
    # setting first_trainable_layer to output means that the latent space
    # q will be defined as input to the output layer, even though the model
    # will not be trained anyway
    overrides = {"model": {"first_trainable_layer": "output"}}
    # where to store the lmdb dataset
    deup_dataset_path = RUNS_DIR / JOB_ID / "deup_dataset"
    # main interface
    ddc = DeupDatasetCreator(trainers_conf=trainers_conf, overrides=overrides)
    # create the dataset
    ddc.create_deup_dataset(
        output_path=deup_dataset_path,
        dataset_strs=["train", "val_id", "val_ood_cat", "val_ood_ads"],
        n_samples=7,
        max_samples=-1,
        batch_size=256,
    )

    base_config = ddc.trainers[0].config
    base_datasets_config = base_config["dataset"]
    # or:
    # base_config = make_config_from_conf_str("faenet-is2re-all")
    # base_datasets_config = base_config["dataset"]

    deup_dataset = DeupDataset(
        {
            **base_datasets_config,
            **{"deup-train-val_id": {"src": deup_dataset_path}},
        },
        "deup-train-val_id",
        transform=get_transforms(base_config),
    )

    deup_sample = deup_dataset[0]
