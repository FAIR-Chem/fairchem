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
    merge_dicts,
    make_config_from_conf_str,
)
from ocpmodels.trainers.single_trainer import SingleTrainer
from ocpmodels.common.registry import registry


@registry.register_trainer("ensemble")
class EnsembleTrainer(SingleTrainer):
    def __init__(self, conf_str=None, **kwargs):
        """
        Create an ensemble of trainers from an ensemble_config dict.

        ensemble_config:
        {
            "checkpoints": Single run dir or list of run dirs, or path to specific ckpts
                and it will be assumed that chekpoints are in run_dir/checkpoints/*.pt.
            "dropout": float, optional, otherwise 0.75
        }

        If the provided ``checkpoints`` are not a list, or if they are
        a list with a single item, it is assumed that this should be
        an MC-Dropout ensemble.


        Args:
            ensemble_config (dict): Ensemble config as a dict.
                Must contain the checkpoints to load.
            overrides: dict of overrides for this trainer
        """
        if conf_str is not None:
            assert isinstance(conf_str, str), "conf_str must be a string"
            self.config = make_config_from_conf_str(conf_str)
        else:
            self.config = {}
        self.checkpoints = self.config.get("ensemble_checkpoints") or kwargs.get(
            "ensemble_checkpoints"
        )
        assert (
            self.checkpoints is not None
        ), "ensemble_checkpoints must be provided in the yaml config or the kwargs"

        if not isinstance(self.checkpoints, list):
            self.checkpoints = [self.checkpoints]

        self.mc_dropout = len(self.checkpoints) == 1
        if self.mc_dropout:
            if "ensemble_dropout" not in self.config:
                print("WARNING: using MC-Dropout without specifying dropout rate")
                print("Using `ensemble_dropout: 0.75`")
            self.dropout = self.config.get("ensemble_dropout", 0.75)

        self.trainers = []
        shared_config = self.load_trainers()
        print("Loading self with shared config", shared_config, "...")
        self.config = merge_dicts(self.config, shared_config)
        self.config = merge_dicts(self.config, kwargs)
        if "silent" not in self.config:
            self.config["silent"] = True
        super().__init__(**self.config)
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

    def load_datasets(self):
        super().load_datasets()

    def load_trainers(self):
        print("Loading checkpoints...")
        for c, ckpt in enumerate(self.checkpoints):
            # find actual checkpoint if a directory is provided
            ckpt = self.find_checkpoint(ckpt)

            print(f"  🚜 Loading trainer from: {str(ckpt)}")

            trainer = make_trainer_from_dir(
                ckpt, "continue", overrides={"load": False, "silent": True}, silent=True
            )
            # trainer.load_seed_from_config()
            # trainer.load_logger()
            trainer.load_datasets()
            trainer.load_task()
            trainer.load_model()
            # trainer.load_loss()
            # trainer.load_optimizer()
            # trainer.load_extras()
            trainer.load_checkpoint(ckpt, silent=True)

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

        # Done!
        print("Loaded all checkpoints.")
        return shared_config

    def forward(self, batch_list, n_samples=-1):
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
        if self.mc_dropout:
            if n_samples <= 0:
                raise ValueError("n_samples must be > 0 for MC-Dropout ensembles.")
            return torch.cat(
                [
                    self.trainers[0].model_forward(batch_list, mode="deup")["energy"][
                        :, None
                    ]
                    for _ in range(n_samples)
                ],
                dim=-1,
            )

        # Deep ensemble
        if n_samples > len(self.models):
            raise ValueError(
                f"n_samples must be less than {len(self.models)}. Received {n_samples}."
            )

        if n_samples > 0:
            models_idx = torch.randperm(len(self.models))[:n_samples].tolist()

            # concatenate random models' outputs for the Energy
            return torch.cat(
                [
                    self.models[i](batch_list, mode="deup")["energy"][:, None]
                    for i in models_idx
                ],
                dim=-1,
            )

        if n_samples != -1:
            print("Warning: n_samples must be -1 to use all models. Using all models.")

        # concatenate all model outputs for the Energy
        return torch.cat(
            [
                model(batch_list, mode="deup")["energy"][:, None]
                for model in self.models
            ],
            dim=-1,
        )

    @torch.no_grad()
    def create_deup_dataset(
        self,
        dataset_strs: List[str],
        n_samples: int = 10,
        output_path: str = None,
        max_samples: int = -1,
    ):
        """
        Checkpoints  : ["/network/.../"]
        dataset_strs : ["train", "val_id", "val_ood_cat"]

        Args:
            checkpoints (List[str]): _description_
            dataset_strs (List[str]): _description_
        """

        if output_path is None:
            output_path = Path(self.config["run_dir"]) / "deup_dataset"
        output_path.mkdir(exist_ok=True)
        (output_path / "deup_config.yaml").write_text(
            dump(
                {
                    "ensemble_config": self.config,
                    "n_samples": n_samples,
                    "datasets": dataset_strs,
                    "output_path": str(output_path),
                }
            )
        )

        self.load_loss(reduction="none")
        stats = {d: {} for d in dataset_strs}

        for dataset_name in dataset_strs:
            deup_samples = []
            deup_ds_size = 0
            for batch_list in tqdm(
                self.trainers[0].loaders[dataset_name],
                desc=f"Infering on dataset: {dataset_name}",
            ):
                batch = batch_list[0]
                preds = self.forward(batch_list, n_samples=10)  # Batch x n
                pred_mean = preds.mean(dim=1)  # Batch
                pred_std = preds.std(dim=1)  # Batch
                loss = self.loss_fn["energy"](
                    pred_mean, batch.y_relaxed.to(pred_mean.device)
                )
                deup_samples += [
                    {
                        "energy_target": batch.y_relaxed,
                        "energy_pred_mean": pred_mean,
                        "energy_pred_std": pred_std,
                        "loss": loss,
                        "s": torch.full_like(
                            loss, bool(dataset_name == "train")
                        ).bool(),
                        "ds": [dataset_name for _ in range(len(loss))],
                        "idx_in_dataset": batch.idx_in_dataset,
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
                        "ensemble_config": self.config,
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
            for s in range(n):
                deup_sample = {
                    k: v[s].item() if isinstance(v, torch.Tensor) else v[s]
                    for k, v in sample.items()
                }
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
