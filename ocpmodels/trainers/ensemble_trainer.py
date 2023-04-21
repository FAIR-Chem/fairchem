from pathlib import Path
from typing import List

import torch
from yaml import dump

from ocpmodels.common.utils import make_trainer_from_dir, resolve


class EnsembleTrainer:
    def __init__(self, config, load=True):
        """
        Create an ensemble of trainers from a config dict.

        config:
        {
            "checkpoints": Single run dir or list of run dirs, or path to specific ckpts
                and it will be assumed that chekpoints are in run_dir/checkpoints/*.pt.
            "dropout": float, optional, otherwise 0.75
        }

        If the provided ``checkpoints`` are not a list, or if they are
        a list with a single item, it is assumed that this should be
        an MC-Dropout ensemble.


        Args:
            config (dict): Ensemble config as a dict.
                Must contain the checkpoints to load.
            load (bool, optional): Whether to load checkpoints immediately
                or let the user call ``.load_checkpoints()``.
                Defaults to True.
        """
        assert isinstance(config, dict), "Ensemble config must be a dict"
        self.config = config

        assert (
            "checkpoints" in self.config
        ), "Ensemble config must have a 'checkpoints' key."

        self.checkpoints = self.config["checkpoints"]

        if not isinstance(self.checkpoints, list):
            self.checkpoints = [self.checkpoints]

        self.mc_dropout = len(self.checkpoints) == 1
        if self.mc_dropout:
            self.dropout = self.config.get("dropout", 0.75)

        self.trainers = []
        if load:
            self.load_trainers()

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

    def load_checkpoints(self):
        print("Loading checkpoints...")
        for c, ckpt in enumerate(self.checkpoints):
            # find actual checkpoint if a directory is provided
            ckpt = self.find_checkpoint(ckpt)

            print(f"  🚜 Loading trainer from: {str(ckpt)}")

            trainer = make_trainer_from_dir(
                ckpt, "continue", overrides={"load": False}, silent=True
            )
            trainer.load_seed_from_config()
            trainer.load_logger()
            trainer.load_datasets()
            trainer.load_task()
            trainer.load_model()
            # trainer.load_loss()
            # trainer.load_optimizer()
            # trainer.load_extras()
            trainer.load_checkpoint(ckpt)

            # load checkpoint
            if self.mc_dropout:
                print("    Setting model dropouts to: ", self.dropout)
                trainer.model.module.set_dropout(self.dropout)
            # store model in ``self.models`` list
            self.trainers.append(trainer)

        # Done!
        print("Loaded all checkpoints.")

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
                }
            )
        )

        deup_samples = []

        for d in dataset_strs:
            for batch_list in self.loaders[d]:
                b = batch_list[0]
                preds = self.forward(b, n=10)  # Batch x n
                pred_mean = preds.mean(dim=1)  # Batch
                pred_std = preds.std(dim=1)  # Batch
                loss = self.loss_fn["energy"](pred_mean, b.y_relaxed)
                deup_samples += [
                    {
                        "energy": b.y_relaxed,
                        "energy_pred": pred_mean,
                        "energy_std": pred_std,
                        "loss": loss,
                        "s": bool(d == "train"),
                        "ds": d,
                    }
                ]
                breakpoint()
        # TODO:
        # reshape deup_samples to be flat
        # write to disk deup_samples and dataset_attrs (lmbdb dataset)
        #   -> see make_qm7x_lmdbs.py
