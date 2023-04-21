import torch
from torch.nn.parallel.distributed import DistributedDataParallel

from ocpmodels.common import dist_utils
from ocpmodels.common.utils import resolve
from ocpmodels.common.data_parallel import OCPDataParallel
from ocpmodels.common.registry import registry


class UncertaintyEnsemble:
    def __init__(self, checkpoints, device=None, load=True):
        """
        Create an ensemble of models from a list of checkpoint paths.

        If the provided ``checkpoints`` are not a list, or if they are
        a list with a single item, it is assumed that this should be
        an MC-Dropout ensemble.


        Args:
            checkpoints (pathlike | List[pathlike]): Checkpoints to load. Can
                be a list of run directories, and it will be assumed that chekpoints
                are in run_dir/checkpoints/*.pt.
            device (str|torch.device, optional): Where to put the models. Will be on
                ``cpu`` or ``cuda:0`` if not provided.
            load (bool, optional): Whether to load checkpoints immediately
                or let the user call ``.load_checkpoints()``.
                Defaults to True.
        """
        if not isinstance(checkpoints, list):
            checkpoints = [checkpoints]

        self.mc_dropout = len(checkpoints) == 1
        self.checkpoints = checkpoints
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.models = []

        if load:
            self.load_checkpoints()

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
        for ckpt in self.checkpoints:
            # find actual checkpoint if a directory is provided
            ckpt = self.find_checkpoint(ckpt)

            print(f"  🚜 Loading checkpoint from: {str(ckpt)} onto {self.device}")

            # load checkpoint
            checkpoint = torch.load(ckpt)
            config = checkpoint["config"]
            # make model
            model = registry.get_model_class(config["model_name"])(**config["model"])
            model = OCPDataParallel(
                model,
                output_device=self.device,
                num_gpus=1 if "cpu" not in str(self.device) else 0,
            )
            if dist_utils.initialized():
                model = DistributedDataParallel(
                    model, device_ids=[self.device], output_device=self.device
                )
            # load state dict, taking multi-gpu into account
            first_key = next(iter(checkpoint["state_dict"]))
            if not dist_utils.initialized() and first_key.split(".")[1] == "module":
                new_dict = {k[7:]: v for k, v in checkpoint["state_dict"].items()}
            elif dist_utils.initialized() and first_key.split(".")[1] != "module":
                new_dict = {
                    f"module.{k}": v for k, v in checkpoint["state_dict"].items()
                }
            else:
                new_dict = checkpoint["state_dict"]
            model.load_state_dict(new_dict)

            # store model in ``self.models`` list
            self.models.append(model)

        # Done!
        print("Loaded all checkpoints.")

    def to(self, device):
        """
        Sends all models to the provided device.

        Args:
            device (str|torch.device): Target device.
        """
        for model in self.models:
            model.to(device)
        self.device = device

    def forward(self, batch, n=-1):
        """
        Passes a batch through the ensemble.
        Returns a tensor of shape (batch_size, n_models).
        Assumes we are interested in ``"energy"`` predictions.

        ``n`` is the number of models to use for inference.
        * In the case of a Deep ensemble, ``n`` must be less than the number
          of underlying models. It can be set to -1 to use all models. If
          ``0 < n < n_models`` then the models to use are randomly sampled.
        * In the case of an MC-Dropout ensemble, ``n`` must be > 0.

        Args:
            batch (torch_geometric.Data): Batch to forward through models
            n (int, optional): Number of inferences requested. Defaults to -1.

        Raises:
            ValueError: If ``n`` is larger than the number of models in the
                Deep ensemble.
            ValueError: If ``n`` is not > 0 in the case of an MC-Dropout ensemble.

        Returns:
            _type_: _description_
        """
        # MC-Dropout
        if self.mc_dropout:
            if n <= 0:
                raise ValueError("n must be > 0 for MC-Dropout ensembles.")
            return torch.cat(
                [self.models[0](batch)["energy"][:, None] for _ in range(n)],
                dim=-1,
            )

        # Deep ensemble
        if n > len(self.models):
            raise ValueError(f"n must be less than {len(self.models)}. Received {n}.")

        if n > 0:
            models_idx = torch.randperm(len(self.models))[:n].tolist()

            # concatenate random models' outputs for the Energy
            return torch.cat(
                [self.models[i](batch)["energy"][:, None] for i in models_idx],
                dim=-1,
            )

        if n != -1:
            print("Warning: n must be -1 to use all models. Using all models.")

        # concatenate all model outputs for the Energy
        return torch.cat(
            [model(batch)["energy"][:, None] for model in self.models],
            dim=-1,
        )
