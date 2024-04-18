import copy
import sys
from pathlib import Path

import torch
import torch.optim as optim
import tqdm
from denmarf import DensityEstimate
from denmarf import flows as fnn
from denmarf.transform import LogitTransform
from torch.utils.data import DataLoader
from torch_scatter import scatter
import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ocpmodels.common.data_parallel import BalancedBatchSampler, ParallelCollater
from ocpmodels.common.utils import set_cpus_to_workers, set_deup_samples_path, resolve
from ocpmodels.datasets.data_transforms import get_transforms
from ocpmodels.datasets.lmdb_dataset import DeupDataset, LmdbDataset


def get_dataloader(dataset, sampler, parallel_collater, num_workers=1):
    loader = DataLoader(
        dataset,
        collate_fn=parallel_collater,
        num_workers=num_workers,
        pin_memory=True,
        batch_sampler=sampler,
    )
    return loader


def get_sampler(dataset, batch_size, shuffle, device):
    sampler = BalancedBatchSampler(
        dataset,
        batch_size=batch_size,
        num_replicas=1,
        rank=0,
        device=device,
        mode="atoms",
        shuffle=shuffle,
        force_balancing=False,
    )
    return sampler


def make_ocp_loaders(config, device):
    transform = get_transforms(config)  # TODO: train/val/test behavior
    batch_size = config["optim"]["batch_size"]
    datasets = {}
    samplers = {}
    loaders = {}

    parallel_collater = ParallelCollater(0 if "cpu" in str(device) else 1, False)

    for split, ds_conf in config["dataset"].items():
        if split == "default_val":
            continue

        if "deup" in split:
            datasets[split] = DeupDataset(config["dataset"], split, transform=transform)
        else:
            datasets[split] = LmdbDataset(ds_conf, transform=transform)

        shuffle = False
        if "train" in split:
            shuffle = True

        samplers[split] = get_sampler(datasets[split], batch_size, shuffle, device)
        loaders[split] = get_dataloader(
            datasets[split],
            samplers[split],
            parallel_collater,
            config["optim"]["num_workers"],
        )

    return loaders


def ocp_fit(
    de,
    loaders,
    num_features,
    bounded=None,
    lower_bounds=None,
    upper_bounds=None,
    num_blocks=32,
    num_hidden=128,
    num_epochs=1000,
    learning_rate=1e-3,
    weight_decay=1e-6,
    batch_size=None,
    p_train=0.5,
    p_test=0.1,
    verbose=True,
):
    """Fit the density estimate to the data.

    Parameters
    ----------
    X : numpy.ndarray
        Training samples.
    bounded : bool, optional
        Whether the distribution is bounded. If True, the distribution will be transformed to the unbounded space
        using logistic transformation.
    lower_bounds : numpy.ndarray, optional
        Lower bounds of the bounded distribution.
    upper_bounds : numpy.ndarray, optional
        Upper bounds of the bounded distribution.
    num_blocks : int, optional
        Number of blocks.
    num_hidden : int, optional
        Number of hidden units.
    num_epochs : int, optional
        Number of epochs.
    learning_rate : float, optional
        Learning rate.
    weight_decay : float, optional
        Weight decay.
    batch_size : int, optional
        Batch size.
    p_train : float, optional
        Percentage (0 < p_train < 1.0) of training samples.
    p_test : float, optional
        Percentage (0 < p_test < 1.0) of test samples. The rest of the samples will be used for validation.
    verbose : bool, optional
        Whether to print progress.

    """

    # For compatibility with older version
    # Deprecation notice: the bounded option will be moved to .fit() in newer versions
    if bounded is not None:
        de.bounded = bounded

    assert not de.bounded

    # Perform logit transformation if distribution is bounded
    if de.bounded:
        assert (
            lower_bounds is not None
        ), "lower_bounds must be specified for bounded distribution"
        assert (
            upper_bounds is not None
        ), "upper_bounds must be specified for bounded distribution"
        de.transformation = LogitTransform(lower_bounds, upper_bounds)

        X = de.transformation.logit_transform(X)

    # Split the data set into training set, validation set and test set

    train_dataloader = loaders["deup-train-val_id"]
    validate_dataloader = loaders["deup-val_ood_cat-val_ood_ads"]

    de.num_features = num_features
    de.num_blocks = num_blocks
    de.num_hidden = num_hidden

    if de.model is None:
        # Construct a new model
        model = de.construct_model(de.num_features, de.num_blocks, de.num_hidden)
    else:
        # Resume from previously saved model
        model = de.model
    model.to(de.device)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    def train(epoch):
        model.train()
        train_loss = 0

        pbar = tqdm.tqdm(train_dataloader, leave=False)
        for batch_idx, batch in enumerate(pbar):
            if isinstance(batch, list):
                if len(batch) > 1:
                    cond_data = batch[1].float()
                    cond_data = cond_data.to(de.device)
                else:
                    cond_data = None

                batch = batch[0]

            data = batch.deup_q.float()
            data = scatter(data, batch.batch, dim=0, reduce="mean")
            data = data.to(de.device)
            optimizer.zero_grad()
            loss = -model.log_probs(data, cond_data).mean()
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                pbar.set_description("Loss: {:.4f}".format(loss.item()))

        for module in model.modules():
            if isinstance(module, fnn.BatchNormFlow):
                module.momentum = 0

        with torch.no_grad():
            sample = train_dataloader.dataset[0].to(data.device)
            data = scatter(sample.deup_q, sample.batch, dim=0, reduce="mean")
            model(data)

        for module in model.modules():
            if isinstance(module, fnn.BatchNormFlow):
                module.momentum = 1

    def validate(epoch, model, loader):
        model.eval()
        val_loss = 0

        pbar = tqdm.tqdm(loader, leave=False)

        for batch_idx, batch in enumerate(pbar):
            if isinstance(batch, list):
                if len(batch) > 1:
                    cond_data = batch[1].float()
                    cond_data = cond_data.to(de.device)
                else:
                    cond_data = None

                batch = batch[0]
            data = batch.deup_q.float()
            data = scatter(data, batch.batch, dim=0, reduce="mean")
            data = data.to(de.device)
            with torch.no_grad():
                val_loss += (
                    -model.log_probs(data, cond_data).sum().item()
                )  # sum up batch loss
            pbar.set_description(f"Val loss: {val_loss / (batch_idx + 1):.4f}")

        return val_loss / len(loader.dataset)

    # Start training the network
    best_validation_loss = float("inf")
    best_validation_epoch = 0
    de.best_model = model

    if verbose:
        pbar = tqdm.tqdm(total=num_epochs)
    for epoch in range(num_epochs):
        train(epoch)
        validation_loss = validate(epoch, model, validate_dataloader)

        if verbose:
            pbar.update()
            pbar.set_description(
                "current average log likelihood: {:.3f}".format(-validation_loss)
            )

        if validation_loss < best_validation_loss:
            best_validation_epoch = epoch
            best_validation_loss = validation_loss
            de.best_model = copy.deepcopy(model)

    best_validation_loss = validate(
        best_validation_epoch, de.best_model, validate_dataloader
    )
    if verbose:
        print("best average log likelihood: {:.3f}".format(-best_validation_loss))
    de.model = de.best_model
    return de


if __name__ == "__main__":
    # https://github.com/ricokaloklo/denmarf
    # X is some np ndarray

    rundir = resolve("$SCRATCH/ocp/runs/$SLURM_JOB_ID")

    ocp_config = {
        "dataset": {
            "default_val": "deup-val_ood_cat-val_ood_ads",
            "deup-train-val_id": {
                "src": "/network/scratch/s/schmidtv/ocp/runs/3301084/deup_dataset"
            },
            "deup-val_ood_cat-val_ood_ads": {
                "src": "/network/scratch/s/schmidtv/ocp/runs/3301084/deup_dataset"
            },
            "train": {
                "src": "/network/scratch/s/schmidtv/ocp/datasets/ocp/is2re/all/train/",
                "normalize_labels": True,
            },
            "val_id": {
                "src": "/network/scratch/s/schmidtv/ocp/datasets/ocp/is2re/all/val_id/"
            },
            "val_ood_cat": {
                "src": "/network/scratch/s/schmidtv/ocp/datasets/ocp/is2re/all/val_ood_cat/"
            },
            "val_ood_ads": {
                "src": "/network/scratch/s/schmidtv/ocp/datasets/ocp/is2re/all/val_ood_ads/"
            },
            "val_ood_both": {
                "src": "/network/scratch/s/schmidtv/ocp/datasets/ocp/is2re/all/val_ood_both/"
            },
        },
        "optim": {
            "batch_size": 1024,
            "num_workers": 0,
        },
        "frame_averaging": None,
        "fa_method": None,
        "silent": False,
        "graph_rewiring": "remove-tag-0",
        "de": {
            "num_blocks": 32,
            "num_hidden": 128,
            "num_epochs": 1000,
            "learning_rate": 1e-3,
            "weight_decay": 1e-6,
        },
    }

    de = DensityEstimate()
    ocp_config = set_deup_samples_path(ocp_config)
    ocp_config = set_cpus_to_workers(ocp_config)
    loaders = make_ocp_loaders(ocp_config, de.device)

    num_features = loaders["deup-train-val_id"].dataset[0].deup_q.shape[-1]

    de = ocp_fit(
        de,
        loaders,
        num_features,
        bounded=None,
        lower_bounds=None,
        upper_bounds=None,
        num_blocks=ocp_config["de"]["num_blocks"],
        num_hidden=ocp_config["de"]["num_hidden"],
        num_epochs=ocp_config["de"]["num_epochs"],
        learning_rate=ocp_config["de"]["learning_rate"],
        weight_decay=ocp_config["de"]["weight_decay"],
        batch_size=None,
        p_train=0.5,
        p_test=0.1,
        verbose=True,
    )

    de.save(str(rundir / "ocp_de.pkl"))
    (rundir / "de_config.yaml").write_text(yaml.safe_dump(ocp_config))

    # Compute the logpdf using the density estimate
    # logpdf_maf = de.score_samples(xeval)

    # Compute the logpdf using the exact form
    # logpdf_truth = gaussian_dist.logpdf(xeval)
