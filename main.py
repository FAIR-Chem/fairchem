import argparse
import datetime
import json
import os
import pickle
import random
import sys
import time
import warnings
from random import sample

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from cgcnn.data import collate_pool, get_train_val_test_loader
from cgcnn.meter import AverageMeter, mae, mae_ratio
from cgcnn.model import CrystalGraphConvNet
from cgcnn.normalizer import Normalizer
from cgcnn.utils import class_eval, save_checkpoint

parser = argparse.ArgumentParser(
    description="Graph Neural Networks for Chemistry"
)
parser.add_argument(
    "--config-yml",
    default="configs/ulissigroup_co/cgcnn.yml",
    help="Path to a config file listing data, model, optim parameters.",
)
parser.add_argument(
    "--identifier",
    default="",
    help="Experiment identifier to append to checkpoint/log/result directory",
)
parser.add_argument(
    "--num-workers",
    default=0,
    type=int,
    help="Number of dataloader workers (default: 0 i.e. use main proc)",
)
parser.add_argument(
    "--print-every",
    default=10,
    type=int,
    help="Log every N iterations (default: 10)",
)
parser.add_argument(
    "--seed", default=0, type=int, help="Seed for torch, cuda, numpy"
)

# =============================================================================
#   INPUT ARGUMENTS AND CONFIG
# =============================================================================

args = parser.parse_args()

# https://pytorch.org/docs/stable/notes/randomness.html
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

config = yaml.safe_load(open(args.config_yml, "r"))

includes = config.get("includes", [])
if not isinstance(includes, list):
    raise AttributeError(
        "Includes must be a list, {} provided".format(type(includes))
    )

for include in includes:
    include_config = yaml.safe_load(open(include, "r"))
    config.update(include_config)

config.pop("includes")

args.cuda = torch.cuda.is_available()

args.timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
if args.identifier:
    args.timestamp += "-{}".format(args.identifier)
args.checkpoint_dir = os.path.join("checkpoints", args.timestamp)
args.results_dir = os.path.join("results", args.timestamp)
args.logs_dir = os.path.join("logs", args.timestamp)

os.makedirs(args.checkpoint_dir)
os.makedirs(args.results_dir)
os.makedirs(args.logs_dir)

print(yaml.dump(config, default_flow_style=False))
for arg in vars(args):
    print("{:<20}: {}".format(arg, getattr(args, arg)))

config["cmd"] = args.__dict__
del args

# Dump config parameters
json.dump(
    config,
    open(os.path.join(config["cmd"]["checkpoint_dir"], "config.json"), "w"),
)

# Tensorboard
log_writer = SummaryWriter(config["cmd"]["logs_dir"])


def main():
    # =========================================================================
    #   SETUP DATALOADER, NORMALIZER, MODEL, LOSS, OPTIMIZER
    # =========================================================================

    if config["task"]["type"] == "regression":
        best_mae_error = 1e10
    else:
        best_mae_error = 0.0

    # TODO: move this out to a separate dataloader interface.
    print("### Loading {}".format(config["task"]["dataset"]))
    if config["task"]["dataset"] in [
        "ulissigroup_co",
        "qm9",
        "xie_grossman_mat_proj",
    ]:
        data = pickle.load(open(config["dataset"]["src"], "rb"))

        structures = data[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]
        num_targets = len(structures[-1])

        if "label_index" in config["task"]:
            num_targets = 1

    else:

        raise NotImplementedError

    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=data,
        collate_fn=collate_pool,
        batch_size=config["optim"]["batch_size"],
        train_size=config["dataset"]["train_size"],
        val_size=config["dataset"]["val_size"],
        test_size=config["dataset"]["test_size"],
        num_workers=config["cmd"]["num_workers"],
        pin_memory=config["cmd"]["cuda"],
        return_test=True,
    )

    # Obtain target value normalizer
    if config["task"]["type"] == "classification":
        raise NotImplementedError
        normalizer = Normalizer(torch.zeros(2))
        normalizer.load_state_dict({"mean": 0.0, "std": 1.0})
    else:
        # Compute mean, std of training set labels.
        _, targets = collate_pool(
            [data[i] for i in range(config["dataset"]["train_size"])]
        )
        if "label_index" in config["task"]:
            targets = targets[:, int(config["task"]["label_index"])]
        normalizer = Normalizer(targets)

    # Build model
    model = CrystalGraphConvNet(
        orig_atom_fea_len,
        nbr_fea_len,
        num_targets,
        classification=config["task"]["type"] == "classification",
        **config["model"],
    )
    if config["cmd"]["cuda"]:
        model.cuda()

    if config["task"]["type"] == "classification":
        criterion = nn.NLLLoss()
    else:
        criterion = nn.L1Loss()

    optimizer = optim.AdamW(model.parameters(), config["optim"]["lr_initial"])

    scheduler = MultiStepLR(
        optimizer,
        milestones=config["optim"]["lr_milestones"],
        gamma=config["optim"]["lr_gamma"],
    )

    # =========================================================================
    #   TRAINING LOOP
    # =========================================================================

    for epoch in range(config["optim"]["max_epochs"]):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, normalizer)

        # evaluate on validation set
        mae_error = validate(val_loader, model, criterion, epoch, normalizer)

        if mae_error != mae_error:
            print("Exit due to NaN")
            sys.exit(1)

        scheduler.step()

        # remember the best mae_eror and save checkpoint
        if config["task"]["type"] == "regression":
            is_best = mae_error < best_mae_error
            best_mae_error = min(mae_error, best_mae_error)
        else:
            is_best = mae_error > best_mae_error
            best_mae_error = max(mae_error, best_mae_error)
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_mae_error": best_mae_error,
                "optimizer": optimizer.state_dict(),
                "normalizer": normalizer.state_dict(),
                "config": config,
            },
            is_best,
            config["cmd"]["checkpoint_dir"],
        )

    # Evaluate best model
    print("---------Evaluate Model on Test Set---------------")
    best_checkpoint = torch.load(
        os.path.join(config["cmd"]["checkpoint_dir"], "model_best.pth.tar")
    )
    model.load_state_dict(best_checkpoint["state_dict"])
    validate(test_loader, model, criterion, epoch, normalizer, test=True)


def train(train_loader, model, criterion, optimizer, epoch, normalizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    if config["task"]["type"] == "regression":
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if "label_index" in config["task"]:
            target = target[:, int(config["task"]["label_index"])].view(-1, 1)

        if config["cmd"]["cuda"]:
            input_var = (
                input[0].cuda(),
                input[1].cuda(),
                input[2].cuda(),
                [crys_idx.cuda() for crys_idx in input[3]],
            )
        else:
            input_var = (input[0], input[1], input[2], input[3])
        # normalize target
        if config["task"]["type"] == "regression":
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        if config["cmd"]["cuda"]:
            target_var = target_normed.cuda()
        else:
            target_var = target_normed

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        if config["task"]["type"] == "regression":
            mae_error = eval(config["task"]["metric"])(
                normalizer.denorm(output.data.cpu()), target
            )
            losses.update(loss.item(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
        else:
            accuracy, precision, recall, fscore, auc_score = class_eval(
                output.data.cpu(), target
            )
            losses.update(loss.item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if config["task"]["type"] == "regression":
            log_writer.add_scalar(
                "Training Loss", losses.val, epoch * len(train_loader) + i
            )
            log_writer.add_scalar(
                "Training MAE", mae_errors.val, epoch * len(train_loader) + i
            )
        else:
            log_writer.add_scalar(
                "Training Loss", losses.val, epoch * len(train_loader) + i
            )
            log_writer.add_scalar(
                "Training Accuracy",
                accuracies.val,
                epoch * len(train_loader) + i,
            )
            log_writer.add_scalar(
                "Training Precision",
                precisions.val,
                epoch * len(train_loader) + i,
            )
            log_writer.add_scalar(
                "Training Recall", recalls.val, epoch * len(train_loader) + i
            )
            log_writer.add_scalar(
                "Training F1", fscores.val, epoch * len(train_loader) + i
            )
            log_writer.add_scalar(
                "Training AUC", auc_scores.val, epoch * len(train_loader) + i
            )

        log_writer.add_scalar(
            "Learning rate",
            optimizer.param_groups[0]["lr"],
            epoch * len(train_loader) + i,
        )

        if i % config["cmd"]["print_every"] == 0:
            if config["task"]["type"] == "regression":
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss: {loss.val:.4f} ({loss.avg:.4f})\t"
                    "MAE: {mae_errors.val:.3f} ({mae_errors.avg:.3f})\t"
                    "Data: {data_time.val:.3f}s\t"
                    "Fwd/bwd: {batch_time.val:.3f}s\t".format(
                        epoch,
                        i,
                        len(train_loader),
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        mae_errors=mae_errors,
                    )
                )
            else:
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss: {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accu: {accu.val:.3f} ({accu.avg:.3f})\t"
                    "Precision: {prec.val:.3f} ({prec.avg:.3f})\t"
                    "Recall: {recall.val:.3f} ({recall.avg:.3f})\t"
                    "F1: {f1.val:.3f} ({f1.avg:.3f})\t"
                    "AUC: {auc.val:.3f} ({auc.avg:.3f})"
                    "Data: {data_time.val:.3f}s\t"
                    "Fwd/bwd: {batch_time.val:.3f}s\t".format(
                        epoch,
                        i,
                        len(train_loader),
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        accu=accuracies,
                        prec=precisions,
                        recall=recalls,
                        f1=fscores,
                        auc=auc_scores,
                    )
                )


def validate(val_loader, model, criterion, epoch, normalizer, test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    if config["task"]["type"] == "regression":
        mae_errors = AverageMeter()

    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if "label_index" in config["task"]:
            target = target[:, int(config["task"]["label_index"])].view(-1, 1)

        with torch.no_grad():
            if config["cmd"]["cuda"]:
                input_var = (
                    input[0].cuda(),
                    input[1].cuda(),
                    input[2].cuda(),
                    [crys_idx.cuda() for crys_idx in input[3]],
                )
            else:
                input_var = (input[0], input[1], input[2], input[3])
        if config["task"]["type"] == "regression":
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        with torch.no_grad():
            if config["cmd"]["cuda"]:
                target_var = target_normed.cuda()
            else:
                target_var = target_normed

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        if config["task"]["type"] == "regression":
            mae_error = eval(config["task"]["metric"])(
                normalizer.denorm(output.data.cpu()), target
            )
            losses.update(loss.item(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
            if test:
                test_pred = normalizer.denorm(output.data.cpu())
                test_target = target
                test_preds += test_pred.view(-1).tolist()
                test_targets += test_target.view(-1).tolist()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if not test:
            if config["task"]["type"] == "regression":
                log_writer.add_scalar(
                    "Validation Loss", losses.val, epoch * len(val_loader) + i
                )
                log_writer.add_scalar(
                    "Validation MAE",
                    mae_errors.val,
                    epoch * len(val_loader) + i,
                )

        if i % config["cmd"]["print_every"] == 0:
            if config["task"]["type"] == "regression":
                print(
                    "Val:   [{0}/{1}]\t\t"
                    "Loss: {loss.val:.4f} ({loss.avg:.4f})\t"
                    "MAE: {mae_errors.val:.3f} ({mae_errors.avg:.3f})\t"
                    "Fwd: {batch_time.val:.3f}s\t".format(
                        i,
                        len(val_loader),
                        batch_time=batch_time,
                        loss=losses,
                        mae_errors=mae_errors,
                    )
                )

    if config["task"]["dataset"] == "qm9":
        print(
            "MAE",
            torch.mean(torch.abs(target - output.cpu()), dim=0).data.numpy()
            * np.array(
                [
                    config["task"]["label_multipliers"][
                        config["task"]["label_index"]
                    ]
                ]
            ),
        )

    if test:
        star_label = "**"
        import csv

        with open(
            os.path.join(config["cmd"]["results_dir"], "test_results.csv"), "w"
        ) as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(
                test_cif_ids, test_targets, test_preds
            ):
                writer.writerow((cif_id, target, pred))
    else:
        star_label = "*"
    if config["task"]["type"] == "regression":
        print(
            " {star} MAE {mae_errors.avg:.3f}".format(
                star=star_label, mae_errors=mae_errors
            )
        )
        return mae_errors.avg


if __name__ == "__main__":
    main()
