import argparse
import datetime
import json
import os
import pickle
import random
import shutil
import sys
import time
import warnings
from random import sample

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

from cgcnn.data import MergeDataset, collate_pool, get_train_val_test_loader
from cgcnn.model import CrystalGraphConvNet

parser = argparse.ArgumentParser(
    description="Crystal Graph Convolutional Neural Networks"
)
parser.add_argument(
    "data_options",
    metavar="OPTIONS",
    nargs="+",
    help="dataset options, started with the path to root dir",
)
parser.add_argument(
    "--identifier",
    default="",
    help="identifier to append to checkpoint / logging directory",
)
parser.add_argument(
    "--task",
    choices=["regression", "classification"],
    default="regression",
    help="complete a regression or classification task (default: regression)",
)
parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
parser.add_argument(
    "-j",
    "--workers",
    default=0,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 0)",
)
parser.add_argument(
    "--epochs",
    default=200,
    type=int,
    metavar="N",
    help="number of total epochs to run (default: 30)",
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256)",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.01,
    type=float,
    metavar="LR",
    help="initial learning rate (default: " "0.01)",
)
parser.add_argument(
    "--lr-milestones",
    default=[100, 150],
    nargs="+",
    type=int,
    metavar="N",
    help="milestones for scheduler (default: " "[100])",
)
parser.add_argument(
    "--momentum", default=0.9, type=float, metavar="M", help="momentum"
)
parser.add_argument(
    "--weight-decay",
    "--wd",
    default=0,
    type=float,
    metavar="W",
    help="weight decay (default: 0)",
)
parser.add_argument(
    "--print-freq",
    "-p",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--train-size",
    default=14000,
    type=int,
    metavar="N",
    help="number of training data to be loaded (default none)",
)
parser.add_argument(
    "--val-size",
    default=1000,
    type=int,
    metavar="N",
    help="number of validation data to be loaded (default " "1000)",
)
parser.add_argument(
    "--test-size",
    default=1000,
    type=int,
    metavar="N",
    help="number of test data to be loaded (default 1000)",
)
parser.add_argument(
    "--optim",
    default="Adam",
    type=str,
    metavar="Adam",
    help="choose an optimizer, SGD or Adam, (default: Adam)",
)
parser.add_argument(
    "--atom-fea-len",
    default=64,
    type=int,
    metavar="N",
    help="number of hidden atom features in conv layers",
)
parser.add_argument(
    "--h-fea-len",
    default=128,
    type=int,
    metavar="N",
    help="number of hidden features after pooling",
)
parser.add_argument(
    "--n-conv", default=6, type=int, metavar="N", help="number of conv layers"
)
parser.add_argument(
    "--n-h",
    default=4,
    type=int,
    metavar="N",
    help="number of hidden layers after pooling",
)

# For reproducibility.
# Refer https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

args = parser.parse_args(sys.argv[1:])

args.cuda = not args.disable_cuda and torch.cuda.is_available()

args.timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
if args.identifier:
    args.timestamp += "-{}".format(args.identifier)
args.checkpoint_dir = os.path.join("checkpoints", args.timestamp)
args.results_dir = os.path.join("results", args.timestamp)
args.log_dir = os.path.join("logs", args.timestamp)

for arg in vars(args):
    print("{:<20}: {}".format(arg, getattr(args, arg)))
os.makedirs(args.checkpoint_dir)
os.makedirs(args.results_dir)
os.makedirs(args.log_dir)

json.dump(
    args.__dict__, open(os.path.join(args.checkpoint_dir, "config.json"), "w")
)

# Tensorboard
tf_log_writer = SummaryWriter(args.log_dir)

if args.task == "regression":
    best_mae_error = 1e10
else:
    best_mae_error = 0.0


def main():
    global args, best_mae_error

    docs = pickle.load(
        open(
            "/srv/share2/abhshkdz/data/electrocatalyst-design/2019_09_19/docs.pkl",
            "rb",
        )
    )
    random.seed(42)
    random.shuffle(docs)
    docs = [doc for doc in docs if -3 < doc["energy"] < 1.0]

    input_list = pickle.load(
        open(
            "/srv/share2/abhshkdz/data/electrocatalyst-design/2019_09_19/distance_all_docs.pkl",
            "rb",
        )
    )
    target_list = torch.Tensor(
        np.array([doc["energy"] for doc in docs], dtype=np.float32).reshape(
            -1, 1
        )
    )

    structures = input_list[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]

    dataset = MergeDataset(input_list, target_list)

    collate_fn = collate_pool
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        train_size=args.train_size,
        num_workers=args.workers,
        val_size=args.val_size,
        test_size=args.test_size,
        pin_memory=args.cuda,
        return_test=True,
    )

    # obtain target value normalizer
    if args.task == "classification":
        normalizer = Normalizer(torch.zeros(2))
        normalizer.load_state_dict({"mean": 0.0, "std": 1.0})
    else:
        sample_data_list = [
            dataset[i] for i in sample(range(len(dataset)), args.train_size)
        ]
        _, sample_target = collate_fn(sample_data_list)
        normalizer = Normalizer(sample_target)

    # build model
    model = CrystalGraphConvNet(
        orig_atom_fea_len,
        nbr_fea_len,
        atom_fea_len=args.atom_fea_len,
        n_conv=args.n_conv,
        h_fea_len=args.h_fea_len,
        n_h=args.n_h,
        classification=True if args.task == "classification" else False,
    )
    if args.cuda:
        model.cuda()

    # define loss func and optimizer
    if args.task == "classification":
        criterion = nn.NLLLoss()
    else:
        criterion = nn.L1Loss()
    if args.optim == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optim == "Adam":
        optimizer = optim.Adam(
            model.parameters(), args.lr, weight_decay=args.weight_decay
        )
    else:
        raise NameError("Only SGD or Adam is allowed as --optim")

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            best_mae_error = checkpoint["best_mae_error"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            normalizer.load_state_dict(checkpoint["normalizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    scheduler = MultiStepLR(
        optimizer, milestones=args.lr_milestones, gamma=0.1
    )

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, normalizer)

        # evaluate on validation set
        mae_error = validate(val_loader, model, criterion, epoch, normalizer)

        if mae_error != mae_error:
            print("Exit due to NaN")
            sys.exit(1)

        scheduler.step()

        # remember the best mae_eror and save checkpoint
        if args.task == "regression":
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
                "args": vars(args),
            },
            is_best,
            args.checkpoint_dir,
        )

    # test best model
    print("---------Evaluate Model on Test Set---------------")
    best_checkpoint = torch.load(
        os.path.join(args.checkpoint_dir, "model_best.pth.tar")
    )
    model.load_state_dict(best_checkpoint["state_dict"])
    validate(test_loader, model, criterion, epoch, normalizer, test=True)


def train(train_loader, model, criterion, optimizer, epoch, normalizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    if args.task == "regression":
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

        if args.cuda:
            input_var = (
                Variable(input[0].cuda()),
                Variable(input[1].cuda()),
                input[2].cuda(),
                [crys_idx.cuda() for crys_idx in input[3]],
            )
        else:
            input_var = (
                Variable(input[0]),
                Variable(input[1]),
                input[2],
                input[3],
            )
        # normalize target
        if args.task == "regression":
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        if args.cuda:
            target_var = Variable(target_normed.cuda())
        else:
            target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        if args.task == "regression":
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu()[0], target.size(0))
            mae_errors.update(mae_error, target.size(0))
        else:
            accuracy, precision, recall, fscore, auc_score = class_eval(
                output.data.cpu(), target
            )
            losses.update(loss.data.cpu()[0], target.size(0))
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

        if args.task == "regression":
            tf_log_writer.add_scalar(
                "Training Loss", losses.val, epoch * len(train_loader) + i
            )
            tf_log_writer.add_scalar(
                "Training MAE", mae_errors.val, epoch * len(train_loader) + i
            )
        else:
            tf_log_writer.add_scalar(
                "Training Loss", losses.val, epoch * len(train_loader) + i
            )
            tf_log_writer.add_scalar(
                "Training Accuracy",
                accuracies.val,
                epoch * len(train_loader) + i,
            )
            tf_log_writer.add_scalar(
                "Training Precision",
                precisions.val,
                epoch * len(train_loader) + i,
            )
            tf_log_writer.add_scalar(
                "Training Recall", recalls.val, epoch * len(train_loader) + i
            )
            tf_log_writer.add_scalar(
                "Training F1", fscores.val, epoch * len(train_loader) + i
            )
            tf_log_writer.add_scalar(
                "Training AUC", auc_scores.val, epoch * len(train_loader) + i
            )

        tf_log_writer.add_scalar(
            "Learning rate",
            optimizer.param_groups[0]["lr"],
            epoch * len(train_loader) + i,
        )

        if i % args.print_freq == 0:
            if args.task == "regression":
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})".format(
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
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accu {accu.val:.3f} ({accu.avg:.3f})\t"
                    "Precision {prec.val:.3f} ({prec.avg:.3f})\t"
                    "Recall {recall.val:.3f} ({recall.avg:.3f})\t"
                    "F1 {f1.val:.3f} ({f1.avg:.3f})\t"
                    "AUC {auc.val:.3f} ({auc.avg:.3f})".format(
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
    if args.task == "regression":
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()
    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if args.cuda:
            input_var = (
                Variable(input[0].cuda(), volatile=True),
                Variable(input[1].cuda(), volatile=True),
                input[2].cuda(),
                [crys_idx.cuda() for crys_idx in input[3]],
            )
        else:
            input_var = (
                Variable(input[0], volatile=True),
                Variable(input[1], volatile=True),
                input[2],
                input[3],
            )
        if args.task == "regression":
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        if args.cuda:
            target_var = Variable(target_normed.cuda(), volatile=True)
        else:
            target_var = Variable(target_normed, volatile=True)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        if args.task == "regression":
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu()[0], target.size(0))
            mae_errors.update(mae_error, target.size(0))
            if test:
                test_pred = normalizer.denorm(output.data.cpu())
                test_target = target
                test_preds += test_pred.view(-1).tolist()
                test_targets += test_target.view(-1).tolist()
        else:
            accuracy, precision, recall, fscore, auc_score = class_eval(
                output.data.cpu(), target
            )
            losses.update(loss.data.cpu()[0], target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))
            if test:
                test_pred = torch.exp(output.data.cpu())
                test_target = target
                assert test_pred.shape[1] == 2
                test_preds += test_pred[:, 1].tolist()
                test_targets += test_target.view(-1).tolist()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if not test:
            if args.task == "regression":
                tf_log_writer.add_scalar(
                    "Validation Loss", losses.val, epoch * len(val_loader) + i
                )
                tf_log_writer.add_scalar(
                    "Validation MAE",
                    mae_errors.val,
                    epoch * len(val_loader) + i,
                )
            else:
                tf_log_writer.add_scalar(
                    "Validation Loss", losses.val, epoch * len(val_loader) + i
                )
                tf_log_writer.add_scalar(
                    "Validation Accuracy",
                    accuracies.val,
                    epoch * len(val_loader) + i,
                )
                tf_log_writer.add_scalar(
                    "Validation Precision",
                    precisions.val,
                    epoch * len(val_loader) + i,
                )
                tf_log_writer.add_scalar(
                    "Validation Recall",
                    recalls.val,
                    epoch * len(val_loader) + i,
                )
                tf_log_writer.add_scalar(
                    "Validation F1", fscores.val, epoch * len(val_loader) + i
                )
                tf_log_writer.add_scalar(
                    "Validation AUC",
                    auc_scores.val,
                    epoch * len(val_loader) + i,
                )

        if i % args.print_freq == 0:
            if args.task == "regression":
                print(
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})".format(
                        i,
                        len(val_loader),
                        batch_time=batch_time,
                        loss=losses,
                        mae_errors=mae_errors,
                    )
                )
            else:
                print(
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accu {accu.val:.3f} ({accu.avg:.3f})\t"
                    "Precision {prec.val:.3f} ({prec.avg:.3f})\t"
                    "Recall {recall.val:.3f} ({recall.avg:.3f})\t"
                    "F1 {f1.val:.3f} ({f1.avg:.3f})\t"
                    "AUC {auc.val:.3f} ({auc.avg:.3f})".format(
                        i,
                        len(val_loader),
                        batch_time=batch_time,
                        loss=losses,
                        accu=accuracies,
                        prec=precisions,
                        recall=recalls,
                        f1=fscores,
                        auc=auc_scores,
                    )
                )

    if test:
        star_label = "**"
        import csv

        with open(
            os.path.join(args.results_dir, "test_results.csv"), "w"
        ) as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(
                test_cif_ids, test_targets, test_preds
            ):
                writer.writerow((cif_id, target, pred))
    else:
        star_label = "*"
    if args.task == "regression":
        print(
            " {star} MAE {mae_errors.avg:.3f}".format(
                star=star_label, mae_errors=mae_errors
            )
        )
        return mae_errors.avg
    else:
        print(
            " {star} AUC {auc.avg:.3f}".format(star=star_label, auc=auc_scores)
        )
        return auc_scores.avg


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average="binary"
        )
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, checkpoint_dir="checkpoints/"):
    filename = os.path.join(checkpoint_dir, "checkpoint.pth.tar")
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(
            filename, os.path.join(checkpoint_dir, "model_best.pth.tar")
        )


def adjust_learning_rate(optimizer, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = args.lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


if __name__ == "__main__":
    main()
