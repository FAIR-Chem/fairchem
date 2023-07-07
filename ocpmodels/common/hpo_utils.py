"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math

from ray import tune


def tune_reporter(
    iters,
    train_metrics,
    val_metrics,
    test_metrics=None,
    metric_to_opt: str = "val_loss",
    min_max: str = "min",
) -> None:
    """
    Wrapper function for tune.report()

    Args:
        iters(dict): dict with training iteration info (e.g. steps, epochs)
        train_metrics(dict): train metrics dict
        val_metrics(dict): val metrics dict
        test_metrics(dict, optional): test metrics dict, default is None
        metric_to_opt(str, optional): str for val metric to optimize, default is val_loss
        min_max(str, optional): either "min" or "max", determines whether metric_to_opt is to be minimized or maximized, default is min

    """
    # labels metric dicts
    train = label_metric_dict(train_metrics, "train")
    val = label_metric_dict(val_metrics, "val")
    # this enables tolerance for NaNs assumes val set is used for optimization
    if math.isnan(val[metric_to_opt]):
        if min_max == "min":
            val[metric_to_opt] = 100000.0
        if min_max == "max":
            val[metric_to_opt] = 0.0
    if test_metrics:
        test = label_metric_dict(test_metrics, "test")
    else:
        test = {}
    # report results to Ray Tune
    tune.report(**iters, **train, **val, **test)


def label_metric_dict(metric_dict, split):
    new_dict = {}
    for key in metric_dict:
        new_dict["{}_{}".format(split, key)] = metric_dict[key]
    metric_dict = new_dict
    return metric_dict
