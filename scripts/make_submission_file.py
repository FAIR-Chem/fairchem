"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import json
import os

import numpy as np


def main(paths, filename):
    submission_file = {}

    for idx, split in enumerate(["id", "ood_ads", "ood_cat", "ood_both"]):
        res = np.load(paths[idx], allow_pickle=True)
        contents = res.files
        for i in contents:
            key = "_".join([split, i])
            submission_file[key] = res[i]

    np.savez(filename, **submission_file)


if __name__ == "__main__":
    """
    Create a submission file for evalAI. Ensure that for the task you are
    submitting for you have generated results files on each of the 4 splits -
    id, ood_ads, ood_cat, ood_both.

    Results file can be obtained as follows for the various tasks:

    S2EF: config["mode"] = "predict"
    IS2RE: config["mode"] = "predict"
    IS2RS: config["mode"] = "run-relaxations" and config["task"]["write_pos"] = True

    Use this script to join the 4 results files in the format evalAI expects
    submissions.
    """

    id_path = "/path/to/id/results_file"
    ood_ads_path = "/path/to/ood_ads/results_file"
    ood_cat_path = "/path/to/ood_cat/results_file"
    ood_both_path = "/path/to/ood_both/results_file"

    paths = [id_path, ood_ads_path, ood_cat_path, ood_both_path]

    main(paths, filename="TASKNAME_evalai_submission.npz")
