import json
import os


def main(paths, filename, data_split="val"):
    submission_file = {}

    for idx, split in enumerate(["id", "ood_ads", "ood_bulk", "ood_ads_bulk"]):
        key = "_".join([data_split, split])
        submission_file[key] = json.load(open(os.path.join(paths[idx]), "r"))

    with open(filename, "w") as f:
        json.dump(submission_file, f)


if __name__ == "__main__":
    """
    Create a submission file for evalAI. Ensure that for the task you are
    submitting for you have generated results files on each of the 4 splits -
    id, ood_ads, ood_bulk, ood_ads_bulk.

    Results file can be obtained as follows for the various tasks:

    S2EF: config["mode"] = "predict"
    IS2RE: config["mode"] = "predict"
    IS2RS: config["mode"] = "run_relaxations" and config["task"]["write_pos"] = True

    Use this script to join the 4 results files in the format evalAI expects
    submissions.
    """

    id_path = "/path/to/id/results_file"
    ood_ads_path = "/path/to/ood_ads/results_file"
    ood_bulk_path = "/path/to/ood_bulk/results_file"
    ood_ads_bulk_path = "/path/to/ood_ads_bulk/results_file"

    paths = [id_path, ood_ads_path, ood_bulk_path, ood_ads_bulk_path]

    main(paths, filename="TASKNAME_evalai_submission.json", data_split="val")
