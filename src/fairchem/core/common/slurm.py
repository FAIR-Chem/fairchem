from __future__ import annotations

import logging
import os
import pickle


def add_timestamp_id_to_submission_pickle(slurm_folder, slurm_job_id, timestamp_id):
    # Try to put the timestamp-id into the original submission pickle's config so that if the node crashes, it can be pick up
    # the correct run to resume
    submission_pickle_path = os.path.join(slurm_folder, f"{slurm_job_id}_submitted.pkl")
    try:
        with open(submission_pickle_path, "rb") as f:
            pkl = pickle.load(f)
            # args passed to the runner function (0 is the input_dict): https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/_cli.py#L36
            pkl.args[0]["timestamp_id"] = timestamp_id
        with open(submission_pickle_path, "wb") as f:
            pickle.dump(pkl, f)
    except Exception as e:
        logging.warn(f"Couldn't modify the submission pickle with error: {e}")
