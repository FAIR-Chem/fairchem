from __future__ import annotations

import logging
import pickle

from submitit.core.utils import JobPaths


def add_timestamp_id_to_submission_pickle(
    slurm_folder: str, slurm_job_id: str, timestamp_id: str
):
    # Try to put the timestamp-id into the original submission pickle's config
    # so that if the node crashes, it can be pick up the correct run to resume
    #
    # we need to do this after the job has started because the timestamp-id is generated at runtime
    # instead a-priori before the submission starts (ie: if we had a db to store a global job unique job)
    submission_pickle_path = JobPaths(
        folder=slurm_folder, job_id=slurm_job_id
    ).submitted_pickle
    try:
        with open(str(submission_pickle_path), "rb") as f:
            pkl = pickle.load(f)
            # args passed to the runner function (0 is the input_dict): https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/_cli.py#L36
            pkl.args[0]["timestamp_id"] = timestamp_id
        with open(str(submission_pickle_path), "wb") as f:
            pickle.dump(pkl, f)
    except Exception as e:
        # Since this only affects the ability to resume jobs, if the pickle doesn't exist
        # or the format changed, throw a warning instead of failing the job here
        logging.warn(f"Couldn't modify the submission pickle with error: {e}")
