Create EvalAI submission files
==============================

EvalAI expects results to be structured in a specific format for a submission to be successful. A submission must contain results from the 4 different splits - in distribution (id), out of distribution adsorbate (ood ads), out of distribution catalyst (ood cat), and out of distribution adsorbate and catalyst (ood both). Constructing the submission file for each of the above tasks is as follows:

S2EF / IS2RE
************

1. Run predictions :obj:`--mode predict` on all 4 splits, generating :obj:`predictions.json` files for each split.
2. Modify :obj:`scripts/make_evalai_json.py` with the corresponding paths of the :obj:`predictions.json` files and run to generate your final submission file :obj:`taskname_split_submission.json` (filename may be modified).
3. Upload :obj:`taskname_split_submission.json` to EvalAI.


IS2RS
*****

1. Ensure :obj:`write_pos: True` is included in your configuration file. Run relaxations :obj:`--mode run_relaxations` on all 4 splits, generating :obj:`relaxed_pos_[DEVICE #].json` files for each split.
2. For each split, if relaxations were run with multiple GPUs, combine :obj:`relaxed_pos_[DEVICE #].json` into one :obj:`relaxed_pos.json` file using :obj:`scripts/make_evalai_json.py`, otherwise skip to 3.
3. Modify :obj:`scripts/make_evalai_json.py` with the corresponding paths of the :obj:`relaxed_pos.json` files and run to generate your final submission file :obj:`taskname_split_submission.json` (filename may be modified).
4. Upload :obj:`taskname_split_submission.json` to EvalAI.
