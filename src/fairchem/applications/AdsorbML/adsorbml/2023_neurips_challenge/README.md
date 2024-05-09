## Validating energy predictions for 2023 Open Catalyst Challenge

The `challenge_eval.py` script takes in your prediction npz file and the model used to generate the ML relaxed strcutures (gemnet-oc-2M, scn-2M, or escn-2M) and returns the success rate. More details on how to run energy predictions on ML relaxed strcutures can be found on the [challenge website](https://opencatalystproject.org/challenge.html) under the evaluation section.

1. Git clone this repository:
    ```
    git clone https://github.com/Open-Catalyst-Project/AdsorbML.git
    ```
2. Change into the 2023_neurips_challenge directory:
    ```
    cd AdsorbML/adsorbml/2023_neurips_challenge
    ```
3. Run script:
    ```
    python challenge_eval.py --model model_used_for_MLRS --results-file /path/to/predictions.npz
    ```
    The `--model` variable should be set to either `gemnet-oc-2M`, `scn-2M`, or `escn-2M` depending on which LMDB you chose.
