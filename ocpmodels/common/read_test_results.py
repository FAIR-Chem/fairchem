import os
from pathlib import Path

import numpy as np
from minydra import resolved_args

args = resolved_args()

timestamp = args.timestamp or "2022-04-27-16-58-40-schnet"
# timestamp = os.listdir(os.path.dirname(__file__) + '/results')[-1]
res_file = args.res_file or "is2re_predictions.npz"
name = Path(__file__).resolve().parent / "results" / timestamp / res_file

data = np.load(name)
lst = data.files
energy = list(data["energy"])

print("Mean energy: ", np.mean(energy))
print("Std: ", np.std(energy))
