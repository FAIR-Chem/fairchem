import os

import numpy as np

TIMESTAMP = "2022-04-26-12-34-08-schnet"
# TIMESTAMP = os.listdir(os.path.dirname(__file__) + '/results')[-1]
RES_FILE = "/is2re_predictions.npz"
name = os.path.dirname(__file__) + "/results/" + TIMESTAMP + RES_FILE

data = np.load(name)
lst = data.files
energy = list(data["energy"])

print("Mean energy: ", np.mean(energy))
print("Std: ", np.std(energy))
