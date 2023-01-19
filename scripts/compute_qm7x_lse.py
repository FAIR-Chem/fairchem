import json
from pathlib import Path
import h5py
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction import DictVectorizer


def count_fn(y):
    return dict(zip(*np.unique(y, return_counts=True)))


if __name__ == "__main__":
    # from  SO3Krates
    # https://github.com/thorben-frank/mlff/blob/v0.1/mlff/src/data/preprocessing.py#L297
    base = Path("/network/projects/ocp/qm7x/source")
    h5_paths = sorted(base.glob("*.hdf5"))
    h5s = [h5py.File(p, "r") for p in h5_paths]
    data = [
        (h5[f"{mol}/{conf}/ePBE0+MBD"][0], h5[f"{mol}/{conf}/atNUM"][:])
        for i, h5 in enumerate(h5s)
        for mol in tqdm(h5, desc=f"Reading file {h5_paths[i].name}", leave=False)
        for conf in tqdm(h5[mol], desc=f"Molecule {mol}", leave=False)
    ]

    q = np.array([d[0] for d in data])
    max_n_atoms = max([len(d[1]) for d in data])
    z = np.array([np.pad(d[1], (0, max_n_atoms - len(d[1]))) for d in data])
    u = np.unique(z)
    idx_ = u != 0  # remove padding with 0
    lhs_counts = list(map(count_fn, z))
    v = DictVectorizer(sparse=False)
    X = v.fit_transform(lhs_counts)
    X = X[..., idx_]

    sol = np.linalg.lstsq(X, q, rcond=None)
    shifts = np.zeros(np.max(u) + 1)
    for k, v in dict(zip(u[idx_], sol[0])).items():
        shifts[k] = v

    (
        Path(__file__).resolve().parent.parent
        / "configs"
        / "models"
        / "qm7x-metadata"
        / "lse-shifts.json"
    ).write_text(json.dumps(shifts.tolist()))

    q_shifts = shifts[z].sum(-1)
