import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from torch_geometric.datasets import QM9


def count_fn(y):
    return dict(zip(*np.unique(y, return_counts=True)))


if __name__ == "__main__":
    # from  SO3Krates
    # https://github.com/thorben-frank/mlff/blob/v0.1/mlff/src/data/preprocessing.py#L297
    base = Path("/network/projects/ocp/qm9")
    ds = QM9(base)

    shifts_per_attr = []

    for attr in tqdm(range(ds[0].y.shape[-1])):

        data = [(d.y[0, attr].numpy(), d.z) for d in ds]
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
        shifts_per_attr.append(shifts.tolist())

    j_dir = (
        Path(__file__).resolve().parent.parent / "configs" / "models" / "qm9-metadata"
    )
    j_dir.mkdir(parents=True, exist_ok=True)
    (j_dir / "lse-shifts-pre-attr.json").write_text(json.dumps(shifts_per_attr))
