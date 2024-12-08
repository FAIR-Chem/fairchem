from __future__ import annotations

from copy import deepcopy

import pandas as pd
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.conversions import StrToComposition


def add_el_features(
    data: pd.DataFrame,
    composition_column="xrf comp",
):
    data_new_features = deepcopy(data[[composition_column]])

    # this should hopefully catch all the problematic cases I have seen
    data_new_features = data_new_features.replace("-", "", regex=True)
    data_new_features = StrToComposition(
        target_col_id="pymatgen__composition"
    ).featurize_dataframe(data_new_features, col_id=composition_column)

    featurizer = ElementProperty.from_preset("matminer")
    # extract only weighted average feature
    featurizer.stats = ["mean"]
    featurizer.features = [
        "X",
        "row",
        "group",
        "atomic_mass",
        "atomic_radius",
        "mendeleev_no",
    ]

    data_new_features = featurizer.featurize_dataframe(
        data_new_features, col_id="pymatgen__composition"
    )
    # removing composition columns
    data_new_features = data_new_features.drop(
        [composition_column, "pymatgen__composition"], axis=1
    )

    # removing columns with repeating values
    data_new_features = data_new_features.loc[:, data_new_features.nunique() > 1]
    # removing columns w/ any NaNs
    data_new_features = data_new_features.dropna(how="any", axis=1)

    return data.join(data_new_features)
