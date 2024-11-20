from __future__ import annotations

import numpy as np
import pandas as pd


def get_normalized_facet_fracs(comp_df: pd.DataFrame, descriptors: list):
    """
    Checks that all of the necessary descriptor data is available and ensures
    that the sum over all facet fractions where information is available
    (including information lost over not having facets up to miller 3) is 1.

    Args:
        comp_df: The computational dataframe which should include a column called
            "facet_fraction_on_wulff_not_normalized"
        descriptors: the list of descriptor strings (i.e. `C_min_sp_e`) which will
            be used in the fitting process

    Returns:
        pd.DataFrame: The modified dataframe with the normalized facet fractions
            on wulff.
    """
    comp_df = comp_df[~comp_df.facet_fraction_on_wulff_not_normalized.isnull()]
    comp_df["facet_fraction_on_wulff"] = comp_df[
        "facet_fraction_on_wulff_not_normalized"
    ].copy()
    df_shell = []
    for bulk_id in comp_df.bulk_id.unique():
        temp_df = comp_df[comp_df.bulk_id == bulk_id].copy()
        entries = temp_df.to_dict("records")
        for entry in entries:
            if any([np.isnan(entry[k]) for k in descriptors]):
                entry["facet_fraction_on_wulff"] = 0

        sum_weights = sum([entry["facet_fraction_on_wulff"] for entry in entries])

        for entry in entries:
            if sum_weights > 0:
                entry["facet_fraction_on_wulff"] /= sum_weights
            else:
                entry["facet_fraction_on_wulff"] = np.nan

        df_shell.extend(entries)
    return pd.DataFrame(df_shell)
