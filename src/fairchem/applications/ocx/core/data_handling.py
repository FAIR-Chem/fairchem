from __future__ import annotations

import pickle
import re

import numpy as np
import pandas as pd
from fairchem.applications.ocx.core.voltage import get_she
from fairchem.applications.ocx.core.wulff import get_normalized_facet_fracs
from scipy.spatial.distance import euclidean


def add_xrf(row, xrf_mapping):
    """
    For the row, add the xrf composition if available in the mapping.

    Args:
        row (pd.Series): The row of the experimental data
        xrf_mapping (dict): The mapping of sample ids to XRF compositions
    Returns:
        str: The XRF composition if available, otherwise None (e.g. Pt-0.5-Pd-0.5)
    """
    source = row["source"]
    sample_id = row["sample id"]

    if source == "vsp":
        xrf_comp = xrf_mapping.get(sample_id, None)
    elif source == "uoft":
        sid = re.sub(r"_rep\d+", "", sample_id)
        xrf_comp = xrf_mapping.get(sid, None)
    return xrf_comp


def get_and_apply_xrf_mapping(xrf_data, expt_data):
    """
    Given the xrf and experimental data, get the XRF mapping
    and apply it to the raw experimental data.

    Args:
        xrf_data (pd.DataFrame): The xrf data
        expt_data (pd.DataFrame): The experimental data

    Returns:
        pd.DataFrame: The experimental data with the XRF composition added
    """

    xrf_mapping = {}
    for idx, row in xrf_data.iterrows():
        source = row["source"]
        sample_id = row["sample id"]

        if source == "vsp":
            xrf_mapping[sample_id] = row["xrf-gde_formula"]
        elif source == "uoft":
            sid = re.sub(r"_rep\d+", "", sample_id)
            xrf_mapping[sid] = row["xrf-wafer_formula"]

    expt_data["xrf comp"] = expt_data.apply(add_xrf, axis=1)
    expt_data = expt_data[~expt_data["xrf comp"].isnull()]
    expt_data = expt_data[(expt_data.rep != "avg") & (expt_data.rep != "std")]
    return expt_data


def load_and_preprocess_data(
    computational_file: str,
    experimental_file: str,
    xrd_data_file: str = None,
    max_rwp: float = 40,
    min_q_score: float = 70,
    cod_to_ocp_lookup: str = None,
    interpolate_v: float = 3.3,
    reaction: str = "CO2R",
    pick_best_match: bool = True,
    filter_on_matched: bool = False,
):
    """
    Process raw experimental and xrd for analysis.

    Args:
        computational_file (str): The path to the computational data
        experimental_file (str): The path to the experimental data
        xrd_data_file (str): The path to the XRD data
        max_rwp (float): The maximum RWP value to consider
        min_q_score (float): The minimum Q score to consider
        cod_to_ocp_lookup (str): The path to the COD to OCP lookup dictionary
        interpolate_v (float): The applied potential for fixed potential results
        pick_best_match (bool): If in the case of duplicate matches, the one with
            the closest XRF composition should be used
        filter_on_matched (bool): if the returned dataframes should be filtered down
            to only matched materials
        structure_df (DataFrame): Dataframe containing structural information. Append structural
            information as part of computational features.
        close_packed_only (bool): If True, only consider close packed structures

    Returns:
        pd.DataFrame: The computational data after preprocessing
        pd.DataFrame: The experimental data after preprocessing at a fixed potential
        pd.DataFrame: The experimental data after preprocessing
    """
    # Load comp data
    df = pd.read_csv(computational_file, low_memory=False)
    # Load expt data
    df_expt = pd.read_csv(experimental_file)

    # load cod mapping
    if cod_to_ocp_lookup is not None:
        with open(cod_to_ocp_lookup, "rb") as f:
            cod_ocp_lookup = pickle.load(f)
    else:
        cod_ocp_lookup = None

    adsorbates = ["C", "CO", "CHO", "COCOH", "H", "OH"]
    df["elements"] = df.slab_comp.apply(lambda x: "".join(x.split("-")[::2]))

    df_expt.voltage = abs(df_expt.voltage)
    # to ensure consistent aggregation across HER+CO2R, sort the df to ensure
    # HER samples show up alongside their CO2R samples.
    df_expt = df_expt.sort_values(by="sample id").reset_index(drop=True)
    df_expt = df_expt[df_expt.reaction == reaction]

    # Load XRD data and make a lookup dictionary
    df_xrd = pd.read_csv(xrd_data_file)
    xrd_lookup = get_xrd_lookup_list(
        df_xrd,
        max_rwp=max_rwp,
        min_q_score=min_q_score,
        cod_ocp_lookup=cod_ocp_lookup,
    )

    # Filter on XRD data
    df_expt["bulk_id"], df_expt["rwp"], df_expt["q_score"] = zip(
        *df_expt["sample id"].apply(map_wrapper, args=(xrd_lookup,))
    )
    df_expt["matched"] = df_expt.bulk_id.apply(lambda x: type(x) == str)

    if filter_on_matched:
        df_expt = df_expt[df_expt.matched].copy()
    df_expt["computational_comp_nearest_xrf"] = df_expt["xrf comp"].apply(
        get_best_comp, args=(df,)
    )
    df_expt["sid"] = df_expt["sample id"].replace(r"_rep\d+", "", regex=True)

    agg_fns = {
        "sample id": lambda x: list(x),
        "composition": lambda x: list(x),
        "sid": lambda x: list(x),
        "rep": lambda x: list(x),
        "source": "first",
        # "post processing id": "first", #TODO: @Jehad needs to add this back into the data files
        "reaction": "first",
        "xrf comp": "first",
        "bulk_id": "first",
        "computational_comp_nearest_xrf": "first",
    }
    df_expt = df_expt.groupby(
        by=["computational_comp_nearest_xrf", "source", "current density"],
        dropna=False,
        as_index=False,
    ).agg(
        {
            **agg_fns,
            **{col: "mean" for col in df_expt.columns if col not in agg_fns},
        }
    )
    # pandas converts NaNs to None above, revert that for bulk_ids which get used later.
    df_expt["bulk_id"] = df_expt.bulk_id.fillna(value=np.nan)
    # if any of the aggregated samples were a match, consider a match
    df_expt["matched"] = df_expt.matched > 0

    df_expt["comp_identifier"] = df_expt.apply(
        lambda row: f"{row['source']}-{row['computational_comp_nearest_xrf']}", axis=1
    )

    if pick_best_match:
        df_expt = get_best_match(df_expt, df)
        if filter_on_matched:
            df_expt = df_expt[df_expt.matched].copy()

    if reaction == "CO2R":
        (
            df_expt["H2_pr"],
            df_expt["CO_pr"],
            df_expt["CH4_pr"],
            df_expt["Total_Liquid_pr"],
            df_expt["C2H4_pr"],
        ) = zip(*df_expt.apply(get_all_production_rates, axis=1))
    elif reaction == "HER":
        df_expt["voltage_she"] = df_expt.voltage.apply(get_she)
    else:
        raise NotImplementedError("Invalid reaction specified. CO2R or HER.")

    # Add mean, wulff, boltz weighted energy
    df_expt = append_energies(df_expt, df, adsorbates=adsorbates)
    df_expt["elements"] = df_expt["xrf comp"].apply(
        lambda x: "".join(x.split("-")[::2])
    )

    # Organize final df
    columns = [
        "sid",
        "sample id",
        "source",
        "batch number",
        "batch date",
        "composition",
        "voltage",
    ]
    rem_columns = [x for x in df_expt.columns if x not in columns]
    df_expt = df_expt[columns + rem_columns]
    df_expt = add_cv_categories(df_expt)

    if reaction == "HER":
        return df, df_expt, None

    # Interpolate CO2RR data to a constant potential
    constant_v_entries = []
    for identifier in df_expt["comp_identifier"].unique():
        df_temp = df_expt[(df_expt["comp_identifier"] == identifier)].copy()
        constant_v_entries.append(
            get_constant_V(
                interpolate_v,
                df_temp,
                [
                    "xrf comp",
                ],
            )
        )

    df_expt_const_v = pd.DataFrame(constant_v_entries)
    df_expt_const_v = add_cv_categories(df_expt_const_v)

    return df, df_expt_const_v, df_expt


def get_best_match(df_expt, df_comp):
    """
    Given an experimental dataframe, drop duplicate entries (by bulk_id) from the
    experimental dataframe. The experimental entry with the closest composition by XRF
    to the computational entry is retained.

    Args:
        df_expt (DataFrame): The experimental dataframe
        df_comp (DataFrame): The computational dataframe
    """

    for bulk_id in df_expt.bulk_id.unique():
        if (
            len(df_expt[df_expt.bulk_id == bulk_id]) > 1
            and bulk_id in df_comp.bulk_id.unique()
        ):
            pure_comp = df_comp[df_comp.bulk_id == bulk_id].slab_comp.values[0]
            compositions = df_expt[df_expt.bulk_id == bulk_id]["xrf comp"].values
            distances = [
                get_distance_between_formulas(val, pure_comp) for val in compositions
            ]
            min_idx = np.argmin(distances)
            indices_to_drop = df_expt[
                (df_expt.bulk_id == bulk_id)
                & (df_expt["xrf comp"] != compositions[min_idx])
            ].index
            df_expt = df_expt.drop(indices_to_drop)

    return df_expt.reset_index(drop=True)


def append_energies(df_expt, df_comp, adsorbates):
    """
    Add the mean, wulff, and boltzmann weighted energies to the experimental
    dataframe. If the bulk_id is available, then the energies are aggregated
    for that bulk. Otherwise, the energies are aggregated for the composition.
    If there are multiple bulks with the same composition, the energies are
    aggregated across all bulks of that composition.

    Args:
        df_expt (DataFrame): The experimental dataframe
        df_comp (DataFrame): The computational dataframe
        adsorbates (list): The adsorbates to consider

    Returns:
        DataFrame: The experimental dataframe with the energies appended
    """
    expt_bulk_ids = df_expt.bulk_id.unique()
    expt_bulk_comps = df_expt.computational_comp_nearest_xrf.unique()

    comp_bulk_stats = [
        compute_comp_stats(df_comp, adsorbates, bulk_id=bulk_id)
        for bulk_id in expt_bulk_ids
    ]
    comp_composition_stats = [
        compute_comp_stats(df_comp, adsorbates, comp=comp) for comp in expt_bulk_comps
    ]

    expt_bulk_id_stats = {}
    for out in comp_bulk_stats:
        expt_bulk_id_stats[out[0]] = out[1]

    expt_comp_stats = {}
    for out in comp_composition_stats:
        expt_comp_stats[out[0]] = out[1]

    def fill_comp(row, key):
        row[key] = None
        # if bulk information exists, use adsorption energy for that bulk
        row[key] = expt_bulk_id_stats[row["bulk_id"]][key]
        # otherwise, average across all bulks of that composition
        if row[key] is None or np.isnan(row[key]):
            row[key] = expt_comp_stats[row["computational_comp_nearest_xrf"]][key]
        return row

    for ads in adsorbates:
        for aggr in ["mean", "wulff", "boltz"]:
            key = f"{ads}_{aggr}_energy"
            df_expt = df_expt.apply(fill_comp, args=(key,), axis=1)

    return df_expt


def get_best_comp(comp, df):
    """
    Given a composition, return the composition in the computational
    df that is closest to the xrf composition

    Args:
        comp (str): The composition as a string (e.g. "Pt-0.5-Pd-0.5")
        df (DataFrame): The computational dataframe
    Returns:
        str: The composition in the computational df that is closest to the xrf
            composition in the same format as comp
    """
    elements = "".join(comp.split("-")[::2])
    comps = df[df.elements == elements].slab_comp.unique()
    distances = [get_distance_between_formulas(comp, val) for val in comps]
    if len(distances) == 0:
        return comp
    min_idx = np.argmin(distances)
    return comps[min_idx]


def get_distance_between_formulas(comp1: str, comp2: str):
    """
    Finds the euclidean distance between two compositions.
    update to be an argument

    Args:
        comp1 (str): The first slab comp
        comp2 (str): The second slab comp

    Returns:
        float: The euclidean distance between the compositions
    """
    stoich1 = tuple(map(float, re.findall(r"\d+.\d+", comp1)))
    stoich2 = tuple(map(float, re.findall(r"\d+.\d+", comp2)))
    return euclidean(stoich1, stoich2)


def get_boltz_energy(energy_distribution, cleavage_energies, T=300):
    """
    Given a distribution of energies, return an aggregate energy that reflects
    the probability of site occupancy. The adsorption energy is weighted by the
    the Boltzmann probability of the adsorption energy as an ensemble across all
    surfaces and the Boltzmann probability of the surface as an ensemble across
    all surface energies. This is normalized by the sum of the product of these
    two probabilities.

    Args:
        energy_distribution (list): The distribution of adsorption energies
        cleavage_energies (list): The distribution of cleavage energies
        T (float): The temperature in Kelvin

    Returns:
        float: The aggregate energy
    """
    e_min = min(energy_distribution)
    e_min_ce = min(cleavage_energies)

    def get_prob(E, E_min, T):
        kb = 8.61733326e-5
        return np.exp(-(E - E_min) / (kb * T))

    num = 0
    den = 0
    for idx, e in enumerate(energy_distribution):
        prob_e = get_prob(e, e_min, T)
        prob_ce = get_prob(cleavage_energies[idx], e_min_ce, T)
        num += prob_e * prob_ce * e
        den += prob_e * prob_ce

    return num / den


def compute_comp_stats(df, adsorbates, bulk_id=None, comp=None):
    """
    The workhorse of `append_energies` that computes the mean, wulff, and boltzmann
    energies for each adsorbate specified. If a bulk_id is provided, the energies are
    aggregated for that bulk. Otherwise, the energies are aggregated for the composition.

    Args:
        df (DataFrame): The computational dataframe
        adsorbates (list): The adsorbates to consider
        bulk_id (str): The bulk_id to consider
        comp (str): The composition to consider

    Returns:
        tuple: The bulk_id or composition and the summary statistics
    """
    summary_stats = {}

    if bulk_id:
        key = "bulk_id"
        value = bulk_id
    elif comp:
        key = "slab_comp"
        value = comp

    entries = df[df[key] == value]
    for ads in adsorbates:
        # Compute mean energies
        energies = entries[f"{ads}_min_sp_e"].dropna().values
        for stat in ["min", "mean", "median", "max", "std"]:
            if len(energies) == 0:
                mean_value = None
            else:
                mean_value = eval(f"np.{stat}")(energies)
            summary_stats[f"{ads}_{stat}_energy"] = mean_value
        # Compute Wulff energies
        wulff_df = entries.copy()
        wulff_df = get_normalized_facet_fracs(wulff_df, [f"{ads}_min_sp_e"])
        if "facet_fraction_on_wulff" not in wulff_df.columns:
            # when wulff information is not available, fall back to mean
            summary_stats[f"{ads}_wulff_energy"] = mean_value
        else:
            wulff_entries = wulff_df[(wulff_df.facet_fraction_on_wulff > 0)]
            wulff_energies = wulff_entries[f"{ads}_min_sp_e"].values
            weights = wulff_entries.facet_fraction_on_wulff.values
            if len(wulff_energies) == 0:
                # when wulff information is not available, fall back to mean
                summary_stats[f"{ads}_wulff_energy"] = mean_value
            else:
                summary_stats[f"{ads}_wulff_energy"] = eval("np.average")(
                    wulff_energies, weights=weights
                )
        # Compute Boltzmann weighted energies
        boltz_energies = entries[
            (~entries[f"{ads}_min_sp_e"].isnull())
            & (~entries["cleavage_energy"].isnull())
        ][f"{ads}_min_sp_e"].values
        cleavage_energies = entries[
            (~entries[f"{ads}_min_sp_e"].isnull())
            & (~entries["cleavage_energy"].isnull())
        ]["cleavage_energy"].values

        if len(boltz_energies) == 0:
            # when boltz information is not available, fall back to mean
            summary_stats[f"{ads}_boltz_energy"] = mean_value
        else:
            summary_stats[f"{ads}_boltz_energy"] = get_boltz_energy(
                boltz_energies, cleavage_energies
            )

    return value, summary_stats


def get_production_rate(FE: float, nel: int, current_density: float):
    """
    Converts Faradaic Efficiency to production rate.

    Args:
        FE (float): is the % faradaic efficiency of the product
        nel (int): is the number of electron transfers required to make the product
        current_density (float):  the experimental current density with units of [mA/cm**2]
    Returns:
        (float): The production rate with units mol/(cm**2 s)

    """
    faraday_const = 96485  # [(s A)/mol]
    return ((FE / 100) * (current_density / 1000)) / (nel * faraday_const)


def get_xrd_lookup_list(xrd_df, max_rwp=None, min_q_score=None, cod_ocp_lookup=None):
    """
    Take df of the xrd data and naively convert it to a lookup dictionary
    where we assume the best single phase entry is correct. Only keep a fit if
    its ranking is better than rank.

    """
    entries = xrd_df.to_dict(orient="records")
    xrd_lookup = {}
    for entry in entries:
        if type(entry["solutions_target_cifids"]) != str:
            continue
        matches_unstripped = entry["solutions_target_cifids"].split("'")
        matches = [matches_unstripped[i] for i in range(1, len(matches_unstripped), 2)]

        rwps_str = re.findall("[0-9]+[.][0-9]+", entry["solutions_target_rwp"])
        rwps = [float(rwp) for rwp in rwps_str]
        if type(entry["solutions_target_qrankings"]) is str:
            q_scores_str = re.findall(
                "[0-9]+[.][0-9]+", entry["solutions_target_qrankings"]
            )
            q_scores = [float(q_score) for q_score in q_scores_str]
            for idx, match in enumerate(matches):
                if max_rwp is None and min_q_score is None:
                    if cod_ocp_lookup is not None:
                        if (
                            match in cod_ocp_lookup
                            and cod_ocp_lookup[match] is not None
                        ):
                            xrd_lookup[entry["sample id"]] = (
                                cod_ocp_lookup[match],
                                rwps[idx],
                                q_scores[idx],
                            )
                            break
                        elif match.split("-")[0] != "cod":
                            xrd_lookup[entry["sample id"]] = (
                                match,
                                rwps[idx],
                                q_scores[idx],
                            )
                            break
                    else:
                        if match.split("-")[0] != "cod":
                            xrd_lookup[entry["sample id"]] = (
                                match,
                                rwps[idx],
                                q_scores[idx],
                            )
                            break
                else:
                    if q_scores[idx] > min_q_score and rwps[idx] < max_rwp:
                        if cod_ocp_lookup is not None:
                            if (
                                match in cod_ocp_lookup
                                and cod_ocp_lookup[match] is not None
                            ):
                                xrd_lookup[entry["sample id"]] = (
                                    cod_ocp_lookup[match],
                                    rwps[idx],
                                    q_scores[idx],
                                )
                                break
                            elif match.split("-")[0] != "cod":
                                xrd_lookup[entry["sample id"]] = (
                                    match,
                                    rwps[idx],
                                    q_scores[idx],
                                )
                                break
                        else:
                            if match.split("-")[0] != "cod":
                                xrd_lookup[entry["sample id"]] = (
                                    match,
                                    rwps[idx],
                                    q_scores[idx],
                                )
                                break
        if entry["sample id"] not in xrd_lookup:
            xrd_lookup[entry["sample id"]] = (np.nan, np.nan, np.nan)

    return xrd_lookup


def get_all_production_rates(row):
    """
    A wrapper to convert all Faradaic Efficiencies to production rates.
    To be used as a dataframe apply function that operates on each row.

    Returns:
        (tuple): The production rates for H2, CO, CH4, liquid, and C2H4 respectively.
    """
    pr_H2 = get_production_rate(row.fe_h2, 2, row["current density"])
    pr_CO = get_production_rate(row.fe_co, 2, row["current density"])
    pr_CH4 = get_production_rate(row.fe_ch4, 8, row["current density"])
    pr_liq = get_production_rate(row["fe_liquid"], 2, row["current density"])
    pr_C2H4 = get_production_rate(row.fe_c2h4, 2, row["current density"])
    return pr_H2, pr_CO, pr_CH4, pr_liq, pr_C2H4


def get_constant_V(
    V: float, df_temp: pd.DataFrame, optional_invariate_values: list = []
) -> dict:
    """
    Given a fixed applied potential, this function interpolates between the
    available data to get a pseudo-fixed results.

    For each column, values are interpolated based on the Voltage index. i.e.
    i.e. Voltage vs XXX is fit for neighboring data points, and predicted for the
    desired potential.


    This assumes a local linear region around the desired potential.

    Args:
        V (float): the desired applied potential
        df_temp (dataframe): the data for which the interpolation will be performed.

    Returns:
        dict: a single data entry at the fixed potential.
    """
    sample_entry = df_temp.to_dict(orient="records")[0]

    new_entry = {
        "composition": sample_entry["composition"],
        "batch number": sample_entry["batch number"],
        "reaction": sample_entry["reaction"],
        "source": sample_entry["source"],
        "batch date": sample_entry["batch date"],
        # "post processing id": sample_entry["post processing id"], #TODO: @Jehad needs to add this back into the data files
        "rep": sample_entry["rep"],
        "voltage": V,
        "sample id": sample_entry["sample id"],
        "bulk_id": sample_entry["bulk_id"],
        "rwp": sample_entry["rwp"],
        "q_score": sample_entry["q_score"],
        "matched": sample_entry["matched"],
        "computational_comp_nearest_xrf": sample_entry[
            "computational_comp_nearest_xrf"
        ],
        "elements": sample_entry["elements"],
    }
    for key in optional_invariate_values:
        new_entry[key] = sample_entry[key]

    df_temp = pd.concat([df_temp, pd.DataFrame([new_entry])])
    df_temp.sort_values("voltage", inplace=True)
    df_temp.set_index("voltage", inplace=True)
    for column in df_temp.columns:
        if column in [
            "fe_co",
            "fe_h2",
            "fe_ch4",
            "fe_c2h4",
            "fe_liquid",
            "CO_pr",
            "H2_pr",
            "CH4_pr",
            "Total_Liquid_pr",
            "C2H4_pr",
        ]:
            df_temp[column] = np.log(df_temp[column])
            df_temp[column] = df_temp[column].interpolate("index")
            df_temp[column] = np.exp(df_temp[column])
        else:
            df_temp[column] = df_temp[column].interpolate("index")
    df_temp = df_temp.reset_index()
    new_entry = df_temp[df_temp["voltage"] == V].to_dict(orient="records")[0]

    return new_entry


def map_wrapper(sample_id, lookup):
    """
    Needed because some experiments dont have XRD so mapping
    fails.
    """
    if sample_id in lookup:
        return lookup[sample_id]
    else:
        return (np.nan, np.nan, np.nan)


def add_cv_categories(df: pd.DataFrame):
    """
    Add cross-validation columns to be used for ML training. LOO corresponds to
    excluding any random example. LOCO corresponds to leaving out a unique
    composition - i.e. AuZn --> exclude all compositions of the form Au-X-Zn-Y
    where X,Y > 0.

    Args:
        df (dataframe): Dataframe to append LOO and LOCO cv categories.

    Returns:
        df (dataframe): Dataframe with LOO/LOCO columns appended.
    """
    unique_comps = df.elements.unique()
    categories = {comp: group for group, comp in enumerate(unique_comps)}
    df["loco_cv"] = df.elements.apply(lambda x: categories[x])
    df["loo_cv"] = df.reset_index(drop=True).index

    return df


if __name__ == "__main__":
    ###DEBUG PURPOSES
    df, df_expt_const_v, df_expt = load_and_preprocess_data(
        "/large_experiments/opencatalyst/ut_expt/release/comp_df_241022.csv",
        "/private/home/mshuaibi/projects/ocp/experimental/vertical/co2rr_vertical/expt/automated_analysis/data/ExpDataDump_241020.csv",
        "/private/home/mshuaibi/projects/ocp/experimental/vertical/co2rr_vertical/expt/automated_analysis/data/XRDDataDump-241020.csv",
        40,
        70,
        "/large_experiments/opencatalyst/ut_expt/moo/cod_matches_lookup.pkl",
        reaction="CO2R",
        filter_on_matched=False,
    )
