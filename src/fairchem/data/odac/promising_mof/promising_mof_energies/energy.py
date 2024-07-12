from __future__ import annotations

import matplotlib.pyploat as plt
import pandas as pd

raw_ads_energy_data = pd.read_csv("adsorption_energy.txt", header=None, sep=" ")
complete_data = pd.DataFrame(
    index=range(raw_ads_energy_data.shape[0]),
    columns=[
        "MOF",
        "defect_conc",
        "defect_index",
        "n_CO2",
        "n_H2O",
        "configuration_index",
        "ads_energy_ev",
    ],
)  # ,'LCD','PLD','metal','OMS'])
for i in range(raw_ads_energy_data.shape[0]):
    temp_split_string = raw_ads_energy_data.iloc[i, 0].split("_w_")
    temp_0_parts = temp_split_string[0].rsplit("_", 2)
    # non-defective
    if len(temp_0_parts) < 3:
        complete_data.iloc[i, 0] = temp_split_string[0]
        complete_data.iloc[i, 1] = None
        complete_data.iloc[i, 2] = None
    elif len(temp_0_parts) == 3:
        if temp_0_parts[-1] in ["0", "1", "2", "3"] and float(temp_0_parts[1]) < 0.21:
            # defective
            complete_data.iloc[i, 0] = temp_0_parts[0]
            complete_data.iloc[i, 1] = temp_0_parts[1]
            complete_data.iloc[i, 2] = temp_0_parts[2]
        else:
            complete_data.iloc[i, 0] = temp_split_string[0]
            complete_data.iloc[i, 1] = None
            complete_data.iloc[i, 2] = None

    if temp_split_string[1].split("_")[-2] == "random":
        complete_data.iloc[i, -2] = "random_" + temp_split_string[1].split("_")[-1]
    #     elif temp_split_string[1].split('_')[-1]=='new':
    #         complete_data.iloc[i,-2]='new_'+temp_split_string[1].split('_')[-1]
    else:
        complete_data.iloc[i, -2] = temp_split_string[1].split("_")[-1]

    #     if len(temp_split_string[1].split('_'))==2:
    #         if temp_split_string[1].split('_')[0]=='CO2':
    #             complete_data.iloc[i,2]=0
    #         elif temp_split_string[1].split('_')[0]=='H2O':
    #             complete_data.iloc[i,1]=0

    if (
        temp_split_string[1].rsplit("_", 1)[0] == "CO2"
        or temp_split_string[1].rsplit("_", 1)[0] == "CO2_random"
    ):
        complete_data.iloc[i, 3] = 1
        complete_data.iloc[i, 4] = 0
    elif (
        temp_split_string[1].rsplit("_", 1)[0] == "H2O"
        or temp_split_string[1].rsplit("_", 1)[0] == "H2O_random"
        or temp_split_string[1].rsplit("_", 1)[0] == "H2O_new"
    ):
        complete_data.iloc[i, 3] = 0
        complete_data.iloc[i, 4] = 1
    elif (
        temp_split_string[1].rsplit("_", 1)[0] == "CO2_H2O"
        or temp_split_string[1].rsplit("_", 1)[0] == "CO2_H2O_random"
    ):
        complete_data.iloc[i, 3] = 1
        complete_data.iloc[i, 4] = 1
    # co2+2h2o
    elif (
        temp_split_string[1].rsplit("_", 1)[0] == "CO2_2H2O"
        or temp_split_string[1].rsplit("_", 1)[0] == "CO2_2H2O_random"
    ):
        complete_data.iloc[i, 3] = 1
        complete_data.iloc[i, 4] = 2
    else:
        print(temp_split_string)

    complete_data.iloc[i, -1] = raw_ads_energy_data.iloc[i, 1]

complete_data["Name"] = raw_ads_energy_data.iloc[:, 0]


###set 2eV bound
complete_data = complete_data[
    (complete_data["ads_energy_ev"] / (complete_data["n_CO2"] + complete_data["n_H2O"]))
    < 2
].copy()


# split into pristine and defective here
complete_data_merged_pristine = complete_data[
    complete_data["defect_conc"].isnull()
].copy()
complete_data_merged_pristine = complete_data.reset_index(drop=True)
complete_data_merged_defective = complete_data[
    complete_data["defect_conc"].notnull()
].copy()
complete_data_merged_defective = complete_data.reset_index(drop=True)
complete_data_merged_defective["defective_MOF_name"] = (
    complete_data_merged_defective["MOF"]
    + "_"
    + complete_data_merged_defective["defect_conc"]
    + "_"
    + complete_data_merged_defective["defect_index"]
)


####get the lowest energy
complete_data_merged_pristine_co2 = complete_data_merged_pristine[
    (complete_data_merged_pristine["n_CO2"] == 1)
    & (complete_data_merged_pristine["n_H2O"] == 0)
].copy()
complete_data_merged_pristine_h2o = complete_data_merged_pristine[
    (complete_data_merged_pristine["n_CO2"] == 0)
    & (complete_data_merged_pristine["n_H2O"] == 1)
].copy()
complete_data_merged_pristine_co_ads = complete_data_merged_pristine[
    (complete_data_merged_pristine["n_CO2"] == 1)
    & (complete_data_merged_pristine["n_H2O"] == 1)
].copy()
complete_data_merged_pristine_co_ads_2 = complete_data_merged_pristine[
    (complete_data_merged_pristine["n_CO2"] == 1)
    & (complete_data_merged_pristine["n_H2O"] == 2)
].copy()
complete_data_merged_defective_co2 = complete_data_merged_defective[
    (complete_data_merged_defective["n_CO2"] == 1)
    & (complete_data_merged_defective["n_H2O"] == 0)
].copy()
complete_data_merged_defective_h2o = complete_data_merged_defective[
    (complete_data_merged_defective["n_CO2"] == 0)
    & (complete_data_merged_defective["n_H2O"] == 1)
].copy()
complete_data_merged_defective_co_ads = complete_data_merged_defective[
    (complete_data_merged_defective["n_CO2"] == 1)
    & (complete_data_merged_defective["n_H2O"] == 1)
].copy()
complete_data_merged_defective_co_ads_2 = complete_data_merged_defective[
    (complete_data_merged_defective["n_CO2"] == 1)
    & (complete_data_merged_defective["n_H2O"] == 2)
].copy()

lowest_energy_data_co2 = pd.DataFrame(columns=complete_data_merged_pristine_co2.columns)
for i in range(complete_data_merged_pristine_co2.shape[0]):
    current_entry = complete_data_merged_pristine_co2.iloc[i, :]
    current_MOF = complete_data_merged_pristine_co2.iloc[i, 0]
    current_n_CO2 = complete_data_merged_pristine_co2.iloc[i, 3]
    current_n_H2O = complete_data_merged_pristine_co2.iloc[i, 4]
    current_configuration_index = complete_data_merged_pristine_co2.iloc[i, 5]
    current_lowest_energy = complete_data_merged_pristine_co2.iloc[i, 6]
    current_name = complete_data_merged_pristine_co2.iloc[i, 7]
    # if this case is not included

    if lowest_energy_data_co2[
        (lowest_energy_data_co2["MOF"] == current_MOF)
        & (lowest_energy_data_co2["defect_conc"].isnull())
        & (lowest_energy_data_co2["defect_index"].isnull())
    ].empty:
        lowest_energy_data_co2 = lowest_energy_data_co2.append(current_entry)
    # if this case is already included
    else:
        # find the index of the this case's entry
        index_this_case = lowest_energy_data_co2[
            (lowest_energy_data_co2["MOF"] == current_MOF)
            & (lowest_energy_data_co2["defect_conc"].isnull())
            & (lowest_energy_data_co2["defect_index"].isnull())
        ].index[0]
        if (
            current_lowest_energy
            < lowest_energy_data_co2.loc[index_this_case, "ads_energy_ev"]
        ):
            lowest_energy_data_co2.loc[index_this_case, "ads_energy_ev"] = (
                current_lowest_energy
            )
            lowest_energy_data_co2.loc[index_this_case, "configuration_index"] = (
                current_configuration_index
            )
            lowest_energy_data_co2.loc[index_this_case, "Name"] = current_name


lowest_energy_data_h2o = pd.DataFrame(columns=complete_data_merged_pristine_h2o.columns)

for i in range(complete_data_merged_pristine_h2o.shape[0]):
    current_entry = complete_data_merged_pristine_h2o.iloc[i, :]
    current_MOF = complete_data_merged_pristine_h2o.iloc[i, 0]
    current_defect_conc = complete_data_merged_pristine_h2o.iloc[i, 1]
    current_defect_index = complete_data_merged_pristine_h2o.iloc[i, 2]
    current_n_CO2 = complete_data_merged_pristine_h2o.iloc[i, 3]
    current_n_H2O = complete_data_merged_pristine_h2o.iloc[i, 4]
    current_configuration_index = complete_data_merged_pristine_h2o.iloc[i, 5]
    current_lowest_energy = complete_data_merged_pristine_h2o.iloc[i, 6]
    current_name = complete_data_merged_pristine_h2o.iloc[i, 7]
    # if this case is not included
    # pristine structure

    if lowest_energy_data_h2o[
        (lowest_energy_data_h2o["MOF"] == current_MOF)
        & (lowest_energy_data_h2o["defect_conc"].isnull())
        & (lowest_energy_data_h2o["defect_index"].isnull())
    ].empty:
        lowest_energy_data_h2o = lowest_energy_data_h2o.append(current_entry)
    # if this case is already included
    else:
        # find the index of the this case's entry
        index_this_case = lowest_energy_data_h2o[
            (lowest_energy_data_h2o["MOF"] == current_MOF)
            & (lowest_energy_data_h2o["defect_conc"].isnull())
            & (lowest_energy_data_h2o["defect_index"].isnull())
        ].index[0]
        if (
            current_lowest_energy
            < lowest_energy_data_h2o.loc[index_this_case, "ads_energy_ev"]
        ):
            lowest_energy_data_h2o.loc[index_this_case, "ads_energy_ev"] = (
                current_lowest_energy
            )
            lowest_energy_data_h2o.loc[index_this_case, "configuration_index"] = (
                current_configuration_index
            )
            lowest_energy_data_h2o.loc[index_this_case, "Name"] = current_name

lowest_energy_data_co_ads = pd.DataFrame(
    columns=complete_data_merged_pristine_co_ads.columns
)

for i in range(complete_data_merged_pristine_co_ads.shape[0]):
    current_entry = complete_data_merged_pristine_co_ads.iloc[i, :]
    current_MOF = complete_data_merged_pristine_co_ads.iloc[i, 0]
    current_defect_conc = complete_data_merged_pristine_co_ads.iloc[i, 1]
    current_defect_index = complete_data_merged_pristine_co_ads.iloc[i, 2]
    current_n_CO2 = complete_data_merged_pristine_co_ads.iloc[i, 3]
    current_n_H2O = complete_data_merged_pristine_co_ads.iloc[i, 4]
    current_configuration_index = complete_data_merged_pristine_co_ads.iloc[i, 5]
    current_lowest_energy = complete_data_merged_pristine_co_ads.iloc[i, 6]
    current_name = complete_data_merged_pristine_co_ads.iloc[i, 7]
    # if this case is not included
    # pristine structure
    if lowest_energy_data_co_ads[
        (lowest_energy_data_co_ads["MOF"] == current_MOF)
        & (lowest_energy_data_co_ads["defect_conc"].isnull())
        & (lowest_energy_data_co_ads["defect_index"].isnull())
    ].empty:
        lowest_energy_data_co_ads = lowest_energy_data_co_ads.append(current_entry)
    # if this case is already included
    else:
        # find the index of the this case's entry
        index_this_case = lowest_energy_data_co_ads[
            (lowest_energy_data_co_ads["MOF"] == current_MOF)
            & (lowest_energy_data_co_ads["defect_conc"].isnull())
            & (lowest_energy_data_co_ads["defect_index"].isnull())
        ].index[0]
        if (
            current_lowest_energy
            < lowest_energy_data_co_ads.loc[index_this_case, "ads_energy_ev"]
        ):
            lowest_energy_data_co_ads.loc[index_this_case, "ads_energy_ev"] = (
                current_lowest_energy
            )
            lowest_energy_data_co_ads.loc[index_this_case, "configuration_index"] = (
                current_configuration_index
            )
            lowest_energy_data_co_ads.loc[index_this_case, "Name"] = current_name


lowest_energy_data_co_ads_2 = pd.DataFrame(
    columns=complete_data_merged_pristine_co_ads_2.columns
)

for i in range(complete_data_merged_pristine_co_ads_2.shape[0]):
    current_entry = complete_data_merged_pristine_co_ads_2.iloc[i, :]
    current_MOF = complete_data_merged_pristine_co_ads_2.iloc[i, 0]
    current_defect_conc = complete_data_merged_pristine_co_ads_2.iloc[i, 1]
    current_defect_index = complete_data_merged_pristine_co_ads_2.iloc[i, 2]
    current_n_CO2 = complete_data_merged_pristine_co_ads_2.iloc[i, 3]
    current_n_H2O = complete_data_merged_pristine_co_ads_2.iloc[i, 4]
    current_configuration_index = complete_data_merged_pristine_co_ads_2.iloc[i, 5]
    current_lowest_energy = complete_data_merged_pristine_co_ads_2.iloc[i, 6]
    current_name = complete_data_merged_pristine_co_ads_2.iloc[i, 7]
    # if this case is not included
    # pristine structure

    if lowest_energy_data_co_ads_2[
        (lowest_energy_data_co_ads_2["MOF"] == current_MOF)
        & (lowest_energy_data_co_ads_2["defect_conc"].isnull())
        & (lowest_energy_data_co_ads_2["defect_index"].isnull())
    ].empty:
        lowest_energy_data_co_ads_2 = lowest_energy_data_co_ads_2.append(current_entry)
    # if this case is already included
    else:
        # find the index of the this case's entry
        index_this_case = lowest_energy_data_co_ads_2[
            (lowest_energy_data_co_ads_2["MOF"] == current_MOF)
            & (lowest_energy_data_co_ads_2["defect_conc"].isnull())
            & (lowest_energy_data_co_ads_2["defect_index"].isnull())
        ].index[0]
        if (
            current_lowest_energy
            < lowest_energy_data_co_ads_2.loc[index_this_case, "ads_energy_ev"]
        ):
            lowest_energy_data_co_ads_2.loc[index_this_case, "ads_energy_ev"] = (
                current_lowest_energy
            )
            lowest_energy_data_co_ads_2.loc[index_this_case, "configuration_index"] = (
                current_configuration_index
            )
            lowest_energy_data_co_ads_2.loc[index_this_case, "Name"] = current_name


# a datafram of mof, ads_co2, ads_h2o, ads_coads, LCD, PLD, metal, OMS
adsorption_data = pd.DataFrame(
    index=range(len(complete_data_merged_pristine.MOF.unique())),
    columns=[
        "MOF",
        "ads_CO2",
        "config_CO2",
        "n_converged_CO2",
        "ads_H2O",
        "config_H2O",
        "n_converged_H2O",
        "ads_co",
        "config_co",
        "n_converged_co",
        "ads_co_2",
        "config_co_2",
        "n_converged_co_2",
    ],
)

count = 0
for mof_name in complete_data_merged_pristine.MOF.unique():
    adsorption_data.iloc[count, 0] = mof_name

    adsorption_data.loc[count, "n_converged_CO2"] = complete_data_merged_pristine[
        (complete_data_merged_pristine["MOF"] == mof_name)
        & (complete_data_merged_pristine["n_CO2"] == 1)
        & (complete_data_merged_pristine["n_H2O"] == 0)
    ].shape[0]
    adsorption_data.loc[count, "n_converged_H2O"] = complete_data_merged_pristine[
        (complete_data_merged_pristine["MOF"] == mof_name)
        & (complete_data_merged_pristine["n_CO2"] == 0)
        & (complete_data_merged_pristine["n_H2O"] == 1)
    ].shape[0]
    adsorption_data.loc[count, "n_converged_co"] = complete_data_merged_pristine[
        (complete_data_merged_pristine["MOF"] == mof_name)
        & (complete_data_merged_pristine["n_CO2"] == 1)
        & (complete_data_merged_pristine["n_H2O"] == 1)
    ].shape[0]
    adsorption_data.loc[count, "n_converged_co_2"] = complete_data_merged_pristine[
        (complete_data_merged_pristine["MOF"] == mof_name)
        & (complete_data_merged_pristine["n_CO2"] == 1)
        & (complete_data_merged_pristine["n_H2O"] == 2)
    ].shape[0]

    if not lowest_energy_data_co2[
        (lowest_energy_data_co2["MOF"] == mof_name)
        & (lowest_energy_data_co2["defect_conc"].isnull())
    ].empty:
        adsorption_data.loc[count, "ads_CO2"] = lowest_energy_data_co2[
            (lowest_energy_data_co2["MOF"] == mof_name)
            & (lowest_energy_data_co2["defect_conc"].isnull())
        ].iloc[0, 6]
        adsorption_data.loc[count, "config_CO2"] = lowest_energy_data_co2[
            (lowest_energy_data_co2["MOF"] == mof_name)
            & (lowest_energy_data_co2["defect_conc"].isnull())
        ].iloc[0, 5]
    if not lowest_energy_data_h2o[
        (lowest_energy_data_h2o["MOF"] == mof_name)
        & (lowest_energy_data_h2o["defect_conc"].isnull())
    ].empty:
        adsorption_data.loc[count, "ads_H2O"] = lowest_energy_data_h2o[
            (lowest_energy_data_h2o["MOF"] == mof_name)
            & (lowest_energy_data_h2o["defect_conc"].isnull())
        ].iloc[0, 6]
        adsorption_data.loc[count, "config_H2O"] = lowest_energy_data_h2o[
            (lowest_energy_data_h2o["MOF"] == mof_name)
            & (lowest_energy_data_h2o["defect_conc"].isnull())
        ].iloc[0, 5]
    if not lowest_energy_data_co_ads[
        (lowest_energy_data_co_ads["MOF"] == mof_name)
        & (lowest_energy_data_co_ads["defect_conc"].isnull())
    ].empty:
        adsorption_data.loc[count, "ads_co"] = lowest_energy_data_co_ads[
            (lowest_energy_data_co_ads["MOF"] == mof_name)
            & (lowest_energy_data_co_ads["defect_conc"].isnull())
        ].iloc[0, 6]
        adsorption_data.loc[count, "config_co"] = lowest_energy_data_co_ads[
            (lowest_energy_data_co_ads["MOF"] == mof_name)
            & (lowest_energy_data_co_ads["defect_conc"].isnull())
        ].iloc[0, 5]
    if not lowest_energy_data_co_ads_2[
        (lowest_energy_data_co_ads_2["MOF"] == mof_name)
        & (lowest_energy_data_co_ads_2["defect_conc"].isnull())
    ].empty:
        adsorption_data.loc[count, "ads_co_2"] = lowest_energy_data_co_ads_2[
            (lowest_energy_data_co_ads_2["MOF"] == mof_name)
            & (lowest_energy_data_co_ads_2["defect_conc"].isnull())
        ].iloc[0, 6]
        adsorption_data.loc[count, "config_co_2"] = lowest_energy_data_co_ads_2[
            (lowest_energy_data_co_ads_2["MOF"] == mof_name)
            & (lowest_energy_data_co_ads_2["defect_conc"].isnull())
        ].iloc[0, 5]


# defective structures
lowest_energy_data_co2_defective = pd.DataFrame(
    columns=complete_data_merged_defective_co2.columns
)

for i in range(complete_data_merged_defective_co2.shape[0]):
    current_entry = complete_data_merged_defective_co2.iloc[i, :]
    current_MOF = complete_data_merged_defective_co2.iloc[i, 0]
    current_defect_conc = complete_data_merged_defective_co2.iloc[i, 1]
    current_defect_index = complete_data_merged_defective_co2.iloc[i, 2]
    current_n_CO2 = complete_data_merged_defective_co2.iloc[i, 3]
    current_n_H2O = complete_data_merged_defective_co2.iloc[i, 4]
    current_configuration_index = complete_data_merged_defective_co2.iloc[i, 5]
    current_lowest_energy = complete_data_merged_defective_co2.iloc[i, 6]
    current_name = complete_data_merged_defective_co2.iloc[i, 7]
    # if this case is not included

    if lowest_energy_data_co2_defective[
        (lowest_energy_data_co2_defective["MOF"] == current_MOF)
        & (lowest_energy_data_co2_defective["defect_conc"] == current_defect_conc)
        & (lowest_energy_data_co2_defective["defect_index"] == current_defect_index)
    ].empty:
        lowest_energy_data_co2_defective = lowest_energy_data_co2_defective.append(
            current_entry
        )
    # if this case is already included
    else:
        # find the index of the this case's entry
        index_this_case = lowest_energy_data_co2_defective[
            (lowest_energy_data_co2_defective["MOF"] == current_MOF)
            & (lowest_energy_data_co2_defective["defect_conc"] == current_defect_conc)
            & (lowest_energy_data_co2_defective["defect_index"] == current_defect_index)
        ].index[0]
        if (
            current_lowest_energy
            < lowest_energy_data_co2_defective.loc[index_this_case, "ads_energy_ev"]
        ):
            lowest_energy_data_co2_defective.loc[index_this_case, "ads_energy_ev"] = (
                current_lowest_energy
            )
            lowest_energy_data_co2_defective.loc[
                index_this_case, "configuration_index"
            ] = current_configuration_index
            lowest_energy_data_co2_defective.loc[index_this_case, "Name"] = current_name


lowest_energy_data_h2o_defective = pd.DataFrame(
    columns=complete_data_merged_defective_h2o.columns
)

for i in range(complete_data_merged_defective_h2o.shape[0]):
    current_entry = complete_data_merged_defective_h2o.iloc[i, :]
    current_MOF = complete_data_merged_defective_h2o.iloc[i, 0]
    current_defect_conc = complete_data_merged_defective_h2o.iloc[i, 1]
    current_defect_index = complete_data_merged_defective_h2o.iloc[i, 2]
    current_n_CO2 = complete_data_merged_defective_h2o.iloc[i, 3]
    current_n_H2O = complete_data_merged_defective_h2o.iloc[i, 4]
    current_configuration_index = complete_data_merged_defective_h2o.iloc[i, 5]
    current_lowest_energy = complete_data_merged_defective_h2o.iloc[i, 6]
    current_name = complete_data_merged_defective_h2o.iloc[i, 7]
    # if this case is not included
    # defective structure

    if lowest_energy_data_h2o_defective[
        (lowest_energy_data_h2o_defective["MOF"] == current_MOF)
        & (lowest_energy_data_h2o_defective["defect_conc"] == current_defect_conc)
        & (lowest_energy_data_h2o_defective["defect_index"] == current_defect_index)
    ].empty:
        lowest_energy_data_h2o_defective = lowest_energy_data_h2o_defective.append(
            current_entry
        )
    # if this case is already included
    else:
        # find the index of the this case's entry
        index_this_case = lowest_energy_data_h2o_defective[
            (lowest_energy_data_h2o_defective["MOF"] == current_MOF)
            & (lowest_energy_data_h2o_defective["defect_conc"] == current_defect_conc)
            & (lowest_energy_data_h2o_defective["defect_index"] == current_defect_index)
        ].index[0]
        if (
            current_lowest_energy
            < lowest_energy_data_h2o_defective.loc[index_this_case, "ads_energy_ev"]
        ):
            lowest_energy_data_h2o_defective.loc[index_this_case, "ads_energy_ev"] = (
                current_lowest_energy
            )
            lowest_energy_data_h2o_defective.loc[
                index_this_case, "configuration_index"
            ] = current_configuration_index
            lowest_energy_data_h2o_defective.loc[index_this_case, "Name"] = current_name

lowest_energy_data_co_ads_defective = pd.DataFrame(
    columns=complete_data_merged_defective_co_ads.columns
)

for i in range(complete_data_merged_defective_co_ads.shape[0]):
    current_entry = complete_data_merged_defective_co_ads.iloc[i, :]
    current_MOF = complete_data_merged_defective_co_ads.iloc[i, 0]
    current_defect_conc = complete_data_merged_defective_co_ads.iloc[i, 1]
    current_defect_index = complete_data_merged_defective_co_ads.iloc[i, 2]
    current_n_CO2 = complete_data_merged_defective_co_ads.iloc[i, 3]
    current_n_H2O = complete_data_merged_defective_co_ads.iloc[i, 4]
    current_configuration_index = complete_data_merged_defective_co_ads.iloc[i, 5]
    current_lowest_energy = complete_data_merged_defective_co_ads.iloc[i, 6]
    current_name = complete_data_merged_defective_co_ads.iloc[i, 7]
    # if this case is not included
    # defective structure

    if lowest_energy_data_co_ads_defective[
        (lowest_energy_data_co_ads_defective["MOF"] == current_MOF)
        & (lowest_energy_data_co_ads_defective["defect_conc"] == current_defect_conc)
        & (lowest_energy_data_co_ads_defective["defect_index"] == current_defect_index)
    ].empty:
        lowest_energy_data_co_ads_defective = (
            lowest_energy_data_co_ads_defective.append(current_entry)
        )
    # if this case is already included
    else:
        # find the index of the this case's entry
        index_this_case = lowest_energy_data_co_ads_defective[
            (lowest_energy_data_co_ads_defective["MOF"] == current_MOF)
            & (
                lowest_energy_data_co_ads_defective["defect_conc"]
                == current_defect_conc
            )
            & (
                lowest_energy_data_co_ads_defective["defect_index"]
                == current_defect_index
            )
        ].index[0]
        if (
            current_lowest_energy
            < lowest_energy_data_co_ads_defective.loc[index_this_case, "ads_energy_ev"]
        ):
            lowest_energy_data_co_ads_defective.loc[
                index_this_case, "ads_energy_ev"
            ] = current_lowest_energy
            lowest_energy_data_co_ads_defective.loc[
                index_this_case, "configuration_index"
            ] = current_configuration_index
            lowest_energy_data_co_ads_defective.loc[index_this_case, "Name"] = (
                current_name
            )

lowest_energy_data_co_ads_2_defective = pd.DataFrame(
    columns=complete_data_merged_defective_co_ads_2.columns
)

for i in range(complete_data_merged_defective_co_ads_2.shape[0]):
    current_entry = complete_data_merged_defective_co_ads_2.iloc[i, :]
    current_MOF = complete_data_merged_defective_co_ads_2.iloc[i, 0]
    current_defect_conc = complete_data_merged_defective_co_ads_2.iloc[i, 1]
    current_defect_index = complete_data_merged_defective_co_ads_2.iloc[i, 2]
    current_n_CO2 = complete_data_merged_defective_co_ads_2.iloc[i, 3]
    current_n_H2O = complete_data_merged_defective_co_ads_2.iloc[i, 4]
    current_configuration_index = complete_data_merged_defective_co_ads_2.iloc[i, 5]
    current_lowest_energy = complete_data_merged_defective_co_ads_2.iloc[i, 6]
    current_name = complete_data_merged_defective_co_ads_2.iloc[i, 7]
    # if this case is not included
    # defective structure

    if lowest_energy_data_co_ads_2_defective[
        (lowest_energy_data_co_ads_2_defective["MOF"] == current_MOF)
        & (lowest_energy_data_co_ads_2_defective["defect_conc"] == current_defect_conc)
        & (
            lowest_energy_data_co_ads_2_defective["defect_index"]
            == current_defect_index
        )
    ].empty:
        lowest_energy_data_co_ads_2_defective = (
            lowest_energy_data_co_ads_2_defective.append(current_entry)
        )
    # if this case is already included
    else:
        # find the index of the this case's entry
        index_this_case = lowest_energy_data_co_ads_2_defective[
            (lowest_energy_data_co_ads_2_defective["MOF"] == current_MOF)
            & (
                lowest_energy_data_co_ads_2_defective["defect_conc"]
                == current_defect_conc
            )
            & (
                lowest_energy_data_co_ads_2_defective["defect_index"]
                == current_defect_index
            )
        ].index[0]
        if (
            current_lowest_energy
            < lowest_energy_data_co_ads_2_defective.loc[
                index_this_case, "ads_energy_ev"
            ]
        ):
            lowest_energy_data_co_ads_2_defective.loc[
                index_this_case, "ads_energy_ev"
            ] = current_lowest_energy
            lowest_energy_data_co_ads_2_defective.loc[
                index_this_case, "configuration_index"
            ] = current_configuration_index
            lowest_energy_data_co_ads_2_defective.loc[index_this_case, "Name"] = (
                current_name
            )


adsorption_data_defective = pd.DataFrame(
    index=range(
        len(
            complete_data_merged_defective.groupby(
                ["MOF", "defect_conc", "defect_index"]
            ).size()
        )
    ),
    columns=[
        "MOF",
        "defect_conc",
        "defect_index",
        "ads_CO2",
        "config_CO2",
        "n_converged_CO2",
        "ads_H2O",
        "config_H2O",
        "n_converged_H2O",
        "ads_co",
        "config_co",
        "n_converged_co",
        "ads_co_2",
        "config_co_2",
        "n_converged_co_2",
    ],
)
unique_combinations_count = complete_data_merged_defective.groupby(
    ["MOF", "defect_conc", "defect_index"]
).size()

# Convert the Series to a DataFrame
def_counts_df = unique_combinations_count.reset_index(name="counts")
adsorption_data_defective.iloc[:, :3] = def_counts_df.iloc[:, :3]

for count in range(adsorption_data_defective.shape[0]):
    mof_name = adsorption_data_defective.loc[count, "MOF"]
    current_defect_conc = adsorption_data_defective.loc[count, "defect_conc"]
    current_defect_index = adsorption_data_defective.loc[count, "defect_index"]

    # adsorption_data_defective_defective.iloc[count,0]=mof_name

    adsorption_data_defective.loc[count, "n_converged_CO2"] = (
        complete_data_merged_defective[
            (complete_data_merged_defective["MOF"] == mof_name)
            & (complete_data_merged_defective["defect_conc"] == current_defect_conc)
            & (complete_data_merged_defective["defect_index"] == current_defect_index)
            & (complete_data_merged_defective["n_CO2"] == 1)
            & (complete_data_merged_defective["n_H2O"] == 0)
        ].shape[0]
    )
    adsorption_data_defective.loc[count, "n_converged_H2O"] = (
        complete_data_merged_defective[
            (complete_data_merged_defective["MOF"] == mof_name)
            & (complete_data_merged_defective["defect_conc"] == current_defect_conc)
            & (complete_data_merged_defective["defect_index"] == current_defect_index)
            & (complete_data_merged_defective["n_CO2"] == 0)
            & (complete_data_merged_defective["n_H2O"] == 1)
        ].shape[0]
    )
    adsorption_data_defective.loc[count, "n_converged_co"] = (
        complete_data_merged_defective[
            (complete_data_merged_defective["MOF"] == mof_name)
            & (complete_data_merged_defective["defect_conc"] == current_defect_conc)
            & (complete_data_merged_defective["defect_index"] == current_defect_index)
            & (complete_data_merged_defective["n_CO2"] == 1)
            & (complete_data_merged_defective["n_H2O"] == 1)
        ].shape[0]
    )
    adsorption_data_defective.loc[count, "n_converged_co_2"] = (
        complete_data_merged_defective[
            (complete_data_merged_defective["MOF"] == mof_name)
            & (complete_data_merged_defective["defect_conc"] == current_defect_conc)
            & (complete_data_merged_defective["defect_index"] == current_defect_index)
            & (complete_data_merged_defective["n_CO2"] == 1)
            & (complete_data_merged_defective["n_H2O"] == 2)
        ].shape[0]
    )

    if not lowest_energy_data_co2_defective[
        (lowest_energy_data_co2_defective["MOF"] == mof_name)
        & (lowest_energy_data_co2_defective["defect_conc"] == current_defect_conc)
        & (lowest_energy_data_co2_defective["defect_index"] == current_defect_index)
    ].empty:
        adsorption_data_defective.loc[count, "ads_CO2"] = (
            lowest_energy_data_co2_defective[
                (lowest_energy_data_co2_defective["MOF"] == mof_name)
                & (
                    lowest_energy_data_co2_defective["defect_conc"]
                    == current_defect_conc
                )
                & (
                    lowest_energy_data_co2_defective["defect_index"]
                    == current_defect_index
                )
            ].iloc[0, 6]
        )
        adsorption_data_defective.loc[count, "config_CO2"] = (
            lowest_energy_data_co2_defective[
                (lowest_energy_data_co2_defective["MOF"] == mof_name)
                & (
                    lowest_energy_data_co2_defective["defect_conc"]
                    == current_defect_conc
                )
                & (
                    lowest_energy_data_co2_defective["defect_index"]
                    == current_defect_index
                )
            ].iloc[0, 5]
        )
    if not lowest_energy_data_h2o_defective[
        (lowest_energy_data_h2o_defective["MOF"] == mof_name)
        & (lowest_energy_data_h2o_defective["defect_conc"] == current_defect_conc)
        & (lowest_energy_data_h2o_defective["defect_index"] == current_defect_index)
    ].empty:
        adsorption_data_defective.loc[count, "ads_H2O"] = (
            lowest_energy_data_h2o_defective[
                (lowest_energy_data_h2o_defective["MOF"] == mof_name)
                & (
                    lowest_energy_data_h2o_defective["defect_conc"]
                    == current_defect_conc
                )
                & (
                    lowest_energy_data_h2o_defective["defect_index"]
                    == current_defect_index
                )
            ].iloc[0, 6]
        )
        adsorption_data_defective.loc[count, "config_H2O"] = (
            lowest_energy_data_h2o_defective[
                (lowest_energy_data_h2o_defective["MOF"] == mof_name)
                & (
                    lowest_energy_data_h2o_defective["defect_conc"]
                    == current_defect_conc
                )
                & (
                    lowest_energy_data_h2o_defective["defect_index"]
                    == current_defect_index
                )
            ].iloc[0, 5]
        )
    if not lowest_energy_data_co_ads_defective[
        (lowest_energy_data_co_ads_defective["MOF"] == mof_name)
        & (lowest_energy_data_co_ads_defective["defect_conc"] == current_defect_conc)
        & (lowest_energy_data_co_ads_defective["defect_index"] == current_defect_index)
    ].empty:
        adsorption_data_defective.loc[count, "ads_co"] = (
            lowest_energy_data_co_ads_defective[
                (lowest_energy_data_co_ads_defective["MOF"] == mof_name)
                & (
                    lowest_energy_data_co_ads_defective["defect_conc"]
                    == current_defect_conc
                )
                & (
                    lowest_energy_data_co_ads_defective["defect_index"]
                    == current_defect_index
                )
            ].iloc[0, 6]
        )
        adsorption_data_defective.loc[count, "config_co"] = (
            lowest_energy_data_co_ads_defective[
                (lowest_energy_data_co_ads_defective["MOF"] == mof_name)
                & (
                    lowest_energy_data_co_ads_defective["defect_conc"]
                    == current_defect_conc
                )
                & (
                    lowest_energy_data_co_ads_defective["defect_index"]
                    == current_defect_index
                )
            ].iloc[0, 5]
        )
    if not lowest_energy_data_co_ads_2_defective[
        (lowest_energy_data_co_ads_2_defective["MOF"] == mof_name)
        & (lowest_energy_data_co_ads_2_defective["defect_conc"] == current_defect_conc)
        & (
            lowest_energy_data_co_ads_2_defective["defect_index"]
            == current_defect_index
        )
    ].empty:
        adsorption_data_defective.loc[count, "ads_co_2"] = (
            lowest_energy_data_co_ads_2_defective[
                (lowest_energy_data_co_ads_2_defective["MOF"] == mof_name)
                & (
                    lowest_energy_data_co_ads_2_defective["defect_conc"]
                    == current_defect_conc
                )
                & (
                    lowest_energy_data_co_ads_2_defective["defect_index"]
                    == current_defect_index
                )
            ].iloc[0, 6]
        )
        adsorption_data_defective.loc[count, "config_co_2"] = (
            lowest_energy_data_co_ads_2_defective[
                (lowest_energy_data_co_ads_2_defective["MOF"] == mof_name)
                & (
                    lowest_energy_data_co_ads_2_defective["defect_conc"]
                    == current_defect_conc
                )
                & (
                    lowest_energy_data_co_ads_2_defective["defect_index"]
                    == current_defect_index
                )
            ].iloc[0, 5]
        )


# read the mofs missing DDEC charges
missing_DDEC = pd.read_csv("missing_DDEC.txt", header=None)
missing_DDEC_pristine = missing_DDEC.iloc[:98, :]
missing_DDEC_defective = missing_DDEC.iloc[98:, :]
# drop the MOFs missing DDEC
index_drop_ddec_pristine = []
for i in range(adsorption_data.shape[0]):
    if adsorption_data.iloc[i, 0] in missing_DDEC_pristine.iloc[:, 0].values:
        index_drop_ddec_pristine.append(i)
adsorption_data = adsorption_data.drop(index=index_drop_ddec_pristine)
index_drop_ddec_defective = []
for i in range(adsorption_data_defective.shape[0]):
    if (
        adsorption_data_defective.loc[i, "Name"]
        in missing_DDEC_defective.iloc[:, 0].values
    ):
        index_drop_ddec_defective.append(i)
adsorption_data_defective = adsorption_data_defective.drop(
    index=index_drop_ddec_defective
)
adsorption_data = adsorption_data.reset_index(drop=True)
adsorption_data_defective = adsorption_data_defective.reset_index(drop=True)

# promising from single adsorbate energy comparison
promising_pristine = adsorption_data[
    (adsorption_data["ads_CO2"] < -0.5)
    & (adsorption_data["ads_CO2"] < adsorption_data["ads_H2O"])
]
promising_defective = adsorption_data_defective[
    (adsorption_data_defective["ads_CO2"] < -0.5)
    & (adsorption_data_defective["ads_CO2"] < adsorption_data_defective["ads_H2O"])
]


fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].scatter(
    adsorption_data["ads_CO2"],
    adsorption_data["ads_H2O"],
    marker=".",
    color=lighten_color("C0", 0.3),
)
ax[0].scatter(
    promising_pristine["ads_CO2"],
    promising_pristine["ads_H2O"],
    marker=".",
    color=lighten_color("C0", 1.2),
)

ax[0].plot([-2, 2], [-2, 2], color="grey")
ax[0].plot([-0.5, -0.5], [-2, 2], color="grey")

ax[1].scatter(
    adsorption_data_defective["ads_CO2"],
    adsorption_data_defective["ads_H2O"],
    marker=".",
    color=lighten_color("C0", 0.3),
)
ax[1].scatter(
    promising_defective["ads_CO2"],
    promising_defective["ads_H2O"],
    marker=".",
    color=lighten_color("C0", 1.2),
)

ax[1].plot([-2, 2], [-2, 2], color="grey")
ax[1].plot([-0.5, -0.5], [-2, 2], color="grey")

ax[0].tick_params(labelsize=14)
ax[1].tick_params(labelsize=14)

ax[0].set_xlabel(r"$\mathregular{E_{ads},CO_{2}}$ / eV", fontsize=14)
ax[0].set_ylabel(r"$\mathregular{E_{ads},H_{2}O}$ / eV", fontsize=14)

ax[1].set_xlabel(r"$\mathregular{E_{ads},CO_{2}}$ / eV", fontsize=14)
ax[1].set_ylabel(r"$\mathregular{E_{ads},H_{2}O}$ / eV", fontsize=14)

ax[0].set_ylim([-2, 2])
ax[0].set_xlim([-2, 2])
ax[1].set_ylim([-2, 2])
ax[1].set_xlim([-2, 2])

ax[0].text(-1.9, 1.7, "(a) Pristine", fontsize=14)
ax[1].text(-1.9, 1.7, "(b) Defective", fontsize=14)
fig.savefig("Fig_parity.png")
