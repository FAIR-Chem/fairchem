from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from ase import neighborlist
from ase.neighborlist import natural_cutoffs


def obtain_metadata(input_dir, split):
    """
    Get the metadata provided input directory and split of data.
    Args:
                input_dir (str)   The path to the directory that has all the input files, including 'metadata.pkl'
                split (str)       Should be one of the 9 split tags, i.e. 'train', 'val_id', 'test_id',
                                  'val_ood_cat/ads/both', and 'test_ood_cat/ads/both'.

    Returns:
                metadata (tuple)  adslab properties.
                                  Ex: ('mp-126', (1,1,1), 0.025, True, '*OH', (0,0,0), 'val_ood_ads')
    """
    m = pickle.load(open(input_dir + "metadata.pkl", "rb"))
    m = m["adsorbed_bulk_metadata"]
    metadata = (*m, split)
    return metadata


def create_df(metadata_lst, df_name=None):
    """
    Create a df from metadata to used check_dataset.py file
    Args:
                metadata_lst  A list of adslab properties in tuple form, each tuple should
                              contain (mpid, miller index, shift, top, adsorbate smile string,
                              adsorption cartesion coordinates tuple, and which split the data belongs to).
                              Ex: ('mp-126', (1,1,1), 0.025, True, '*OH', (0,0,0), 'val_ood_ads')
    Returns:
                df            A pandas DataFrame
    """
    if df_name is None:
        print("You did not provide a dataframe name, we will store it as df.csv")
        df_name = "df"
    if np.shape(metadata_lst)[1] != 7:
        raise ValueError(
            "The metadata is missing a value, check to make sure you have mpid, miller index, \
                          shift, top, adsorbate smile string, adsorption site coordinates, and which split data belongs to"
        )
    df = pd.DataFrame(
        metadata_lst,
        columns=[
            "mpid",
            "miller",
            "shift",
            "top",
            "adsorbate",
            "adsorption_site",
            "tag",
        ],
    )
    df.to_csv(f"{df_name}.csv")
    return df


def adslabs_are_unique(
    df, unique_by=["mpid", "miller", "shift", "top", "adsorbate", "adsorption_site"]
):
    """
    Test if there are duplicate adslabs given a df. If the input is another
    format, convert it to df first.
    Args:
            df         A pd.DataFrame containing metadata of the adslabs being checked.
            unique_by  df column names that are used to detect duplicates. The default
                       list is the fingerprints represent a unique adslab.
    """
    assert isinstance(df, pd.DataFrame)
    unique_adslabs = df.drop_duplicates(subset=unique_by)
    if len(unique_adslabs) != len(df):
        raise ValueError("There are duplicates in the dataframe provided")


def check_commonelems(df, split1, split2, check="adsorbate"):
    """
    Given a df containing all the metadata of the calculations, check to see if there are
    any bulk or adsorbate duplicates between train and val/test_ood. The dataframe should
    have a "split_tag" column indicate which split (i.e. train, val_ood_ads, etc) a data belongs to.
    Args:
        df               A pd.DataFrame containing metadata of the adslabs being checked.
        split1, split2   two of the splits from 'train', 'val_id', 'test_id',
                         'val_ood_cat/ads/both', or 'test_ood_cat/ads/both'.
    """
    split1_df = df.loc[df.tag == split1]
    split2_df = df.loc[df.tag == split2]

    if check == "adsorbate":
        common_elems = set(split1_df.adsorbate.values) & set(split2_df.adsorbate.values)
        if len(common_elems) != 0:
            raise ValueError(f"{common_elems} are in both datasets!")
    elif check == "bulk":
        common_elems = set(split1_df.mpid.values) & set(split2_df.mpid.values)
        if len(common_elems) != 0:
            raise ValueError(f"{common_elems} are in both dataset!")


def is_adsorbate_placed_correct(adslab_input, atoms_tag):
    """
    Make sure all adsorbate atoms are connected after placement.
    False means there is at least one isolated adsorbate atom.
    It should be used after input generation but before DFT to avoid
    unneccessarily computations.
    Args:
            adslab_input        `ase.Atoms` of the structure in its initial state
            atoms_tag (list)    0=bulk, 1=surface, 2=adsorbate
    Returns:
            boolean    If there is any stand alone adsorbate atoms after placement,
                       return False.

    """
    adsorbate_idx = [idx for idx, tag in enumerate(atoms_tag) if tag == 2]
    connectivity = _get_connectivity(adslab_input[adsorbate_idx])
    return np.all(np.sum(connectivity, axis=0) != 0)


def _get_connectivity(atoms):
    """
    Generate the connectivity of an atoms obj.
    Args:
                atoms      An `ase.Atoms` object
    Returns:
                matrix     The connectivity matrix of the atoms object.
    """
    cutoff = natural_cutoffs(atoms)
    neighborList = neighborlist.NeighborList(
        cutoff, self_interaction=False, bothways=True
    )
    neighborList.update(atoms)
    matrix = neighborlist.get_connectivity_matrix(neighborList.nl).toarray()
    return matrix
