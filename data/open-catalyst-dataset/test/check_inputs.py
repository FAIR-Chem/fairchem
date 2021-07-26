import numpy as np
import pandas as pd

def create_df(metadata, df_name=None):
    """
    Create a df from metadata to used check_dataset.py file
    Args:
                metadata      A list of adslab/slab properties in tupple form, each tuple should
                              contain (mpid, miller index, shift, top, adsorbate smile string,
                              adsorption cartesion coordinate tuple, and which split data belongs to).
                              Ex: ('mp-126', (1,1,1), '0.025', True, '*OH', (0,0,0), 'val_ood_ads')
    Returns:
                df            A pandas DataFrame
    """
    if df_name is None:
        print("You did not provide a dataframe name, we will store it as df.csv")
        df_name = 'df'
    if np.shape(metadata)[1] != 7:
        raise ValueError("The metadata is missing a value, check to make sure you have mpid, miller index, \
                          shift, top, adsorbate smile string, adsorption site coordinates, and which split data belongs to")
    df = pd.DataFrame(metadata, columns=["mpid", "miller", "shift", "top",
                                         "adsorbate", "adsorption_site", "tag"])
    df.to_csv('{}.csv'.format(df_name))
    return df

def adslabs_are_unique(df, unique_by=["mpid", "miller", "shift", "top",
                                      "adsorbate", "adsorption_site"]):
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

def check_commonelems(df, split1, split2, check='adsorbate'):
    """
    Given a df containing all the metadata of the calculations, check to see if there are
    any bulk or adsorbate duplicates between train and val/test_ood. The dataframe should
    have a "split_tag" column indicate which split (i.e. train, val_odd_ads, etc) a data belongs to.
    Args:
        df               A pd.DataFrame containing metadata of the adslabs being checked.
        split1, split2   two of the splits from (train, test/val_id, test/val_ood_ads,
                         test/val_ood_cat, test_val/ood_both)
    """
    split1_df = df.loc[df.tag==split1]
    split2_df = df.loc[df.tag==split2]

    if check == "adsorbate":
        common_elems = set(split1_df.adsorbate.values)&set(split2_df.adsorbate.values)
        if len(common_elems) != 0:
            raise ValueError("{} are in both datasets!".format(common_elems))
    elif check == "bulk":
        common_elems = set(split1_df.mpid.values)&set(split2_df.mpid.values)
        if len(common_elems) != 0:
            raise ValueError("{} are in both dataset!".format(split))