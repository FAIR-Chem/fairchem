:py:mod:`fairchem.data.oc.tests.old_tests.check_inputs`
=======================================================

.. py:module:: fairchem.data.oc.tests.old_tests.check_inputs


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   fairchem.data.oc.tests.old_tests.check_inputs.obtain_metadata
   fairchem.data.oc.tests.old_tests.check_inputs.create_df
   fairchem.data.oc.tests.old_tests.check_inputs.adslabs_are_unique
   fairchem.data.oc.tests.old_tests.check_inputs.check_commonelems
   fairchem.data.oc.tests.old_tests.check_inputs.is_adsorbate_placed_correct
   fairchem.data.oc.tests.old_tests.check_inputs._get_connectivity



.. py:function:: obtain_metadata(input_dir, split)

   Get the metadata provided input directory and split of data.
   :param input_dir:
   :type input_dir: str
   :param split: 'val_ood_cat/ads/both', and 'test_ood_cat/ads/both'.
   :type split: str

   :returns:

             metadata (tuple)  adslab properties.
                               Ex: ('mp-126', (1,1,1), 0.025, True, '*OH', (0,0,0), 'val_ood_ads')


.. py:function:: create_df(metadata_lst, df_name=None)

   Create a df from metadata to used check_dataset.py file
   :param metadata_lst  A list of adslab properties in tuple form: contain (mpid, miller index, shift, top, adsorbate smile string,
                                                                   adsorption cartesion coordinates tuple, and which split the data belongs to).
                                                                   Ex: ('mp-126', (1,1,1), 0.025, True, '*OH', (0,0,0), 'val_ood_ads')
   :param each tuple should: contain (mpid, miller index, shift, top, adsorbate smile string,
                             adsorption cartesion coordinates tuple, and which split the data belongs to).
                             Ex: ('mp-126', (1,1,1), 0.025, True, '*OH', (0,0,0), 'val_ood_ads')

   :returns: df            A pandas DataFrame


.. py:function:: adslabs_are_unique(df, unique_by=['mpid', 'miller', 'shift', 'top', 'adsorbate', 'adsorption_site'])

   Test if there are duplicate adslabs given a df. If the input is another
   format, convert it to df first.
   :param df         A pd.DataFrame containing metadata of the adslabs being checked.:
   :param unique_by  df column names that are used to detect duplicates. The default: list is the fingerprints represent a unique adslab.


.. py:function:: check_commonelems(df, split1, split2, check='adsorbate')

   Given a df containing all the metadata of the calculations, check to see if there are
   any bulk or adsorbate duplicates between train and val/test_ood. The dataframe should
   have a "split_tag" column indicate which split (i.e. train, val_ood_ads, etc) a data belongs to.
   :param df               A pd.DataFrame containing metadata of the adslabs being checked.:
   :param split1: 'val_ood_cat/ads/both', or 'test_ood_cat/ads/both'.
   :param split2   two of the splits from 'train': 'val_ood_cat/ads/both', or 'test_ood_cat/ads/both'.
   :param 'val_id': 'val_ood_cat/ads/both', or 'test_ood_cat/ads/both'.
   :param 'test_id': 'val_ood_cat/ads/both', or 'test_ood_cat/ads/both'.
   :param : 'val_ood_cat/ads/both', or 'test_ood_cat/ads/both'.


.. py:function:: is_adsorbate_placed_correct(adslab_input, atoms_tag)

   Make sure all adsorbate atoms are connected after placement.
   False means there is at least one isolated adsorbate atom.
   It should be used after input generation but before DFT to avoid
   unneccessarily computations.
   :param adslab_input        `ase.Atoms` of the structure in its initial state:
   :param atoms_tag:
   :type atoms_tag: list

   :returns:

             boolean    If there is any stand alone adsorbate atoms after placement,
                        return False.


.. py:function:: _get_connectivity(atoms)

   Generate the connectivity of an atoms obj.
   :param atoms      An `ase.Atoms` object:

   :returns: matrix     The connectivity matrix of the atoms object.


