:py:mod:`fairchem.applications.AdsorbML.adsorbml.scripts.process_mlrs`
======================================================================

.. py:module:: fairchem.applications.AdsorbML.adsorbml.scripts.process_mlrs

.. autoapi-nested-parse::

   This script processes ML relaxations and sets it up for the next step.
   - Reads final energy and structure for each relaxation
   - Filters out anomalies
   - Groups together all configurations for one adsorbate-surface system
   - Sorts configs by lowest energy first

   The following files are saved out:
   - cache_sorted_byE.pkl: dict going from the system ID (bulk, surface, adsorbate)
       to a list of configs and their relaxed structures, sorted by lowest energy first.
       This is later used by write_top_k_vasp.py.
   - anomalies_by_sid.pkl: dict going from integer sid to boolean representing
       whether it was an anomaly. Anomalies are already excluded from cache_sorted_byE.pkl
       and this file is only used for extra analyses.
   - errors_by_sid.pkl: any errors that occurred



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   fairchem.applications.AdsorbML.adsorbml.scripts.process_mlrs.parse_args
   fairchem.applications.AdsorbML.adsorbml.scripts.process_mlrs.min_diff
   fairchem.applications.AdsorbML.adsorbml.scripts.process_mlrs.process_mlrs



Attributes
~~~~~~~~~~

.. autoapisummary::

   fairchem.applications.AdsorbML.adsorbml.scripts.process_mlrs.SURFACE_CHANGE_CUTOFF_MULTIPLIER
   fairchem.applications.AdsorbML.adsorbml.scripts.process_mlrs.DESORPTION_CUTOFF_MULTIPLIER
   fairchem.applications.AdsorbML.adsorbml.scripts.process_mlrs.args


.. py:data:: SURFACE_CHANGE_CUTOFF_MULTIPLIER
   :value: 1.5

   

.. py:data:: DESORPTION_CUTOFF_MULTIPLIER
   :value: 1.5

   

.. py:function:: parse_args()


.. py:function:: min_diff(atoms_init, atoms_final)


.. py:function:: process_mlrs(arg)


.. py:data:: args

   

