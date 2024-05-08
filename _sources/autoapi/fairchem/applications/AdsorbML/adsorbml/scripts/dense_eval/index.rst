:py:mod:`fairchem.applications.AdsorbML.adsorbml.scripts.dense_eval`
====================================================================

.. py:module:: fairchem.applications.AdsorbML.adsorbml.scripts.dense_eval

.. autoapi-nested-parse::

   AdsorbML evaluation script. This script expects the results-file to be
   organized in a very specific structure in order to evaluate successfully.

   Results are to be saved out in a dictionary pickle file, where keys are the
   `system_id` and the values are energies and compute information for a
   specified `config_id`. For each `config_id` that successfully passes the
   physical constraints defined in the manuscript, the following information must
   be provided:

       ml_energy: The ML predicted adsorption energy on that particular `config_id`.

       ml+dft_energy: The DFT adsorption energy (SP or RX) as evaluated on
       the predicted ML `config_id` structure. Do note use raw DFT energies,
       ensure these are referenced correctly. None if not available.

       scf_steps: Total number of SCF steps involved in determining the DFT
       adsorption energy on the predicted ML `config_id`. For relaxation
       methods (ML+RX), sum all SCF steps across all frames. 0 if not
       available.

       ionic_steps: Total number of ionic steps in determining the DFT
       adsorption energy on the predicted ML `config_id`. This will be 1 for
       single-point methods (ML+SP). 0 if not available.

   NOTE - It is possible that due to the required filtering of physical
   constraints, no configurations are valid for a particular `system_id`. In
   this case the  system or config id can be excluded entirely from the
   results file and will be treated as a failure point at evaluation time.

   e.g.
       {
           "6_1134_23":
               {
                   "rand11": {
                       "ml_energy": -1.234,
                       "ml+dft_energy": -1.456,
                       "scf_steps": 33,
                       "ionic_steps": 1,
                   },
                   "rand5": {
                       "ml_energy": -2.489,
                       "ml+dft_energy": -2.109,
                       "scf_steps": 16,
                       "ionic_steps": 1,
                   },
                   .
                   .
                   .
               },
           "7_6566_62" :
               {
                   "rand79": {
                       "ml_energy": -1.234,
                       "ml+dft_energy": -1.456,
                       "scf_steps": 33,
                       "ionic_steps": 1,
                   },
                   .
                   .
                   .

               },
           .
           .
           .
       }



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   fairchem.applications.AdsorbML.adsorbml.scripts.dense_eval.is_successful
   fairchem.applications.AdsorbML.adsorbml.scripts.dense_eval.compute_hybrid_success
   fairchem.applications.AdsorbML.adsorbml.scripts.dense_eval.compute_valid_ml_success
   fairchem.applications.AdsorbML.adsorbml.scripts.dense_eval.get_dft_data
   fairchem.applications.AdsorbML.adsorbml.scripts.dense_eval.get_dft_compute
   fairchem.applications.AdsorbML.adsorbml.scripts.dense_eval.filter_ml_data



Attributes
~~~~~~~~~~

.. autoapisummary::

   fairchem.applications.AdsorbML.adsorbml.scripts.dense_eval.SUCCESS_THRESHOLD
   fairchem.applications.AdsorbML.adsorbml.scripts.dense_eval.parser


.. py:data:: SUCCESS_THRESHOLD
   :value: 0.1

   

.. py:function:: is_successful(best_ml_dft_energy, best_dft_energy)

   Computes the success rate given the best ML+DFT energy and the best ground
   truth DFT energy.


   success_parity: The standard definition for success, where ML needs to be
   within the SUCCESS_THRESHOLD, or lower, of the DFT energy.

   success_much_better: A system in which the ML energy is predicted to be
   much lower (less than the SUCCESS_THRESHOLD) of the DFT energy.


.. py:function:: compute_hybrid_success(ml_data, dft_data, k)

   Computes AdsorbML success rates at varying top-k values.
   Here, results are generated for the hybrid method, where the top-k ML
   energies are used to to run DFT on the corresponding ML structures. The
   resulting energies are then compared to the ground truth DFT energies.

   Return success rates and DFT compute usage at varying k.


.. py:function:: compute_valid_ml_success(ml_data, dft_data)

   Computes validated ML success rates.
   Here, results are generated only from ML. DFT single-points are used to
   validate whether the ML energy is within 0.1eV of the DFT energy of the
   predicted structure. If valid, the ML energy is compared to the ground
   truth DFT energy, otherwise it is discarded.

   Return validated ML success rates.


.. py:function:: get_dft_data(targets)

   Organizes the released target mapping for evaluation lookup.

   oc20dense_targets.pkl:
       ['system_id 1': [('config_id 1', dft_adsorption_energy), ('config_id 2', dft_adsorption_energy)], `system_id 2]

   Returns: Dict:
       {
          'system_id 1': {'config_id 1': dft_ads_energy, 'config_id 2': dft_ads_energy},
          'system_id 2': {'config_id 1': dft_ads_energy, 'config_id 2': dft_ads_energy},
          ...
       }


.. py:function:: get_dft_compute(counts)

   Calculates the total DFT compute associated with establishing a ground
   truth using the released DFT timings: oc20dense_compute.pkl.

   Compute is measured in the total number of self-consistent steps (SC). The
   total number of ionic steps is also included for reference.


.. py:function:: filter_ml_data(ml_data, dft_data)

   For ML systems in which no configurations made it through the physical
   constraint checks, set energies to an arbitrarily high value to ensure
   a failure case in evaluation.


.. py:data:: parser

   

