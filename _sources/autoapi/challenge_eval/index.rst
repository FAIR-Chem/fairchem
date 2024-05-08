:py:mod:`challenge_eval`
========================

.. py:module:: challenge_eval


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   challenge_eval.is_successful
   challenge_eval.compute_valid_ml_success
   challenge_eval.get_dft_data
   challenge_eval.process_ml_data
   challenge_eval.parse_args
   challenge_eval.main



.. py:function:: is_successful(best_pred_energy, best_dft_energy, SUCCESS_THRESHOLD=0.1)

   Computes the success rate given the best predicted energy
   and the best ground truth DFT energy.

   success_parity: The standard definition for success, where ML needs to be
   within the SUCCESS_THRESHOLD, or lower, of the DFT energy.

   Returns: Bool


.. py:function:: compute_valid_ml_success(ml_data, dft_data)

   Computes validated ML success rates.
   Here, results are generated only from ML. DFT single-points are used to
   validate whether the ML energy is within 0.1eV of the DFT energy of the
   predicted structure. If valid, the ML energy is compared to the ground
   truth DFT energy, otherwise it is discarded.

   Return validated ML success rates.


.. py:function:: get_dft_data(targets)

   Organizes the released target mapping for evaluation lookup.

   Returns: Dict:
       {
          'system_id 1': {'config_id 1': dft_ads_energy, 'config_id 2': dft_ads_energy},
          'system_id 2': {'config_id 1': dft_ads_energy, 'config_id 2': dft_ads_energy},
          ...
       }


.. py:function:: process_ml_data(results_file, model, metadata, ml_dft_targets, dft_data)

   For ML systems in which no configurations made it through the physical
   constraint checks, set energies to an arbitrarily high value to ensure
   a failure case in evaluation.

   Returns: Dict:
       {
          'system_id 1': {'config_id 1': {'ml_energy': predicted energy, 'ml+dft_energy': dft energy of ML structure} ...},
          'system_id 2': {'config_id 1': {'ml_energy': predicted energy, 'ml+dft_energy': dft energy of ML structure} ...},
          ...
       }


.. py:function:: parse_args()


.. py:function:: main()

   This script takes in your prediction file (npz format)
   and the ML model name used for ML relaxations.
   Then using a mapping file, dft ground truth energy,
   and ML relaxed dft energy returns the success rate of your predictions.


