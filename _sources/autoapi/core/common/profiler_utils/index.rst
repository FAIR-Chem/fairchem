core.common.profiler_utils
==========================

.. py:module:: core.common.profiler_utils


Classes
-------

.. autoapisummary::

   core.common.profiler_utils.ProfilerCallback


Functions
---------

.. autoapisummary::

   core.common.profiler_utils.get_default_profiler_handler
   core.common.profiler_utils.get_profile_schedule


Module Contents
---------------

.. py:function:: get_default_profiler_handler(run_id: str, output_dir: str, logger: fairchem.core.common.logger.Logger, all_ranks: bool = False)

   Get a standard callback handle for the pytorch profiler


.. py:function:: get_profile_schedule(wait: int = 5, warmup: int = 5, active: int = 2)

   Get a profile schedule and total number of steps to run
   check pytorch docs on the meaning of these paramters:
   https://pytorch.org/docs/stable/profiler.html#torch.profiler.schedule
   Example usage:
   ```
   trace_handler = get_default_profiler_handler(run_id = self.config["cmd"]["timestamp_id"],
                                                   output_dir = self.config["cmd"]["results_dir"],
                                                   logger = self.logger)
   profile_schedule, total_profile_steps = get_profile_schedule()

   with profile(
       activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
       schedule=profile_schedule,
       on_trace_ready=trace_handler
   ) as p:
       for i in steps:
           <code block to profile>
           if i < total_profile_steps:
               p.step()


.. py:class:: ProfilerCallback(job_config: omegaconf.DictConfig, wait_steps: int = 5, warmup_steps: int = 5, active_steps: int = 2, all_ranks: bool = False, activities: tuple = (ProfilerActivity.CPU, ProfilerActivity.CUDA))

   Bases: :py:obj:`torchtnt.framework.callback.Callback`


   A Callback is an optional extension that can be used to supplement your loop with additional functionality. Good candidates
   for such logic are ones that can be re-used across units. Callbacks are generally not intended for modeling code; this should go
   in your `Unit <https://www.internalfb.com/intern/staticdocs/torchtnt/framework/unit.html>`_. To write your own callback,
   subclass the Callback class and add your own code into the hooks.

   Below is an example of a basic callback which prints a message at various points during execution.

   .. code-block:: python

     from torchtnt.framework.callback import Callback
     from torchtnt.framework.state import State
     from torchtnt.framework.unit import TEvalUnit, TPredictUnit, TTrainUnit

     class PrintingCallback(Callback):
         def on_train_start(self, state: State, unit: TTrainUnit) -> None:
             print("Starting training")

         def on_train_end(self, state: State, unit: TTrainUnit) -> None:
             print("Ending training")

         def on_eval_start(self, state: State, unit: TEvalUnit) -> None:
             print("Starting evaluation")

         def on_eval_end(self, state: State, unit: TEvalUnit) -> None:
             print("Ending evaluation")

         def on_predict_start(self, state: State, unit: TPredictUnit) -> None:
             print("Starting prediction")

         def on_predict_end(self, state: State, unit: TPredictUnit) -> None:
             print("Ending prediction")

   To use a callback, instantiate the class and pass it in the ``callbacks`` parameter to the :py:func:`~torchtnt.framework.train`, :py:func:`~torchtnt.framework.evaluate`,
   :py:func:`~torchtnt.framework.predict`, or :py:func:`~torchtnt.framework.fit` entry point.

   .. code-block:: python

     printing_callback = PrintingCallback()
     train(train_unit, train_dataloader, callbacks=[printing_callback])


   .. py:attribute:: profiler


   .. py:method:: on_train_start(state: torchtnt.framework.State, unit: torchtnt.framework.TTrainUnit) -> None

      Hook called before training starts.



   .. py:method:: on_train_step_end(state: torchtnt.framework.State, unit: torchtnt.framework.TTrainUnit) -> None

      Hook called after a train step ends.



