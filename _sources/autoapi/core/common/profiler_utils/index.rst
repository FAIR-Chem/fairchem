core.common.profiler_utils
==========================

.. py:module:: core.common.profiler_utils


Functions
---------

.. autoapisummary::

   core.common.profiler_utils.get_default_profiler_handler
   core.common.profiler_utils.get_profile_schedule


Module Contents
---------------

.. py:function:: get_default_profiler_handler(run_id: str, output_dir: str, logger: fairchem.core.common.logger.Logger)

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


