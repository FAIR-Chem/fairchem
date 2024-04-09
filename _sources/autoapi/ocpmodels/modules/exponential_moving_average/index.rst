:py:mod:`ocpmodels.modules.exponential_moving_average`
======================================================

.. py:module:: ocpmodels.modules.exponential_moving_average

.. autoapi-nested-parse::

   Copied (and improved) from:
   https://github.com/fadel/pytorch_ema/blob/master/torch_ema/ema.py (MIT license)



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   ocpmodels.modules.exponential_moving_average.ExponentialMovingAverage




.. py:class:: ExponentialMovingAverage(parameters: Iterable[torch.nn.Parameter], decay: float, use_num_updates: bool = False)


   Maintains (exponential) moving average of a set of parameters.

   :param parameters: Iterable of `torch.nn.Parameter` (typically from
                      `model.parameters()`).
   :param decay: The exponential decay.
   :param use_num_updates: Whether to use number of updates when computing
                           averages.

   .. py:method:: _get_parameters(parameters: Optional[Iterable[torch.nn.Parameter]]) -> Iterable[torch.nn.Parameter]


   .. py:method:: update(parameters: Optional[Iterable[torch.nn.Parameter]] = None) -> None

      Update currently maintained parameters.

      Call this every time the parameters are updated, such as the result of
      the `optimizer.step()` call.

      :param parameters: Iterable of `torch.nn.Parameter`; usually the same set of
                         parameters used to initialize this object. If `None`, the
                         parameters with which this `ExponentialMovingAverage` was
                         initialized will be used.


   .. py:method:: copy_to(parameters: Optional[Iterable[torch.nn.Parameter]] = None) -> None

      Copy current parameters into given collection of parameters.

      :param parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                         updated with the stored moving averages. If `None`, the
                         parameters with which this `ExponentialMovingAverage` was
                         initialized will be used.


   .. py:method:: store(parameters: Optional[Iterable[torch.nn.Parameter]] = None) -> None

      Save the current parameters for restoring later.

      :param parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                         temporarily stored. If `None`, the parameters of with which this
                         `ExponentialMovingAverage` was initialized will be used.


   .. py:method:: restore(parameters: Optional[Iterable[torch.nn.Parameter]] = None) -> None

      Restore the parameters stored with the `store` method.
      Useful to validate the model with EMA parameters without affecting the
      original optimization process. Store the parameters before the
      `copy_to` method. After validation (or model saving), use this to
      restore the former parameters.

      :param parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                         updated with the stored parameters. If `None`, the
                         parameters with which this `ExponentialMovingAverage` was
                         initialized will be used.


   .. py:method:: state_dict() -> dict

      Returns the state of the ExponentialMovingAverage as a dict.


   .. py:method:: load_state_dict(state_dict: dict) -> None

      Loads the ExponentialMovingAverage state.

      :param state_dict: EMA state. Should be an object returned
                         from a call to :meth:`state_dict`.
      :type state_dict: dict



