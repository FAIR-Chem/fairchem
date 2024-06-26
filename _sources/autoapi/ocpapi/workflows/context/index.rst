ocpapi.workflows.context
========================

.. py:module:: ocpapi.workflows.context


Functions
---------

.. autoapisummary::

   ocpapi.workflows.context.set_context_var


Module Contents
---------------

.. py:function:: set_context_var(context_var: contextvars.ContextVar, value: Any) -> Generator[None, None, None]

   Sets the input convext variable to the input value and yields control
   back to the caller. When control returns to this function, the context
   variable is reset to its original value.

   :param context_var: The context variable to set.
   :param value: The value to assign to the variable.


