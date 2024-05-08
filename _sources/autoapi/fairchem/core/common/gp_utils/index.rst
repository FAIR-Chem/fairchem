:py:mod:`fairchem.core.common.gp_utils`
=======================================

.. py:module:: fairchem.core.common.gp_utils

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   fairchem.core.common.gp_utils.CopyToModelParallelRegion
   fairchem.core.common.gp_utils.ReduceFromModelParallelRegion
   fairchem.core.common.gp_utils.ScatterToModelParallelRegion
   fairchem.core.common.gp_utils.GatherFromModelParallelRegion



Functions
~~~~~~~~~

.. autoapisummary::

   fairchem.core.common.gp_utils.ensure_div
   fairchem.core.common.gp_utils.divide_and_check_no_remainder
   fairchem.core.common.gp_utils.setup_gp
   fairchem.core.common.gp_utils.cleanup_gp
   fairchem.core.common.gp_utils.initialized
   fairchem.core.common.gp_utils.get_dp_group
   fairchem.core.common.gp_utils.get_gp_group
   fairchem.core.common.gp_utils.get_dp_rank
   fairchem.core.common.gp_utils.get_gp_rank
   fairchem.core.common.gp_utils.get_dp_world_size
   fairchem.core.common.gp_utils.get_gp_world_size
   fairchem.core.common.gp_utils.pad_tensor
   fairchem.core.common.gp_utils.trim_tensor
   fairchem.core.common.gp_utils._split_tensor
   fairchem.core.common.gp_utils._reduce
   fairchem.core.common.gp_utils._split
   fairchem.core.common.gp_utils._gather
   fairchem.core.common.gp_utils._gather_with_padding
   fairchem.core.common.gp_utils.copy_to_model_parallel_region
   fairchem.core.common.gp_utils.reduce_from_model_parallel_region
   fairchem.core.common.gp_utils.scatter_to_model_parallel_region
   fairchem.core.common.gp_utils.gather_from_model_parallel_region



Attributes
~~~~~~~~~~

.. autoapisummary::

   fairchem.core.common.gp_utils._GRAPH_PARALLEL_GROUP
   fairchem.core.common.gp_utils._DATA_PARALLEL_GROUP


.. py:data:: _GRAPH_PARALLEL_GROUP

   

.. py:data:: _DATA_PARALLEL_GROUP

   

.. py:function:: ensure_div(a: int, b: int) -> None


.. py:function:: divide_and_check_no_remainder(a: int, b: int) -> int


.. py:function:: setup_gp(config) -> None


.. py:function:: cleanup_gp() -> None


.. py:function:: initialized() -> bool


.. py:function:: get_dp_group()


.. py:function:: get_gp_group()


.. py:function:: get_dp_rank() -> int


.. py:function:: get_gp_rank() -> int


.. py:function:: get_dp_world_size() -> int


.. py:function:: get_gp_world_size() -> int


.. py:function:: pad_tensor(tensor: torch.Tensor, dim: int = -1, target_size: int | None = None) -> torch.Tensor


.. py:function:: trim_tensor(tensor: torch.Tensor, sizes: torch.Tensor | None = None, dim: int = 0)


.. py:function:: _split_tensor(tensor: torch.Tensor, num_parts: int, dim: int = -1, contiguous_chunks: bool = False)


.. py:function:: _reduce(ctx: Any, input: torch.Tensor) -> torch.Tensor


.. py:function:: _split(input: torch.Tensor, dim: int = -1) -> torch.Tensor


.. py:function:: _gather(input: torch.Tensor, dim: int = -1) -> torch.Tensor


.. py:function:: _gather_with_padding(input: torch.Tensor, dim: int = -1) -> torch.Tensor


.. py:class:: CopyToModelParallelRegion(*args, **kwargs)


   Bases: :py:obj:`torch.autograd.Function`

   Base class to create custom `autograd.Function`.

   To create a custom `autograd.Function`, subclass this class and implement
   the :meth:`forward` and :meth:`backward` static methods. Then, to use your custom
   op in the forward pass, call the class method ``apply``. Do not call
   :meth:`forward` directly.

   To ensure correctness and best performance, make sure you are calling the
   correct methods on ``ctx`` and validating your backward function using
   :func:`torch.autograd.gradcheck`.

   See :ref:`extending-autograd` for more details on how to use this class.

   Examples::

       >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
       >>> class Exp(Function):
       >>>     @staticmethod
       >>>     def forward(ctx, i):
       >>>         result = i.exp()
       >>>         ctx.save_for_backward(result)
       >>>         return result
       >>>
       >>>     @staticmethod
       >>>     def backward(ctx, grad_output):
       >>>         result, = ctx.saved_tensors
       >>>         return grad_output * result
       >>>
       >>> # Use it by calling the apply method:
       >>> # xdoctest: +SKIP
       >>> output = Exp.apply(input)

   .. py:method:: forward(ctx, input: torch.Tensor) -> torch.Tensor
      :staticmethod:

      Define the forward of the custom autograd Function.

      This function is to be overridden by all subclasses.
      There are two ways to define forward:

      Usage 1 (Combined forward and ctx)::

          @staticmethod
          def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
              pass

      - It must accept a context ctx as the first argument, followed by any
        number of arguments (tensors or other types).
      - See :ref:`combining-forward-context` for more details

      Usage 2 (Separate forward and ctx)::

          @staticmethod
          def forward(*args: Any, **kwargs: Any) -> Any:
              pass

          @staticmethod
          def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
              pass

      - The forward no longer accepts a ctx argument.
      - Instead, you must also override the :meth:`torch.autograd.Function.setup_context`
        staticmethod to handle setting up the ``ctx`` object.
        ``output`` is the output of the forward, ``inputs`` are a Tuple of inputs
        to the forward.
      - See :ref:`extending-autograd` for more details

      The context can be used to store arbitrary data that can be then
      retrieved during the backward pass. Tensors should not be stored
      directly on `ctx` (though this is not currently enforced for
      backward compatibility). Instead, tensors should be saved either with
      :func:`ctx.save_for_backward` if they are intended to be used in
      ``backward`` (equivalently, ``vjp``) or :func:`ctx.save_for_forward`
      if they are intended to be used for in ``jvp``.


   .. py:method:: backward(ctx, grad_output: torch.Tensor) -> torch.Tensor
      :staticmethod:

      Define a formula for differentiating the operation with backward mode automatic differentiation.

      This function is to be overridden by all subclasses.
      (Defining this function is equivalent to defining the ``vjp`` function.)

      It must accept a context :attr:`ctx` as the first argument, followed by
      as many outputs as the :func:`forward` returned (None will be passed in
      for non tensor outputs of the forward function),
      and it should return as many tensors, as there were inputs to
      :func:`forward`. Each argument is the gradient w.r.t the given output,
      and each returned value should be the gradient w.r.t. the
      corresponding input. If an input is not a Tensor or is a Tensor not
      requiring grads, you can just pass None as a gradient for that input.

      The context can be used to retrieve tensors saved during the forward
      pass. It also has an attribute :attr:`ctx.needs_input_grad` as a tuple
      of booleans representing whether each input needs gradient. E.g.,
      :func:`backward` will have ``ctx.needs_input_grad[0] = True`` if the
      first input to :func:`forward` needs gradient computed w.r.t. the
      output.



.. py:class:: ReduceFromModelParallelRegion(*args, **kwargs)


   Bases: :py:obj:`torch.autograd.Function`

   Base class to create custom `autograd.Function`.

   To create a custom `autograd.Function`, subclass this class and implement
   the :meth:`forward` and :meth:`backward` static methods. Then, to use your custom
   op in the forward pass, call the class method ``apply``. Do not call
   :meth:`forward` directly.

   To ensure correctness and best performance, make sure you are calling the
   correct methods on ``ctx`` and validating your backward function using
   :func:`torch.autograd.gradcheck`.

   See :ref:`extending-autograd` for more details on how to use this class.

   Examples::

       >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
       >>> class Exp(Function):
       >>>     @staticmethod
       >>>     def forward(ctx, i):
       >>>         result = i.exp()
       >>>         ctx.save_for_backward(result)
       >>>         return result
       >>>
       >>>     @staticmethod
       >>>     def backward(ctx, grad_output):
       >>>         result, = ctx.saved_tensors
       >>>         return grad_output * result
       >>>
       >>> # Use it by calling the apply method:
       >>> # xdoctest: +SKIP
       >>> output = Exp.apply(input)

   .. py:method:: forward(ctx, input: torch.Tensor) -> torch.Tensor
      :staticmethod:

      Define the forward of the custom autograd Function.

      This function is to be overridden by all subclasses.
      There are two ways to define forward:

      Usage 1 (Combined forward and ctx)::

          @staticmethod
          def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
              pass

      - It must accept a context ctx as the first argument, followed by any
        number of arguments (tensors or other types).
      - See :ref:`combining-forward-context` for more details

      Usage 2 (Separate forward and ctx)::

          @staticmethod
          def forward(*args: Any, **kwargs: Any) -> Any:
              pass

          @staticmethod
          def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
              pass

      - The forward no longer accepts a ctx argument.
      - Instead, you must also override the :meth:`torch.autograd.Function.setup_context`
        staticmethod to handle setting up the ``ctx`` object.
        ``output`` is the output of the forward, ``inputs`` are a Tuple of inputs
        to the forward.
      - See :ref:`extending-autograd` for more details

      The context can be used to store arbitrary data that can be then
      retrieved during the backward pass. Tensors should not be stored
      directly on `ctx` (though this is not currently enforced for
      backward compatibility). Instead, tensors should be saved either with
      :func:`ctx.save_for_backward` if they are intended to be used in
      ``backward`` (equivalently, ``vjp``) or :func:`ctx.save_for_forward`
      if they are intended to be used for in ``jvp``.


   .. py:method:: backward(ctx, grad_output: torch.Tensor) -> torch.Tensor
      :staticmethod:

      Define a formula for differentiating the operation with backward mode automatic differentiation.

      This function is to be overridden by all subclasses.
      (Defining this function is equivalent to defining the ``vjp`` function.)

      It must accept a context :attr:`ctx` as the first argument, followed by
      as many outputs as the :func:`forward` returned (None will be passed in
      for non tensor outputs of the forward function),
      and it should return as many tensors, as there were inputs to
      :func:`forward`. Each argument is the gradient w.r.t the given output,
      and each returned value should be the gradient w.r.t. the
      corresponding input. If an input is not a Tensor or is a Tensor not
      requiring grads, you can just pass None as a gradient for that input.

      The context can be used to retrieve tensors saved during the forward
      pass. It also has an attribute :attr:`ctx.needs_input_grad` as a tuple
      of booleans representing whether each input needs gradient. E.g.,
      :func:`backward` will have ``ctx.needs_input_grad[0] = True`` if the
      first input to :func:`forward` needs gradient computed w.r.t. the
      output.



.. py:class:: ScatterToModelParallelRegion(*args, **kwargs)


   Bases: :py:obj:`torch.autograd.Function`

   Base class to create custom `autograd.Function`.

   To create a custom `autograd.Function`, subclass this class and implement
   the :meth:`forward` and :meth:`backward` static methods. Then, to use your custom
   op in the forward pass, call the class method ``apply``. Do not call
   :meth:`forward` directly.

   To ensure correctness and best performance, make sure you are calling the
   correct methods on ``ctx`` and validating your backward function using
   :func:`torch.autograd.gradcheck`.

   See :ref:`extending-autograd` for more details on how to use this class.

   Examples::

       >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
       >>> class Exp(Function):
       >>>     @staticmethod
       >>>     def forward(ctx, i):
       >>>         result = i.exp()
       >>>         ctx.save_for_backward(result)
       >>>         return result
       >>>
       >>>     @staticmethod
       >>>     def backward(ctx, grad_output):
       >>>         result, = ctx.saved_tensors
       >>>         return grad_output * result
       >>>
       >>> # Use it by calling the apply method:
       >>> # xdoctest: +SKIP
       >>> output = Exp.apply(input)

   .. py:method:: forward(ctx, input: torch.Tensor, dim: int = -1) -> torch.Tensor
      :staticmethod:

      Define the forward of the custom autograd Function.

      This function is to be overridden by all subclasses.
      There are two ways to define forward:

      Usage 1 (Combined forward and ctx)::

          @staticmethod
          def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
              pass

      - It must accept a context ctx as the first argument, followed by any
        number of arguments (tensors or other types).
      - See :ref:`combining-forward-context` for more details

      Usage 2 (Separate forward and ctx)::

          @staticmethod
          def forward(*args: Any, **kwargs: Any) -> Any:
              pass

          @staticmethod
          def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
              pass

      - The forward no longer accepts a ctx argument.
      - Instead, you must also override the :meth:`torch.autograd.Function.setup_context`
        staticmethod to handle setting up the ``ctx`` object.
        ``output`` is the output of the forward, ``inputs`` are a Tuple of inputs
        to the forward.
      - See :ref:`extending-autograd` for more details

      The context can be used to store arbitrary data that can be then
      retrieved during the backward pass. Tensors should not be stored
      directly on `ctx` (though this is not currently enforced for
      backward compatibility). Instead, tensors should be saved either with
      :func:`ctx.save_for_backward` if they are intended to be used in
      ``backward`` (equivalently, ``vjp``) or :func:`ctx.save_for_forward`
      if they are intended to be used for in ``jvp``.


   .. py:method:: backward(ctx, grad_output: torch.Tensor)
      :staticmethod:

      Define a formula for differentiating the operation with backward mode automatic differentiation.

      This function is to be overridden by all subclasses.
      (Defining this function is equivalent to defining the ``vjp`` function.)

      It must accept a context :attr:`ctx` as the first argument, followed by
      as many outputs as the :func:`forward` returned (None will be passed in
      for non tensor outputs of the forward function),
      and it should return as many tensors, as there were inputs to
      :func:`forward`. Each argument is the gradient w.r.t the given output,
      and each returned value should be the gradient w.r.t. the
      corresponding input. If an input is not a Tensor or is a Tensor not
      requiring grads, you can just pass None as a gradient for that input.

      The context can be used to retrieve tensors saved during the forward
      pass. It also has an attribute :attr:`ctx.needs_input_grad` as a tuple
      of booleans representing whether each input needs gradient. E.g.,
      :func:`backward` will have ``ctx.needs_input_grad[0] = True`` if the
      first input to :func:`forward` needs gradient computed w.r.t. the
      output.



.. py:class:: GatherFromModelParallelRegion(*args, **kwargs)


   Bases: :py:obj:`torch.autograd.Function`

   Base class to create custom `autograd.Function`.

   To create a custom `autograd.Function`, subclass this class and implement
   the :meth:`forward` and :meth:`backward` static methods. Then, to use your custom
   op in the forward pass, call the class method ``apply``. Do not call
   :meth:`forward` directly.

   To ensure correctness and best performance, make sure you are calling the
   correct methods on ``ctx`` and validating your backward function using
   :func:`torch.autograd.gradcheck`.

   See :ref:`extending-autograd` for more details on how to use this class.

   Examples::

       >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
       >>> class Exp(Function):
       >>>     @staticmethod
       >>>     def forward(ctx, i):
       >>>         result = i.exp()
       >>>         ctx.save_for_backward(result)
       >>>         return result
       >>>
       >>>     @staticmethod
       >>>     def backward(ctx, grad_output):
       >>>         result, = ctx.saved_tensors
       >>>         return grad_output * result
       >>>
       >>> # Use it by calling the apply method:
       >>> # xdoctest: +SKIP
       >>> output = Exp.apply(input)

   .. py:method:: forward(ctx, input: torch.Tensor, dim: int = -1) -> torch.Tensor
      :staticmethod:

      Define the forward of the custom autograd Function.

      This function is to be overridden by all subclasses.
      There are two ways to define forward:

      Usage 1 (Combined forward and ctx)::

          @staticmethod
          def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
              pass

      - It must accept a context ctx as the first argument, followed by any
        number of arguments (tensors or other types).
      - See :ref:`combining-forward-context` for more details

      Usage 2 (Separate forward and ctx)::

          @staticmethod
          def forward(*args: Any, **kwargs: Any) -> Any:
              pass

          @staticmethod
          def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
              pass

      - The forward no longer accepts a ctx argument.
      - Instead, you must also override the :meth:`torch.autograd.Function.setup_context`
        staticmethod to handle setting up the ``ctx`` object.
        ``output`` is the output of the forward, ``inputs`` are a Tuple of inputs
        to the forward.
      - See :ref:`extending-autograd` for more details

      The context can be used to store arbitrary data that can be then
      retrieved during the backward pass. Tensors should not be stored
      directly on `ctx` (though this is not currently enforced for
      backward compatibility). Instead, tensors should be saved either with
      :func:`ctx.save_for_backward` if they are intended to be used in
      ``backward`` (equivalently, ``vjp``) or :func:`ctx.save_for_forward`
      if they are intended to be used for in ``jvp``.


   .. py:method:: backward(ctx, grad_output: torch.Tensor)
      :staticmethod:

      Define a formula for differentiating the operation with backward mode automatic differentiation.

      This function is to be overridden by all subclasses.
      (Defining this function is equivalent to defining the ``vjp`` function.)

      It must accept a context :attr:`ctx` as the first argument, followed by
      as many outputs as the :func:`forward` returned (None will be passed in
      for non tensor outputs of the forward function),
      and it should return as many tensors, as there were inputs to
      :func:`forward`. Each argument is the gradient w.r.t the given output,
      and each returned value should be the gradient w.r.t. the
      corresponding input. If an input is not a Tensor or is a Tensor not
      requiring grads, you can just pass None as a gradient for that input.

      The context can be used to retrieve tensors saved during the forward
      pass. It also has an attribute :attr:`ctx.needs_input_grad` as a tuple
      of booleans representing whether each input needs gradient. E.g.,
      :func:`backward` will have ``ctx.needs_input_grad[0] = True`` if the
      first input to :func:`forward` needs gradient computed w.r.t. the
      output.



.. py:function:: copy_to_model_parallel_region(input: torch.Tensor) -> torch.Tensor


.. py:function:: reduce_from_model_parallel_region(input: torch.Tensor) -> torch.Tensor


.. py:function:: scatter_to_model_parallel_region(input: torch.Tensor, dim: int = -1) -> torch.Tensor


.. py:function:: gather_from_model_parallel_region(input: torch.Tensor, dim: int = -1) -> torch.Tensor


