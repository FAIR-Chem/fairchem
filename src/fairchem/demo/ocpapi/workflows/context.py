from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Generator


@contextmanager
def set_context_var(
    context_var: ContextVar,
    value: Any,
) -> Generator[None, None, None]:
    """
    Sets the input convext variable to the input value and yields control
    back to the caller. When control returns to this function, the context
    variable is reset to its original value.

    Args:
        context_var: The context variable to set.
        value: The value to assign to the variable.
    """
    token = context_var.set(value)
    try:
        yield
    finally:
        context_var.reset(token)
