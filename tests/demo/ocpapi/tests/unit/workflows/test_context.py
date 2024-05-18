from contextvars import ContextVar
from unittest import TestCase as UnitTestCase
from uuid import uuid4

from fairchem.demo.ocpapi.workflows.context import set_context_var


class TestContext(UnitTestCase):
    def test_set_context_var(self) -> None:
        # Set an initial value for a context var
        ctx_var = ContextVar(str(uuid4()))
        ctx_var.set("initial")

        # Update the context variable and make sure it is changed
        with set_context_var(ctx_var, "updated"):
            self.assertEqual("updated", ctx_var.get())

        # After exiting the context manager, make sure the context var was
        # reset to its original value
        self.assertEqual("initial", ctx_var.get())
