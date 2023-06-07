from typing import cast

from typing_extensions import override

from ...callbacks.bad_gradients import PrintBadGradientsCallback, print_bad_gradients
from ...util.typing_utils import mixin_base_type
from ..config import BaseConfig
from .callback import CallbackModuleMixin


class FiniteChecksModuleMixin(mixin_base_type(CallbackModuleMixin)):
    def print_bad_gradients(self):
        print_bad_gradients(self)

    @override
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_callback(
            PrintBadGradientsCallback(
                enabled=lambda: cast(BaseConfig, self.config).trainer.grad_finite_checks
            )
        )
