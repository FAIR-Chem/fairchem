from collections import abc
from logging import getLogger
from typing import Any, Callable, Iterable, cast, final

from lightning.pytorch import Callback, LightningModule
from lightning.pytorch.callbacks import LambdaCallback
from typing_extensions import override

from ...util.typing_utils import mixin_base_type

log = getLogger(__name__)

CallbackFn = Callable[[], Callback | Iterable[Callback] | None]


class CallbackRegistrarModuleMixin:
    @override
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._ll_callbacks: list[CallbackFn] = []

    def register_callback(
        self,
        callback: Callback | Iterable[Callback] | CallbackFn | None = None,
        *,
        setup: Callable | None = None,
        teardown: Callable | None = None,
        on_fit_start: Callable | None = None,
        on_fit_end: Callable | None = None,
        on_sanity_check_start: Callable | None = None,
        on_sanity_check_end: Callable | None = None,
        on_train_batch_start: Callable | None = None,
        on_train_batch_end: Callable | None = None,
        on_train_epoch_start: Callable | None = None,
        on_train_epoch_end: Callable | None = None,
        on_validation_epoch_start: Callable | None = None,
        on_validation_epoch_end: Callable | None = None,
        on_test_epoch_start: Callable | None = None,
        on_test_epoch_end: Callable | None = None,
        on_validation_batch_start: Callable | None = None,
        on_validation_batch_end: Callable | None = None,
        on_test_batch_start: Callable | None = None,
        on_test_batch_end: Callable | None = None,
        on_train_start: Callable | None = None,
        on_train_end: Callable | None = None,
        on_validation_start: Callable | None = None,
        on_validation_end: Callable | None = None,
        on_test_start: Callable | None = None,
        on_test_end: Callable | None = None,
        on_exception: Callable | None = None,
        on_save_checkpoint: Callable | None = None,
        on_load_checkpoint: Callable | None = None,
        on_before_backward: Callable | None = None,
        on_after_backward: Callable | None = None,
        on_before_optimizer_step: Callable | None = None,
        on_before_zero_grad: Callable | None = None,
        on_predict_start: Callable | None = None,
        on_predict_end: Callable | None = None,
        on_predict_batch_start: Callable | None = None,
        on_predict_batch_end: Callable | None = None,
        on_predict_epoch_start: Callable | None = None,
        on_predict_epoch_end: Callable | None = None,
    ):
        if callback is None:
            callback = LambdaCallback(
                setup=setup,
                teardown=teardown,
                on_fit_start=on_fit_start,
                on_fit_end=on_fit_end,
                on_sanity_check_start=on_sanity_check_start,
                on_sanity_check_end=on_sanity_check_end,
                on_train_batch_start=on_train_batch_start,
                on_train_batch_end=on_train_batch_end,
                on_train_epoch_start=on_train_epoch_start,
                on_train_epoch_end=on_train_epoch_end,
                on_validation_epoch_start=on_validation_epoch_start,
                on_validation_epoch_end=on_validation_epoch_end,
                on_test_epoch_start=on_test_epoch_start,
                on_test_epoch_end=on_test_epoch_end,
                on_validation_batch_start=on_validation_batch_start,
                on_validation_batch_end=on_validation_batch_end,
                on_test_batch_start=on_test_batch_start,
                on_test_batch_end=on_test_batch_end,
                on_train_start=on_train_start,
                on_train_end=on_train_end,
                on_validation_start=on_validation_start,
                on_validation_end=on_validation_end,
                on_test_start=on_test_start,
                on_test_end=on_test_end,
                on_exception=on_exception,
                on_save_checkpoint=on_save_checkpoint,
                on_load_checkpoint=on_load_checkpoint,
                on_before_backward=on_before_backward,
                on_after_backward=on_after_backward,
                on_before_optimizer_step=on_before_optimizer_step,
                on_before_zero_grad=on_before_zero_grad,
                on_predict_start=on_predict_start,
                on_predict_end=on_predict_end,
                on_predict_batch_start=on_predict_batch_start,
                on_predict_batch_end=on_predict_batch_end,
                on_predict_epoch_start=on_predict_epoch_start,
                on_predict_epoch_end=on_predict_epoch_end,
            )

        if not callable(callback):
            callback_ = cast(CallbackFn, lambda: callback)
        else:
            callback_ = callback

        self._ll_callbacks.append(callback_)


class CallbackModuleMixin(
    CallbackRegistrarModuleMixin, mixin_base_type(LightningModule)
):
    def _gather_all_callbacks(self):
        modules: list[Any] = []
        if isinstance(self, CallbackRegistrarModuleMixin):
            modules.append(self)
        if (
            datamodule := getattr(self.trainer, "datamodule", None)
        ) is not None and isinstance(datamodule, CallbackRegistrarModuleMixin):
            modules.append(datamodule)
        modules.extend(
            module
            for module in self.children()
            if isinstance(module, CallbackRegistrarModuleMixin)
        )
        for module in modules:
            yield from module._ll_callbacks

    @final
    @override
    def configure_callbacks(self):
        callbacks = super().configure_callbacks()
        if not isinstance(callbacks, abc.Sequence):
            callbacks = [callbacks]

        callbacks = list(callbacks)
        for callback_fn in self._gather_all_callbacks():
            callback_result = callback_fn()
            if callback_result is None:
                continue

            if not isinstance(callback_result, abc.Iterable):
                callback_result = [callback_result]

            for callback in callback_result:
                log.info(
                    f"Registering {callback.__class__.__qualname__} callback {callback}"
                )
                callbacks.append(callback)

        return callbacks
