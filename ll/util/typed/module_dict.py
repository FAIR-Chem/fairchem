from typing import Generic, Iterable, Mapping

import torch.nn as nn
from typing_extensions import TypeVar

TModule = TypeVar("TModule", bound=nn.Module, infer_variance=True)


class TypedModuleDict(nn.Module, Generic[TModule]):
    def __init__(
        self,
        modules: Mapping[str, TModule] | None = None,
        key_prefix: str = "_typed_moduledict_",
        # we use a key prefix to avoid attribute name collisions
        # (which is a common issue in nn.ModuleDict as it uses `__setattr__` to set the modules)
    ):
        super().__init__()

        self.key_prefix = key_prefix
        self._module_dict = nn.ModuleDict(
            {self._with_prefix(k): v for k, v in modules.items()}
        )

    def _with_prefix(self, key: str) -> str:
        return f"{self.key_prefix}{key}"

    def _remove_prefix(self, key: str) -> str:
        assert key.startswith(
            self.key_prefix
        ), f"{key} does not start with {self.key_prefix}"
        return key[len(self.key_prefix) :]

    def __setitem__(self, key: str, module: TModule) -> None:
        key = self._with_prefix(key)
        return self._module_dict.__setitem__(key, module)

    def __getitem__(self, key: str) -> TModule:
        key = self._with_prefix(key)
        return self._module_dict.__getitem__(key)  # type: ignore

    def update(self, modules: Mapping[str, TModule]) -> None:
        return self._module_dict.update(
            {self._with_prefix(k): v for k, v in modules.items()}
        )

    def get(self, key: str) -> TModule | None:
        key = self._with_prefix(key)
        return self._module_dict.get(key)

    def keys(self) -> Iterable[str]:
        r"""Return an iterable of the ModuleDict keys."""
        return [self._remove_prefix(k) for k in self._module_dict.keys()]

    def items(self) -> Iterable[tuple[str, TModule]]:
        r"""Return an iterable of the ModuleDict key/value pairs."""
        return [(self._remove_prefix(k), v) for k, v in self._module_dict.items()]

    def values(self) -> Iterable[TModule]:
        r"""Return an iterable of the ModuleDict values."""
        return self._module_dict.values()
