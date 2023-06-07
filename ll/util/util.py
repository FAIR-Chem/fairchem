import importlib
from typing import Callable, Dict, List

import torch


def convert_list_of_dicts(
    data_list: List[Dict[str, torch.Tensor]]
) -> Dict[str, List[torch.Tensor]]:
    return {
        k: [d[k] if torch.is_tensor(d[k]) else torch.tensor(d[k]) for d in data_list]
        for k in data_list[0].keys()
    }


def compose(*transforms: Callable):
    def composed(x):
        for transform in transforms:
            x = transform(x)
        return x

    return composed


def get_absolute_mapping(name: str):
    # in this case, the `name` should be the fully qualified name of the class
    # e.g., `ocpmodels.tasks.base_task.BaseTask`
    # we can use importlib to get the module (e.g., `ocpmodels.tasks.base_task`)
    # and then import the class (e.g., `BaseTask`)

    module_name = ".".join(name.split(".")[:-1])
    class_name = name.split(".")[-1]

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise RuntimeError(
            f"Could not import module {module_name=} for class {name=}"
        ) from e

    try:
        return getattr(module, class_name)
    except AttributeError as e:
        raise RuntimeError(
            f"Could not import class {class_name=} from module {module_name=}"
        ) from e
