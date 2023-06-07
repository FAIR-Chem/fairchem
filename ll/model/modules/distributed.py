from typing import Any, Literal, cast

import torch.distributed
from lightning.fabric.utilities.distributed import ReduceOp
from lightning.pytorch import LightningModule
from lightning_fabric.utilities.distributed import _distributed_available
from typing_extensions import TypeVar

from ...util.typing_utils import mixin_base_type

T = TypeVar("T", infer_variance=True)

ReduceOpStr = Literal[
    "avg",
    "mean",
    "band",
    "bor",
    "bxor",
    "max",
    "min",
    "premul_sum",
    "product",
    "sum",
]
VALID_REDUCE_OPS = (
    "avg",
    "mean",
    "band",
    "bor",
    "bxor",
    "max",
    "min",
    "premul_sum",
    "product",
    "sum",
)


class DistributedMixin(mixin_base_type(LightningModule)):
    def all_gather_object(
        self,
        object: T,
        group: torch.distributed.ProcessGroup | None = None,
    ) -> list[T]:
        if not _distributed_available():
            return [object]

        object_list = [cast(T, None) for _ in range(self.trainer.world_size)]
        torch.distributed.all_gather_object(object_list, object, group=group)
        return object_list

    def barrier(self, name: str | None = None):
        self.trainer.strategy.barrier(name=name)

    def reduce(
        self,
        tensor: torch.Tensor,
        reduce_op: ReduceOp | ReduceOpStr,
        group: Any | None = None,
    ) -> torch.Tensor:
        if isinstance(reduce_op, str):
            # validate reduce_op
            if not reduce_op in VALID_REDUCE_OPS:
                raise ValueError(
                    f"reduce_op must be one of {VALID_REDUCE_OPS}, got {reduce_op}"
                )

        return self.trainer.strategy.reduce(tensor, group=group, reduce_op=reduce_op)
