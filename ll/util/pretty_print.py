import numpy as np
import torch


class PrettyPrintMixin:
    def __repr__(self):
        base_cls_name = self.__class__.__name__
        prop_list = []
        properties = {}
        properties.update(
            {
                k: v
                for k in dir(self)
                if (attr := getattr(self.__class__, k, None)) is not None
                and isinstance(attr, property)
                and (v := getattr(self, k, None)) is not None
            }
        )
        properties.update(self.__dict__)
        for k, v in properties.items():
            if isinstance(v, (int, float, str)):
                prop_list.append(f"{k}={v}")
            elif isinstance(v, (np.ndarray, torch.Tensor)):
                numel = v.numel() if isinstance(v, torch.Tensor) else np.prod(v.shape)
                if numel == 1:
                    prop_list.append(f"{k}={v.item()}")
                else:
                    shape = list(v.shape)
                    prop_list.append(f"{k}={shape}")
            else:
                prop_list.append(f"{k}={type(v)}")
        prop_repr = ", \n".join(f"\t{p}" for p in prop_list)
        return f"{base_cls_name}(\n{prop_repr}\n)"
