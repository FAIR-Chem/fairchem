import torch


class ModuleListInfo(torch.nn.ModuleList):
    def __init__(self, info_str, modules=None) -> None:
        super().__init__(modules)
        self.info_str = str(info_str)

    def __repr__(self) -> str:
        return self.info_str
