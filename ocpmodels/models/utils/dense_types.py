from typing import List, NamedTuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data as TorchGeoData

atom_list = [
    1,
    5,
    6,
    7,
    8,
    11,
    13,
    14,
    15,
    16,
    17,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    55,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
]
unk_idx = len(atom_list) + 1
atom_mapper = torch.full((128,), unk_idx)
for idx, atom in enumerate(atom_list):
    atom_mapper[atom] = idx + 1  # reserve 0 for paddin


cell_offsets = torch.tensor(
    [
        [-1, -1, 0],
        [-1, 0, 0],
        [-1, 1, 0],
        [0, -1, 0],
        [0, 1, 0],
        [1, -1, 0],
        [1, 0, 0],
        [1, 1, 0],
    ],
).float()
n_cells = cell_offsets.size(0)
cutoff = 8.0
filter_by_tag = True


class Data(NamedTuple):
    pos: torch.Tensor
    atoms: torch.Tensor
    tags: torch.Tensor
    real_mask: torch.Tensor
    deltapos: torch.Tensor
    y_relaxed: torch.Tensor
    fixed: torch.Tensor
    natoms: torch.Tensor

    def to(self, device):
        return Data(
            pos=self.pos.to(device),
            atoms=self.atoms.to(device),
            tags=self.tags.to(device),
            real_mask=self.real_mask.to(device),
            deltapos=self.deltapos.to(device),
            y_relaxed=self.y_relaxed.to(device),
            fixed=self.fixed.to(device),
            natoms=self.natoms.to(device),
        )

    @classmethod
    def from_torch_geometric_data(cls, data: TorchGeoData):
        pos = data.pos
        pos_relaxed = data.pos_relaxed
        cell = data.cell
        atoms = data.atomic_numbers.long()
        tags = data.tags.long()
        fixed = data.fixed.long()

        global atom_mapper, cell_offsets, n_cells, cutoff, filter_by_tag
        atoms = atom_mapper[atoms]
        offsets = torch.matmul(cell_offsets, cell).view(n_cells, 1, 3)
        expand_pos = (pos.unsqueeze(0).expand(n_cells, -1, -1) + offsets).view(
            -1, 3
        )
        expand_pos_relaxed = (
            pos.unsqueeze(0).expand(n_cells, -1, -1) + offsets
        ).view(-1, 3)
        src_pos = pos[tags > 1] if filter_by_tag else pos

        dist: torch.Tensor = (
            src_pos.unsqueeze(1) - expand_pos.unsqueeze(0)
        ).norm(dim=-1)
        used_mask = (dist < cutoff).any(dim=0) & tags.ne(2).repeat(
            n_cells
        )  # not copy ads
        used_expand_pos = expand_pos[used_mask]
        used_expand_pos_relaxed = expand_pos_relaxed[used_mask]

        used_expand_tags = tags.repeat(n_cells)[used_mask]
        used_expand_fixed = fixed.repeat(n_cells)[used_mask]
        return cls(
            pos=torch.cat([pos, used_expand_pos], dim=0),
            atoms=torch.cat([atoms, atoms.repeat(n_cells)[used_mask]]),
            tags=torch.cat([tags, used_expand_tags]),
            real_mask=torch.cat(
                [
                    torch.ones_like(tags, dtype=torch.bool),
                    torch.zeros_like(used_expand_tags, dtype=torch.bool),
                ]
            ),
            deltapos=torch.cat(
                [pos_relaxed - pos, used_expand_pos_relaxed - used_expand_pos],
                dim=0,
            ),
            y_relaxed=torch.tensor([data.y_relaxed], dtype=torch.float),
            natoms=torch.tensor([data.num_nodes], dtype=torch.long),
            fixed=torch.cat([fixed, used_expand_fixed]),
        )


def _pad(
    data_list: List[Data],
    attr: str,
    batch_first: bool = True,
    padding_value: float = 0,
):
    return pad_sequence(
        [getattr(d, attr) for d in data_list],
        batch_first=batch_first,
        padding_value=padding_value,
    )


class Batch(NamedTuple):
    pos: torch.Tensor
    atoms: torch.Tensor
    tags: torch.Tensor
    real_mask: torch.Tensor
    deltapos: torch.Tensor
    y_relaxed: torch.Tensor
    fixed: torch.Tensor
    natoms: torch.Tensor

    def to(self, device):
        return Batch(
            pos=self.pos.to(device),
            atoms=self.atoms.to(device),
            tags=self.tags.to(device),
            real_mask=self.real_mask.to(device),
            deltapos=self.deltapos.to(device),
            y_relaxed=self.y_relaxed.to(device),
            fixed=self.fixed.to(device),
            natoms=self.natoms.to(device),
        )

    @classmethod
    def from_data_list(cls, data_list: List[Data]):
        batch = cls(
            pos=_pad(data_list, "pos"),
            atoms=_pad(data_list, "atoms"),
            tags=_pad(data_list, "tags"),
            real_mask=_pad(data_list, "real_mask"),
            deltapos=_pad(data_list, "deltapos"),
            y_relaxed=_pad(data_list, "y_relaxed"),
            fixed=_pad(data_list, "fixed"),
            natoms=_pad(data_list, "natoms"),
        )
        return [batch]
