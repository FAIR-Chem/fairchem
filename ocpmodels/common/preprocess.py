import math
import time
from collections import defaultdict, deque

import torch


class Preprocess:
    def perturb_atom_positions(
        atom_pos, grid_resolution, randomly_rotate=True
    ):
        device = atom_pos.device

        # Add a random offset to positions
        atom_pos_perturb = atom_pos + (
            torch.rand(3, device=device) * grid_resolution
        )
        bounding_box = torch.zeros(2, 3, device=device)
        (bounding_box[0], indices) = torch.min(atom_pos_perturb, dim=0)
        (bounding_box[1], indices) = torch.max(atom_pos_perturb, dim=0)
        atom_center = (bounding_box[1] + bounding_box[0]) / 2.0
        atom_pos_perturb = atom_pos_perturb - atom_center

        inv_rot = torch.eye(3, device=device)
        # Randomly sample an angle
        if randomly_rotate is True:
            ang_a = 2.0 * math.pi * torch.rand(1)[0]
            ang_b = 2.0 * math.pi * torch.rand(1)[0]
            ang_c = 2.0 * math.pi * torch.rand(1)[0]
            cos_a = math.cos(ang_a)
            cos_b = math.cos(ang_b)
            cos_c = math.cos(ang_c)
            sin_a = math.sin(ang_a)
            sin_b = math.sin(ang_b)
            sin_c = math.sin(ang_c)

            rot_a = torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, cos_a, sin_a], [0.0, -sin_a, cos_a]],
                device=device,
            ).float()
            rot_b = torch.tensor(
                [[cos_b, 0.0, -sin_b], [0.0, 1.0, 0.0], [sin_b, 0.0, cos_b]],
                device=device,
            ).float()
            rot_c = torch.tensor(
                [[cos_c, sin_c, 0.0], [-sin_c, cos_c, 0.0], [0.0, 0.0, 1.0]],
                device=device,
            ).float()

            rotation_m = torch.mm(torch.mm(rot_a, rot_b), rot_c)
            for i in range(len(atom_pos_perturb)):
                atom_pos_perturb[i] = torch.mv(rotation_m, atom_pos_perturb[i])
            inv_rot = torch.inverse(rotation_m)

        return atom_pos_perturb + atom_center, inv_rot
