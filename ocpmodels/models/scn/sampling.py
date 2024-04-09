"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import math

import torch

### Methods for sample points on a sphere


def CalcSpherePoints(num_points: int, device: str = "cpu") -> torch.Tensor:
    goldenRatio = (1 + 5**0.5) / 2
    i = torch.arange(num_points, device=device).view(-1, 1)
    theta = 2 * math.pi * i / goldenRatio
    phi = torch.arccos(1 - 2 * (i + 0.5) / num_points)
    points = torch.cat(
        [
            torch.cos(theta) * torch.sin(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(phi),
        ],
        dim=1,
    )

    # weight the points by their density
    pt_cross = points.view(1, -1, 3) - points.view(-1, 1, 3)
    pt_cross = torch.sum(pt_cross**2, dim=2)
    pt_cross = torch.exp(-pt_cross / (0.5 * 0.3))
    scalar = 1.0 / torch.sum(pt_cross, dim=1)
    scalar = num_points * scalar / torch.sum(scalar)
    return points * (scalar.view(-1, 1))


def CalcSpherePointsRandom(num_points: int, device) -> torch.Tensor:
    pts = 2.0 * (torch.rand(num_points, 3, device=device) - 0.5)
    radius = torch.sum(pts**2, dim=1)
    while torch.max(radius) > 1.0:
        replace_pts = 2.0 * (torch.rand(num_points, 3, device=device) - 0.5)
        replace_mask = radius.gt(0.99)
        pts.masked_scatter_(replace_mask.view(-1, 1).repeat(1, 3), replace_pts)
        radius = torch.sum(pts**2, dim=1)

    return pts / radius.view(-1, 1)
