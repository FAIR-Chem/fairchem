from __future__ import annotations

import logging
import math

import torch


# Algorithm from Ken Whatmough (https://math.stackexchange.com/users/918128/ken-whatmough)
def vec3_to_perp_vec3(v):
    """
    Small proof:
        input  = x          y         z
        output = s(x)|z|   s(y)|z|   -s(z)(|x|+|y|)

        input dot output
            = x*s(x)*|z|  + y*s(y)*|z|  - z*s(z)*|x| - z*s(z)*|y|
        a*s(a)=|a| ,
            = |x|*|z| + |y|*|z| - |z|*|x| - |z|*|y| = 0

    """
    return torch.hstack(
        [
            v[:, [2]].copysign(v[:, [0, 1]]),
            -v[:, [0, 1]].copysign(v[:, [2]]).sum(axis=1, keepdim=True),
        ]
    )


# https://en.wikipedia.org/wiki/Rodrigues'_rotation_formula#Matrix_notation
def vec3_rotate_around_axis(v, axis, thetas):
    # v_rot= v + (sTheta)*(axis X v) + (1-cTheta)*(axis X (axis X v))
    Kv = torch.cross(axis, v, dim=1)
    KKv = torch.cross(axis, Kv, dim=1)
    s_theta = torch.sin(thetas)
    c_theta = torch.cos(thetas)
    return v + s_theta * Kv + (1 - c_theta) * KKv


def init_edge_rot_mat(edge_distance_vec):
    edge_vec_0 = edge_distance_vec.detach()
    edge_vec_0_distance = torch.linalg.norm(edge_vec_0, axis=1, keepdim=True)
    # Make sure the atoms are far enough apart
    # assert torch.min(edge_vec_0_distance) < 0.0001
    if torch.min(edge_vec_0_distance) < 0.0001:
        logging.error(f"Error edge_vec_0_distance: {torch.min(edge_vec_0_distance)}")

    norm_x = edge_vec_0 / edge_vec_0_distance

    perp_to_norm_x = vec3_to_perp_vec3(norm_x)
    random_rotated_in_plane_perp_to_norm_x = vec3_rotate_around_axis(
        perp_to_norm_x,
        norm_x,
        torch.rand((norm_x.shape[0], 1), device=norm_x.device) * 2 * math.pi,
    )

    norm_z = random_rotated_in_plane_perp_to_norm_x / torch.linalg.norm(
        random_rotated_in_plane_perp_to_norm_x, axis=1, keepdim=True
    )

    norm_y = torch.cross(norm_x, norm_z, dim=1)
    norm_y /= torch.linalg.norm(norm_y, dim=1, keepdim=True)

    # Construct the 3D rotation matrix
    norm_x = norm_x.view(-1, 1, 3)
    norm_y = -norm_y.view(-1, 1, 3)
    norm_z = norm_z.view(-1, 1, 3)
    return torch.cat([norm_z, norm_x, norm_y], dim=1).contiguous()
