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


def init_edge_rot_mat_new(data, edge_index, edge_distance_vec):
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


# Initialize the edge rotation matrics
def init_edge_rot_mat_og(data, edge_index, edge_distance_vec):
    edge_vec_0 = edge_distance_vec
    edge_vec_0_distance = torch.sqrt(torch.sum(edge_vec_0**2, dim=1))

    # Make sure the atoms are far enough apart
    if torch.min(edge_vec_0_distance) < 0.0001:
        logging.error(f"Error edge_vec_0_distance: {torch.min(edge_vec_0_distance)}")
        (minval, minidx) = torch.min(edge_vec_0_distance, 0)
        logging.error(
            f"Error edge_vec_0_distance: {minidx} {edge_index[0, minidx]} {edge_index[1, minidx]} {data.pos[edge_index[0, minidx]]} {data.pos[edge_index[1, minidx]]}"
        )

    norm_x = edge_vec_0 / (edge_vec_0_distance.view(-1, 1))

    edge_vec_2 = torch.rand_like(edge_vec_0) - 0.5
    edge_vec_2 = edge_vec_2 / (torch.sqrt(torch.sum(edge_vec_2**2, dim=1)).view(-1, 1))
    # Create two rotated copys of the random vectors in case the random vector is aligned with norm_x
    # With two 90 degree rotated vectors, at least one should not be aligned with norm_x
    edge_vec_2b = edge_vec_2.clone()
    edge_vec_2b[:, 0] = -edge_vec_2[:, 1]
    edge_vec_2b[:, 1] = edge_vec_2[:, 0]
    edge_vec_2c = edge_vec_2.clone()
    edge_vec_2c[:, 1] = -edge_vec_2[:, 2]
    edge_vec_2c[:, 2] = edge_vec_2[:, 1]
    vec_dot_b = torch.abs(torch.sum(edge_vec_2b * norm_x, dim=1)).view(-1, 1)
    vec_dot_c = torch.abs(torch.sum(edge_vec_2c * norm_x, dim=1)).view(-1, 1)

    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
    edge_vec_2 = torch.where(torch.gt(vec_dot, vec_dot_b), edge_vec_2b, edge_vec_2)
    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
    edge_vec_2 = torch.where(torch.gt(vec_dot, vec_dot_c), edge_vec_2c, edge_vec_2)

    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1))
    # Check the vectors aren't aligned
    assert torch.max(vec_dot) < 0.99

    norm_z = torch.cross(norm_x, edge_vec_2, dim=1)
    norm_z = norm_z / (torch.sqrt(torch.sum(norm_z**2, dim=1, keepdim=True)))
    norm_z = norm_z / (torch.sqrt(torch.sum(norm_z**2, dim=1)).view(-1, 1))
    norm_y = torch.cross(norm_x, norm_z, dim=1)
    norm_y = norm_y / (torch.sqrt(torch.sum(norm_y**2, dim=1, keepdim=True)))

    # Construct the 3D rotation matrix
    norm_x = norm_x.view(-1, 3, 1)
    norm_y = -norm_y.view(-1, 3, 1)
    norm_z = norm_z.view(-1, 3, 1)

    edge_rot_mat_inv = torch.cat([norm_z, norm_x, norm_y], dim=2)
    edge_rot_mat = torch.transpose(edge_rot_mat_inv, 1, 2)

    return edge_rot_mat.detach()
