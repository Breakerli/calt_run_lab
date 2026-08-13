# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Privileged observation helpers for critic (nogait-aligned)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from legged_lab.envs.ultra_game_yaw.ultra_motion_run_env import UltraGameYawMotionRunEnv


def height_scan_grid_shape(size: tuple[float, float], resolution: float) -> tuple[int, int]:
    nx = int(round(size[0] / resolution)) + 1
    ny = int(round(size[1] / resolution)) + 1
    return nx, ny


def height_scan_num_points(size: tuple[float, float], resolution: float) -> int:
    nx, ny = height_scan_grid_shape(size, resolution)
    return nx * ny


def height_scan_center_index(size: tuple[float, float], resolution: float) -> int:
    nx, ny = height_scan_grid_shape(size, resolution)
    return (nx // 2) * ny + (ny // 2)


def height_scan_y_mirror_permutation(
    size: tuple[float, float], resolution: float, device: torch.device | str
) -> torch.Tensor:
    """Permutation to mirror a flattened xy-ordered height scan across body y."""
    nx, ny = height_scan_grid_shape(size, resolution)
    perm = []
    for i in range(nx):
        for j in range(ny):
            perm.append(i * ny + (ny - 1 - j))
    return torch.tensor(perm, dtype=torch.long, device=device)


def sanitize_hit_heights_z(hit_z: torch.Tensor, sensor_z: torch.Tensor) -> torch.Tensor:
    """Replace inf/nan ray hits with sensor height (flat ground fallback)."""
    fallback = sensor_z.unsqueeze(-1).expand_as(hit_z) if hit_z.dim() > 1 else sensor_z.unsqueeze(1)
    valid = torch.isfinite(hit_z)
    return torch.where(valid, hit_z, fallback)


def compute_height_scan(
    scanner: RayCaster,
    height_offset: float,
    scale: float,
) -> torch.Tensor:
    """Relative height scan: sensor_z - hit_z - offset, shape (N, num_rays)."""
    hit_z = sanitize_hit_heights_z(scanner.data.ray_hits_w[..., 2], scanner.data.pos_w[:, 2])
    rel = scanner.data.pos_w[:, 2].unsqueeze(1) - hit_z - height_offset
    return rel * scale


def feet_ground_height_mean(feet_scanners: Sequence[RayCaster]) -> torch.Tensor:
    """Mean ground z under each foot grid, shape (N, num_feet)."""
    ground = []
    for scanner in feet_scanners:
        hit_z = sanitize_hit_heights_z(scanner.data.ray_hits_w[..., 2], scanner.data.pos_w[:, 2])
        ground.append(hit_z.mean(dim=-1, keepdim=True))
    return torch.cat(ground, dim=-1)


def root_lin_vel_yaw_frame(asset: Articulation, lin_vel_scale: float) -> torch.Tensor:
    """Root linear velocity in yaw frame (x,y from world vel, z kept)."""
    yaw_q = math_utils.yaw_quat(asset.data.root_quat_w)
    vel_yaw = math_utils.quat_apply_inverse(yaw_q, asset.data.root_lin_vel_w[:, :3])
    return vel_yaw * lin_vel_scale


def domain_rand_features(env: "UltraGameYawMotionRunEnv") -> torch.Tensor:
    """friction(1) + restitution(1) + p_gain_scale(13) + d_gain_scale(13) + motor_strength(13)."""
    robot = env.robot
    friction = getattr(env, "friction_coeffs", None)
    restitution = getattr(env, "restitution_coeffs", None)
    if friction is None:
        friction = torch.ones(env.num_envs, 1, device=env.device)
    if restitution is None:
        restitution = torch.ones(env.num_envs, 1, device=env.device)

    default_kp = robot.data.default_joint_stiffness.clamp(min=1.0e-6)
    default_kd = robot.data.default_joint_damping.clamp(min=1.0e-6)
    p_scale = robot.data.joint_stiffness / default_kp
    d_scale = robot.data.joint_damping / default_kd
    motor = env.motor_strength
    return torch.cat([friction, restitution, p_scale, d_scale, motor], dim=-1)
