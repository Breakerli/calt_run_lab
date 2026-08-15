# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env import BaseEnv
    from legged_lab.envs.ultra_js_wb.ultra_motion_run_env import UltraMotionRunEnv


def _std_scaled_by_cmd_x(
    env: BaseEnv | UltraMotionRunEnv,
    std: float,
    std_speed_max: float | None = None,
    speed_walk: float = 1.0,
    speed_max: float = 15.0,
) -> torch.Tensor:
    """按 |cmd_x| 在 [speed_walk, speed_max] 上 smoothstep 放宽 std；未设 std_speed_max 则恒定 std."""
    device = env.device
    if std_speed_max is None:
        return torch.full((env.num_envs,), std, device=device, dtype=torch.float32)
    cmd_x = torch.abs(env.command_generator.command[:, 0])
    alpha = ((cmd_x - speed_walk) / (speed_max - speed_walk)).clamp(0.0, 1.0)
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    return std + (std_speed_max - std) * alpha


def quat_to_euler_xyz(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (w, x, y, z) to Euler angles (XYZ order)."""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    sinp = torch.clamp(sinp, -1.0, 1.0)  # Clamp to avoid numerical issues
    pitch = torch.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=-1)


def euler_xyz_to_quat(euler: torch.Tensor) -> torch.Tensor:
    """Convert Euler angles (XYZ order) to quaternion (w, x, y, z)."""
    roll, pitch, yaw = euler[..., 0], euler[..., 1], euler[..., 2]

    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return torch.stack([w, x, y, z], dim=-1)


def track_lin_vel_x_yaw_frame_exp(
    env: BaseEnv | UltraMotionRunEnv,
    std: float,
    std_speed_max: float | None = None,
    speed_walk: float = 1.0,
    speed_max: float = 15.0,
    avg_vel_window_s: float | None = None,
    avg_vel_window_s_speed_max: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    lin_vel_x = _tracked_lin_vel_x_yaw(
        env, asset, avg_vel_window_s, avg_vel_window_s_speed_max, speed_walk, speed_max
    )
    lin_vel_error_x = torch.abs(env.command_generator.command[:, 0] - lin_vel_x)
    std_eff = _std_scaled_by_cmd_x(env, std, std_speed_max, speed_walk, speed_max)
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return torch.exp(-lin_vel_error_x / (std_eff * std_eff)) * mask


def _tracked_lin_vel_x_yaw(
    env: BaseEnv | UltraMotionRunEnv,
    asset: Articulation,
    avg_vel_window_s: float | None,
    avg_vel_window_s_speed_max: float | None = None,
    speed_walk: float = 1.0,
    speed_max: float = 15.0,
) -> torch.Tensor:
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    if avg_vel_window_s is None:
        return vel_yaw[:, 0]
    if avg_vel_window_s_speed_max is None and hasattr(env, "avg_lin_vel_x_yaw"):
        return env.avg_lin_vel_x_yaw

    if not hasattr(env, "track_lin_vel_x_yaw_avg_buffer"):
        env.track_lin_vel_x_yaw_avg_buffer = vel_yaw[:, 0].clone()
    cmd_x = torch.abs(env.command_generator.command[:, 0])
    blend = ((cmd_x - speed_walk) / max(speed_max - speed_walk, 1e-3)).clamp(0.0, 1.0)
    blend = blend * blend * (3.0 - 2.0 * blend)
    window = avg_vel_window_s + (avg_vel_window_s_speed_max - avg_vel_window_s) * blend
    alpha = (env.step_dt / window.clamp(min=1e-3)).clamp(max=1.0)
    env.track_lin_vel_x_yaw_avg_buffer.mul_(1.0 - alpha).add_(alpha * vel_yaw[:, 0])
    return env.track_lin_vel_x_yaw_avg_buffer


def _tracked_lin_vel_y_yaw(
    env: BaseEnv | UltraMotionRunEnv, asset: Articulation, avg_vel_window_s: float | None
) -> torch.Tensor:
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    if avg_vel_window_s is not None and hasattr(env, "avg_lin_vel_y_yaw"):
        return env.avg_lin_vel_y_yaw
    return vel_yaw[:, 1]


def _tracked_ang_vel_z(
    env: BaseEnv | UltraMotionRunEnv, asset: Articulation, avg_vel_window_s: float | None
) -> torch.Tensor:
    if avg_vel_window_s is not None and hasattr(env, "avg_ang_vel_z"):
        return env.avg_ang_vel_z
    return asset.data.root_ang_vel_w[:, 2]


def track_lin_vel_y_yaw_frame_exp(
    env: BaseEnv | UltraMotionRunEnv,
    std: float,
    avg_vel_window_s: float | None = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    lin_vel_y = _tracked_lin_vel_y_yaw(env, asset, avg_vel_window_s)
    lin_vel_error_y = torch.abs(env.command_generator.command[:, 1] - lin_vel_y)
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return torch.exp(-lin_vel_error_y / std**2) * mask


def track_lin_vel_z_yaw_frame_exp(
    env: BaseEnv | UltraMotionRunEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error_z = torch.abs(vel_yaw[:, 2])
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return torch.exp(-lin_vel_error_z / std**2) * mask


def track_lin_vel_xy_yaw_frame_exp(
    env: BaseEnv | UltraMotionRunEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(torch.square(env.command_generator.command[:, :2] - vel_yaw[:, :2]), dim=1)
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return torch.exp(-lin_vel_error / std**2) * mask


def track_ang_vel_z_world_exp(
    env: BaseEnv | UltraMotionRunEnv,
    std: float,
    avg_vel_window_s: float | None = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_z = _tracked_ang_vel_z(env, asset, avg_vel_window_s)
    ang_vel_error = torch.square(env.command_generator.command[:, 2] - ang_vel_z)
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return torch.exp(-ang_vel_error / std**2) * mask


def track_lin_vel_x_yaw_frame_bonus(
    env: BaseEnv | UltraMotionRunEnv,
    std: float,
    std_speed_max: float | None = None,
    speed_walk: float = 1.0,
    speed_max: float = 15.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error_x = torch.abs(env.command_generator.command[:, 0] - vel_yaw[:, 0])
    std_eff = _std_scaled_by_cmd_x(env, std, std_speed_max, speed_walk, speed_max)
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    cmd_scale = torch.abs(env.command_generator.command[:, 0])
    return torch.exp(-lin_vel_error_x / (std_eff * std_eff)) * cmd_scale * mask


def track_lin_vel_x_progress_bonus(
    env: BaseEnv | UltraMotionRunEnv,
    speed_threshold: float = 2.0,
    speed_max: float = 15.0,
    normalize_speed: bool = True,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    target = env.command_generator.command[:, 0].clamp(min=0.0, max=speed_max)
    actual = vel_yaw[:, 0].clamp(min=0.0, max=speed_max)
    rewardable = torch.minimum(actual, target)
    reward = (rewardable - speed_threshold).clamp(min=0.0, max=speed_max - speed_threshold)
    if normalize_speed:
        reward = reward / max(speed_max - speed_threshold, 1e-3)
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return reward * mask


def track_lin_vel_x_yaw_frame_neg(
    env: BaseEnv | UltraMotionRunEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error_x = torch.abs(env.command_generator.command[:, 0] - vel_yaw[:, 0])
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return lin_vel_error_x * mask


def track_lin_vel_y_yaw_frame_neg(
    env: BaseEnv | UltraMotionRunEnv,
    avg_vel_window_s: float | None = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    """Penalize lateral velocity tracking error in yaw frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    lin_vel_y = _tracked_lin_vel_y_yaw(env, asset, avg_vel_window_s)
    lin_vel_error_y = torch.abs(env.command_generator.command[:, 1] - lin_vel_y)
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return lin_vel_error_y * mask


def track_ang_vel_z_world_bonus(
    env: BaseEnv | UltraMotionRunEnv,
    std: float,
    avg_vel_window_s: float | None = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_z = _tracked_ang_vel_z(env, asset, avg_vel_window_s)
    ang_vel_error = torch.square(env.command_generator.command[:, 2] - ang_vel_z)
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    cmd_scale = torch.abs(env.command_generator.command[:, 0])
    return torch.exp(-ang_vel_error / std**2) * cmd_scale * mask


def track_ang_vel_z_world_neg(
    env: BaseEnv | UltraMotionRunEnv,
    avg_vel_window_s: float | None = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    """Penalize angular velocity tracking error."""
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_z = _tracked_ang_vel_z(env, asset, avg_vel_window_s)
    ang_vel_error = torch.abs(env.command_generator.command[:, 2] - ang_vel_z)
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return ang_vel_error * mask


def ang_vel_xy_body_exp(
    env: BaseEnv | UltraMotionRunEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    """Reward low roll/pitch angular velocity (body frame) with exp(std) shaping."""
    asset: Articulation = env.scene[asset_cfg.name]
    err = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return torch.exp(-err / std**2) * mask


def lin_vel_z_world_exp(
    env: BaseEnv | UltraMotionRunEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    """Reward low vertical linear velocity (world z) with exp(std) shaping."""
    asset: Articulation = env.scene[asset_cfg.name]
    err = torch.abs(asset.data.root_lin_vel_w[:, 2])
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return torch.exp(-err / std**2) * mask


def lin_vel_z_l2(
    env: BaseEnv | UltraMotionRunEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), style_mask: list[int] | None = None
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return torch.square(asset.data.root_lin_vel_b[:, 2]) * mask


def ang_vel_xy_l1(env: BaseEnv | UltraMotionRunEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.root_ang_vel_b[:, :2]), dim=1)


def ang_vel_xy_l2(env: BaseEnv | UltraMotionRunEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def ang_vel_x_l1(
    env: BaseEnv | UltraMotionRunEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), style_mask: list[int] | None = None
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return torch.abs(asset.data.root_ang_vel_b[:, 0]) * mask


def energy(env: BaseEnv | UltraMotionRunEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward


def joint_acc_l2(env: BaseEnv | UltraMotionRunEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_vel_limits_neg(
    env: BaseEnv | UltraMotionRunEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soft_limit: float = 0.9,
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    """Penalize joint velocities only after they exceed a soft fraction of the limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    vel_limits = asset.data.joint_vel_limits[:, asset_cfg.joint_ids] * soft_limit
    penalty = 0.1 * torch.sum((joint_vel - vel_limits).clip(min=0.0, max=3.0), dim=1)
    if style_mask is None:
        mask = torch.ones_like(penalty)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return penalty * mask


def torque_limits_neg(
    env: BaseEnv | UltraMotionRunEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soft_limit: float = 0.85,
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    """Penalize actuator torques only after they exceed a soft fraction of the limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    torques = asset.data.applied_torque[:, asset_cfg.joint_ids]
    torque_limits = asset.data.joint_effort_limits[:, asset_cfg.joint_ids] * soft_limit
    penalty = 0.005 * torch.sum(
        (torch.abs(torques) - torch.abs(torque_limits)).clip(min=0.0), dim=1
    )
    if style_mask is None:
        mask = torch.ones_like(penalty)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return penalty * mask


def action_rate_l2(env: BaseEnv | UltraMotionRunEnv) -> torch.Tensor:
    return torch.sum(
        torch.square(
            env.action_buffer._circular_buffer.buffer[:, -1, :] - env.action_buffer._circular_buffer.buffer[:, -2, :]
        ),
        dim=1,
    )


def _action_rate_speed_scale(
    env: UltraMotionRunEnv,
    min_scale: float = 0.4,
    max_scale: float = 2.0,
    speed_relax: float = 3.0,
) -> torch.Tensor:
    """Smoothly relax action-rate penalty as forward command increases."""
    cmd_x = torch.abs(env.command_generator.command[:, 0])
    alpha = (cmd_x / max(speed_relax, 1e-3)).clamp(0.0, 1.0)
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    return max_scale + (min_scale - max_scale) * alpha

def action_rate_weighted_l2(
    env: UltraMotionRunEnv,
    style_mask: list[int] | None = None,
    min_speed_scale: float = 0.4,
    max_speed_scale: float = 2.0,
    speed_relax: float = 3.0,
    max_penalty: float = 20.0,
) -> torch.Tensor:
    """Per-joint weighted first-order action rate (2026gym ``_reward_action_smooth_first``)."""
    buf = env.action_buffer._circular_buffer.buffer
    diff = buf[:, -1, :] - buf[:, -2, :]
    w = _xhumanoid_joint_weights(env, _XHUMANOID_ACTION_RATE_WEIGHTS, "_xhumanoid_action_rate_weights")
    penalty = torch.sum(w * torch.square(diff), dim=1) * _action_rate_speed_scale(
        env, min_speed_scale, max_speed_scale, speed_relax
    )
    penalty = penalty.clamp(max=max_penalty)
    if style_mask is None:
        mask = torch.ones_like(env.style_ids, dtype=torch.float32)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return penalty * mask


def action_rate2_weighted_l2(
    env: UltraMotionRunEnv,
    style_mask: list[int] | None = None,
    min_speed_scale: float = 0.4,
    max_speed_scale: float = 2.0,
    speed_relax: float = 3.0,
    max_penalty: float = 40.0,
) -> torch.Tensor:
    """Per-joint weighted second-order action rate (2026gym ``_reward_action_smooth_second``)."""
    buf = env.action_buffer._circular_buffer.buffer
    ddiff = buf[:, -1, :] - 2.0 * buf[:, -2, :] + buf[:, -3, :]
    w = _xhumanoid_joint_weights(env, _XHUMANOID_ACTION_RATE_WEIGHTS, "_xhumanoid_action_rate_weights")
    penalty = torch.sum(w * torch.square(ddiff), dim=1) * _action_rate_speed_scale(
        env, min_speed_scale, max_speed_scale, speed_relax
    )
    penalty = penalty.clamp(max=max_penalty)
    if style_mask is None:
        mask = torch.ones_like(env.style_ids, dtype=torch.float32)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return penalty * mask


def shoulder_pitch_action_neg(
    env: BaseEnv | UltraMotionRunEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    """Penalize shoulder pitch actions."""
    asset: Articulation = env.scene[asset_cfg.name]
    actions = env.action_buffer._circular_buffer.buffer[:, -1, :]
    shoulder_ids, _ = asset.find_joints(
        name_keys=["shoulder_pitch_l_joint", "shoulder_pitch_r_joint"], preserve_order=True
    )
    penalty = torch.sum(torch.square(actions[:, shoulder_ids]), dim=1)
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return penalty * mask


def shoulder_pitch_pos_neg(
    env: BaseEnv | UltraMotionRunEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    relax_speed: float = 15.0,
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    """Penalize absolute shoulder pitch position, relaxing as forward command increases."""
    asset: Articulation = env.scene[asset_cfg.name]
    shoulder_ids, _ = asset.find_joints(
        name_keys=["shoulder_pitch_l_joint", "shoulder_pitch_r_joint"], preserve_order=True
    )
    penalty = torch.sum(torch.square(asset.data.joint_pos[:, shoulder_ids]), dim=1)
    relax = (torch.abs(env.command_generator.command[:, 0]) / max(relax_speed, 1e-3)).clamp(0.0, 1.0)
    penalty = penalty * (1.0 - relax)
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return penalty * mask


def shoulder_pitch_vel_neg(
    env: BaseEnv | UltraMotionRunEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    """Penalize shoulder pitch velocities."""
    asset: Articulation = env.scene[asset_cfg.name]
    shoulder_ids, _ = asset.find_joints(
        name_keys=["shoulder_pitch_l_joint", "shoulder_pitch_r_joint"], preserve_order=True
    )
    penalty = torch.sum(torch.square(asset.data.joint_vel[:, shoulder_ids]), dim=1)
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return penalty * mask


def shoulder_pitch_acc_neg(
    env: BaseEnv | UltraMotionRunEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    """Penalize shoulder pitch accelerations."""
    asset: Articulation = env.scene[asset_cfg.name]
    shoulder_ids, _ = asset.find_joints(
        name_keys=["shoulder_pitch_l_joint", "shoulder_pitch_r_joint"], preserve_order=True
    )
    penalty = torch.sum(torch.square(asset.data.joint_acc[:, shoulder_ids]), dim=1)
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return penalty * mask


def arm_vel_ang_yaw_neg(
    env: BaseEnv | UltraMotionRunEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    """Penalize arm joint velocities during yaw commands."""
    asset: Articulation = env.scene[asset_cfg.name]
    arm_ids = env.left_arm_ids + env.right_arm_ids
    penalty = torch.sum(torch.square(asset.data.joint_vel[:, arm_ids]), dim=1)
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    cmd_scale = torch.abs(env.command_generator.command[:, 2]) * torch.abs(env.command_generator.command[:, 0]).clip(max=2.0)
    return penalty * cmd_scale * mask


def undesired_contacts(env: BaseEnv | UltraMotionRunEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def fly(env: BaseEnv | UltraMotionRunEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=-1) < 0.5


def flat_orientation_l2(
    env: BaseEnv | UltraMotionRunEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    penalty = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    if style_mask is None:
        mask = torch.ones_like(env.style_ids, dtype=torch.float32)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return penalty * mask


def is_alive(env: BaseEnv | UltraMotionRunEnv) -> torch.Tensor:
    """Reward for not terminating; scales with episode progress (later steps weigh more)."""
    max_len = float(env.max_episode_length)
    progress = env.episode_length_buf.float().clamp(max=max_len) / max_len
    return (~env.reset_buf).float() * progress


def is_terminated(env: BaseEnv | UltraMotionRunEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.reset_buf * ~env.time_out_buf


def feet_air_time_positive_biped(
    env: BaseEnv | UltraMotionRunEnv, 
    threshold: float, 
    sensor_cfg: SceneEntityCfg, 
    style_mask: list[int] | None = None,
    lateral_cmd_scale: bool = False,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command; yaw-only walk should also encourage stepping
    cmd_active = (torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])) > 0.1
    reward *= cmd_active
    if lateral_cmd_scale:
        lat_max = max(abs(float(env.cfg.commands.ranges.lin_vel_y[0])), abs(float(env.cfg.commands.ranges.lin_vel_y[1])), 1e-3)
        reward *= (torch.abs(env.command_generator.command[:, 1]) / lat_max).clamp(0.0, 1.0)
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return reward * mask


def cadence_neg(
    env: BaseEnv | UltraMotionRunEnv,
    target_cadence: float = 5.0,
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    """惩罚过高的步频（步/秒），间接鼓励更大步幅、更低步频。

    使用逐步累计的 run style 步数与 run 时长计算局内平均步频（即 ``Gait/cadence``
    同一口径）；超过 ``target_cadence`` 的相对幅度线性惩罚，**不封顶**——
    旧实现 ``clamp(max=1.0)`` 在 cadence > 2×target 时惩罚饱和、梯度为零，
    实测策略陷入 8.7 步/s（target 3.0）局部最优时该奖励形同虚设。
    仅在 run style 生效，避免 stand/walk 拉低步频导致无惩罚。
    """
    if not hasattr(env, "_run_step_count") or not hasattr(env, "_run_style_steps"):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    run_time = env._run_style_steps * env.step_dt
    cadence = env._run_step_count / run_time.clamp(min=env.step_dt)  # steps/s
    excess = (cadence / max(target_cadence, 1e-3) - 1.0).clamp(min=0.0)
    if style_mask is None:
        mask = torch.ones_like(env.style_ids, dtype=torch.float32)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    # 步数足够才生效，避免 episode 初期少量步数导致步频尖峰（噪声惩罚）
    enough = (env._run_step_count >= 4).float()
    return excess * mask * enough


def foot_strike_com_offset(
    env: BaseEnv | UltraMotionRunEnv,
    target_offset: float = 0.5,
    std: float = 0.1,
    behind_penalty: float = 2.0,
    target_stride: float = 0.5,
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    """奖励前脚落脚点相对质心(COM)的前向位移逼近目标；落在质心后硬惩罚。

    落脚点相对质心的前向位移在落脚沿采样并缓存在 ``env._foot_strike_offset``，
    这里逐 step 读取最近一次落脚的 offset，gaussian 到 ``target_offset``。
    offset < 0（落脚点在质心后，支撑点落在重心之后 → 前倾失衡摔倒）线性惩罚。
    正向 gaussian 奖励乘以**平移门控**：本局平均步幅（净前移，``_run_forward_dist``
    口径）达到 ``target_stride`` 才全额发放——原地踏步也能让脚落在质心前，
    不门控会被"高步频原地踏步"钻空子（实测 stride≈0.05m 仍能领 ~73%）。
    负向 behind_penalty 不门控，防摔信号始终保持。
    仅 run style 生效；落地次数足够才生效，避免 episode 初期 offset=0 的虚假信号。
    """
    if not hasattr(env, "_foot_strike_offset"):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    offset = env._foot_strike_offset  # (num_envs,) 前向为正
    err = offset - target_offset
    reward = torch.exp(-0.5 * (err / max(std, 1e-3)) ** 2)
    # 平移门控：stride = 2 × 净前移 / 步数（与 Gait/stride_length 同口径）
    if hasattr(env, "_run_forward_dist") and hasattr(env, "_run_step_count"):
        stride = 2.0 * env._run_forward_dist / env._run_step_count.clamp(min=1.0)
        reward = reward * (stride / max(target_stride, 1e-3)).clamp(0.0, 1.0)
    reward = reward - behind_penalty * torch.clamp(-offset, min=0.0)
    if style_mask is None:
        mask = torch.ones_like(env.style_ids, dtype=torch.float32)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    # 至少落地 1 次才生效，避免 offset 初始值 0 造成虚假的 gaussian 奖励
    enough = (getattr(env, "_run_step_count", torch.zeros_like(offset)) >= 1).float()
    return reward * mask * enough


def flight_ratio_neg(
    env: BaseEnv | UltraMotionRunEnv,
    max_flight_ratio: float = 0.7,
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    """惩罚 run style 腾空占比过高（跳跃/坠落），把腾空从"目标"降为"约束"。

    旧 ``feet_air_time_run`` 直接奖励腾空时间，会诱导"原地跳跃"退化解（flight_ratio→0.95）；
    这里改为对超过 ``max_flight_ratio`` 的腾空占比线性惩罚。
    """
    if not hasattr(env, "_run_flight_steps") or not hasattr(env, "_run_style_steps"):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    flight = env._run_flight_steps / env._run_style_steps.clamp(min=1.0)
    excess = (flight - max_flight_ratio).clamp(min=0.0)
    if style_mask is None:
        mask = torch.ones_like(env.style_ids, dtype=torch.float32)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return excess * mask


def feet_clearance(
    env: BaseEnv | UltraMotionRunEnv,
    target_height: float = 0.08,
    std: float = 0.04,
    sensor_cfg: SceneEntityCfg | None = None,
    style_mask: list[int] | None = None,
    lateral_cmd_scale: bool = False,
) -> torch.Tensor:
    """Reward swing-foot reaching a target ground clearance during locomotion.

    For each foot in swing phase (no ground contact), give a Gaussian bonus around
    ``target_height`` based on world-frame foot z. Only granted when the base is
    commanded to move (``|cmd_xy| + |cmd_yaw| > 0.1``) so that lifting feet in place is not rewarded.
    Sum across both feet, then masked by style.
    """
    asset: Articulation = env.scene["robot"]
    feet_body_ids = env.feet_body_ids  # [left, right]

    if sensor_cfg is not None:
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        sensor_body_ids = sensor_cfg.body_ids
    else:
        contact_sensor = env.contact_sensor
        sensor_body_ids = env.feet_cfg.body_ids

    foot_z = asset.data.body_state_w[:, feet_body_ids, 2]  # (N, 2)
    contact_force = contact_sensor.data.net_forces_w_history[:, :, sensor_body_ids, :].norm(dim=-1).max(dim=1)[0]
    in_swing = (contact_force <= 1.0).float()  # (N, 2)

    height_score = torch.exp(-torch.square(foot_z - target_height) / (std**2))
    reward = (height_score * in_swing).sum(dim=-1)

    cmd_active = (torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])) > 0.1
    reward = reward * cmd_active.float()
    if lateral_cmd_scale:
        lat_max = max(abs(float(env.cfg.commands.ranges.lin_vel_y[0])), abs(float(env.cfg.commands.ranges.lin_vel_y[1])), 1e-3)
        reward = reward * (torch.abs(env.command_generator.command[:, 1]) / lat_max).clamp(0.0, 1.0)

    if style_mask is None:
        mask = torch.ones_like(env.style_ids, dtype=torch.float32)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return reward * mask


def feet_slide(
    env: BaseEnv | UltraMotionRunEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_contact_fz: float = 100.0,
    slip_speed_threshold: float = 0.35,
    only_lateral_yaw: bool = True,
    high_speed_slip_threshold: float | None = None,
    high_speed_relax: float = 3.0,
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    """Penalize foot slip while in contact.

    ``only_lateral_yaw=True`` (default): only lateral slip vs base in yaw frame, so
    sagittal motion during late-stance / forefoot push-off is not punished.
    ``slip_speed_threshold``: deadband [m/s] for small rolling at the contact patch.
    If ``high_speed_slip_threshold`` is set, use ground-relative foot velocity:
    lateral slip keeps the low threshold, while sagittal slip relaxes with |cmd_x|.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    fz = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    in_contact = fz > min_contact_fz

    yaw_q = math_utils.yaw_quat(asset.data.root_quat_w)
    foot_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    num_feet = foot_vel_w.shape[1]
    yaw_rep = yaw_q.unsqueeze(1).expand(-1, num_feet, -1).reshape(-1, 4)
    foot_yaw = math_utils.quat_apply_inverse(yaw_rep, foot_vel_w.reshape(-1, 3)).reshape(
        env.num_envs, num_feet, 3
    )

    if high_speed_slip_threshold is not None:
        cmd_x = torch.abs(env.command_generator.command[:, 0])
        alpha = (cmd_x / max(high_speed_relax, 1e-3)).clamp(0.0, 1.0)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        sagittal_threshold = slip_speed_threshold + (high_speed_slip_threshold - slip_speed_threshold) * alpha

        lateral_excess = (torch.abs(foot_yaw[:, :, 1]) - slip_speed_threshold).clamp(min=0.0)
        sagittal_excess = (torch.abs(foot_yaw[:, :, 0]) - sagittal_threshold.unsqueeze(1)).clamp(min=0.0)
        reward = torch.sum((lateral_excess + sagittal_excess) * in_contact.float(), dim=1)
    else:
        base_vel_yaw = math_utils.quat_apply_inverse(yaw_q, asset.data.root_lin_vel_w[:, :3])[:, :2]
        rel_xy = foot_yaw[:, :, :2] - base_vel_yaw.unsqueeze(1)
        if only_lateral_yaw:
            slip = torch.abs(rel_xy[:, :, 1])
        else:
            slip = torch.norm(rel_xy, dim=-1)
        reward = torch.sum((slip - slip_speed_threshold).clamp(min=0.0) * in_contact.float(), dim=1)

    if style_mask is None:
        mask = torch.ones_like(env.style_ids, dtype=torch.float32)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return reward * mask


def feet_lin_vel_stand_neg(
    env: BaseEnv | UltraMotionRunEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="ankle_pitch.*"),
    horizontal_only: bool = True,
    relax_speed: float = 1.0,
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    """Penalize foot link velocity while standing, relaxing as actual base speed grows."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    if horizontal_only:
        foot_vel = foot_vel[:, :, :2]
    penalty = torch.sum(torch.square(foot_vel), dim=(1, 2))
    actual_speed = torch.norm(asset.data.root_lin_vel_w[:, :2], dim=1)
    relax = (actual_speed / max(relax_speed, 1e-3)).clamp(0.0, 1.0)
    relax = relax * relax * (3.0 - 2.0 * relax)
    penalty = penalty * (1.0 - relax)
    if style_mask is None:
        mask = torch.ones_like(penalty)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return penalty * mask


def feet_contact_ang_vel_z(
    env: BaseEnv | UltraMotionRunEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_contact_fz: float = 100.0,
    max_penalty: float = 10.0,
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    """Penalize foot yaw angular velocity while the foot is in ground contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    fz = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    in_contact = fz > min_contact_fz

    foot_ang_vel_z = asset.data.body_ang_vel_w[:, asset_cfg.body_ids, 2]
    penalty = torch.sum(torch.square(foot_ang_vel_z) * in_contact.float(), dim=1).clamp(max=max_penalty)
    if style_mask is None:
        mask = torch.ones_like(env.style_ids, dtype=torch.float32)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return penalty * mask


def body_force(
    env: BaseEnv | UltraMotionRunEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward


def joint_deviation_l1(env: BaseEnv | UltraMotionRunEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    zero_flag = (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) < 0.1
    return torch.sum(torch.abs(angle), dim=1) * zero_flag


def stand_still(
    env: BaseEnv | UltraMotionRunEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_angles: list[float] | torch.Tensor = None,
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    if target_angles is None:
        target_angles = asset.data.default_joint_pos
    elif isinstance(target_angles, list):
        target_angles = torch.tensor(target_angles, device=asset.data.joint_pos.device)
        target_angles = target_angles.unsqueeze(0).expand_as(asset.data.joint_pos)

    # print(env.robot.data.joint_names)

    angle = asset.data.joint_pos - target_angles
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return torch.exp(-torch.sum(torch.square(angle), dim=1)) * mask


def base_height(env: BaseEnv, std: float, target_height: float, style_mask: list[int] | None = None) -> torch.Tensor:
    root_height = env.robot.data.root_pos_w[:, 2]
    height_error = torch.square(root_height - target_height)
    reward_height = torch.exp(-height_error / (std**2))
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()

    return reward_height * mask


def base_height_neg(
    env: BaseEnv | UltraMotionRunEnv, target_height: float, style_mask: list[int] | None = None) -> torch.Tensor:
    root_height = env.robot.data.root_pos_w[:, 2]
    height_error = (root_height - target_height).clamp(max=0.0, min=-0.5)
    penalty = torch.abs(height_error)
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return penalty * mask


def body_orientation_exp(
    env: BaseEnv | UltraMotionRunEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_apply_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W
    )
    
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    
    return torch.exp(-100 * torch.sum(torch.square(body_orientation[:, :2]), dim=1)) * mask


def body_orientation_l2(
    env: BaseEnv | UltraMotionRunEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), style_mask: list[int] | None = None
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_apply_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W
    )
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
        
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1) * mask


def base_pitch_speed_neg(
    env: BaseEnv | UltraMotionRunEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cmd_scale_max: float = 8.0,
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    """Penalize base pitch magnitude scaled by forward command.

    ``cmd_scale_max`` caps the command-magnitude weighting so the penalty saturates at
    high commanded speeds (default 8.0 keeps the original ultra88p behavior; tune lower
    e.g. 5.0 for slower robots like omni).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    pitch_error = quat_to_euler_xyz(asset.data.root_quat_w)[:, 1]
    penalty = torch.abs(pitch_error)
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    cmd_scale = torch.abs(env.command_generator.command[:, 0]).clip(max=cmd_scale_max)
    return penalty * cmd_scale * mask


def base_yaw_speed_neg(
    env: BaseEnv | UltraMotionRunEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cmd_scale_max: float = 8.0,
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    """Penalize base yaw magnitude scaled by forward command.

    ``cmd_scale_max`` caps the command-magnitude weighting (default 8.0 = ultra88p
    behavior; lower it for slower robots).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    yaw_error = quat_to_euler_xyz(asset.data.root_quat_w)[:, 2]
    penalty = torch.abs(yaw_error)
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    cmd_scale = torch.abs(env.command_generator.command[:, 0]).clip(max=cmd_scale_max)
    return penalty * cmd_scale * mask


def feet_orientation_l2(
    env: BaseEnv | UltraMotionRunEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    gravity_vec = asset.data.GRAVITY_VEC_W.unsqueeze(1)

    body_orientation = math_utils.quat_apply_inverse(body_quat, gravity_vec)
    l2 = torch.sum(torch.square(body_orientation[:, :, :2]), dim=-1)
    zero_flag = (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) < 0.1
    return l2.sum(dim=1) * zero_flag


def feet_stumble(env: BaseEnv | UltraMotionRunEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )


def feet_too_near_humanoid(
    env: BaseEnv | UltraMotionRunEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


# Regularization Reward
def ankle_torque(env: UltraMotionRunEnv) -> torch.Tensor:
    """Penalize large torques on the ankle joints."""
    return torch.sum(torch.square(env.robot.data.applied_torque[:, env.ankle_joint_ids]), dim=1)


def ankle_action(env: UltraMotionRunEnv) -> torch.Tensor:
    """Penalize ankle joint actions."""
    return torch.sum(torch.abs(env.action[:, env.ankle_joint_ids]), dim=1)


def ankle_pitch_pos_limits_neg(
    env: UltraMotionRunEnv,
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    """Penalize ankle pitch joint positions beyond soft limits."""
    asset: Articulation = env.scene["robot"]
    ankle_ids = env.ankle_joint_ids
    joint_pos = asset.data.joint_pos[:, ankle_ids]
    pos_limits = asset.data.soft_joint_pos_limits[:, ankle_ids]
    out_of_limits = -(joint_pos - pos_limits[:, :, 0]).clip(max=0.0)
    out_of_limits += (joint_pos - pos_limits[:, :, 1]).clip(min=0.0)
    penalty = torch.sum(out_of_limits, dim=1)
    if style_mask is None:
        mask = torch.ones_like(env.style_ids, dtype=torch.float32)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return penalty * mask


def ankle_roll_action(env: UltraMotionRunEnv) -> torch.Tensor:
    # 用 env.ankle_roll_ids (按名字索引, 长度=2 [左, 右]) 而非 ankle_joint_ids[2:4] 的位置索引,
    # 这样不论 ankle_joint_ids 内部排布如何 reward 都正确.
    return torch.sum(torch.abs(env.action[:, env.ankle_roll_ids]), dim=1)


def hip_roll_action(env: UltraMotionRunEnv) -> torch.Tensor:
    """Penalize hip roll joint actions."""
    # 用 env.hip_roll_ids 而非 left_leg_ids[0]/right_leg_ids[0]: 后者在 omni_7dof
    # ([hip_pitch, hip_roll, hip_yaw, ...]) 上会指向 hip_pitch, 完全错位.
    return torch.sum(torch.abs(env.action[:, env.hip_roll_ids]), dim=1)


def hip_roll_action_air_neg(
    env: BaseEnv | UltraMotionRunEnv,
    delta_t: float = 0.02,
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    """Penalize hip roll positions during swing phase."""
    left_frc_clock = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[0]
    right_frc_clock = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[0]
    penalty = (
        left_frc_clock * torch.abs(env.action[:, env.hip_roll_ids[0]])
        + right_frc_clock * torch.abs(env.action[:, env.hip_roll_ids[1]])
    )
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    cmd_scale = torch.abs(env.command_generator.command[:, 0])
    return penalty * cmd_scale * mask




def hip_yaw_action(
    env: UltraMotionRunEnv,
    yaw_cmd_relax_at: float = 0.5,
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    """Penalize hip yaw joint actions; relaxes as yaw velocity command increases.

    ``penalty = sum(|a_hip_yaw|) * (1 - clamp(|cmd_yaw| / yaw_cmd_relax_at, 0, 1))``.
    """
    penalty = torch.sum(torch.abs(env.action[:, env.hip_yaw_ids]), dim=1)

    # print("hip_yaw_action", env.action[:, env.hip_yaw_ids])

    cmd_yaw = torch.abs(env.command_generator.command[:, 2])
    relax = (cmd_yaw / yaw_cmd_relax_at).clamp(0.0, 1.0)
    penalty = penalty * (1.0 - relax)

    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return penalty * mask


def hip_yaw_pos_neg(
    env: UltraMotionRunEnv,
    yaw_cmd_relax_at: float = 0.5,
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    """Penalize hip yaw joint positions; relaxes as yaw velocity command increases.

    ``penalty = sum(|q_hip_yaw|) * (1 - clamp(|cmd_yaw| / yaw_cmd_relax_at, 0, 1))``.
    """
    asset: Articulation = env.scene["robot"]
    penalty = torch.sum(torch.abs(asset.data.joint_pos[:, env.hip_yaw_ids]), dim=1)


    cmd_yaw = torch.abs(env.command_generator.command[:, 2])
    relax = (cmd_yaw / yaw_cmd_relax_at).clamp(0.0, 1.0)
    penalty = penalty * (1.0 - relax)

    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return penalty * mask


def hip_yaw_vel_neg(
    env: BaseEnv | UltraMotionRunEnv,
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    """Penalize hip yaw velocities based on commands."""
    asset: Articulation = env.scene["robot"]
    hip_yaw_ids, _ = asset.find_joints(name_keys=["hip_yaw_l_joint", "hip_yaw_r_joint"], preserve_order=True)
    penalty = torch.sum(torch.square(asset.data.joint_vel[:, hip_yaw_ids]), dim=1)
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    cmd_scale = (1.0 - torch.abs(env.command_generator.command[:, 2]).clip(max=1.0)) * torch.abs(env.command_generator.command[:, 0])
    return penalty * cmd_scale * mask


def feet_y_distance(env: UltraMotionRunEnv, distance: float = 0.2, style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize foot y-distance when the commanded y-velocity is low, to maintain a reasonable spacing."""
    leftfoot = env.robot.data.body_pos_w[:, env.feet_body_ids[0], :] - env.robot.data.root_link_pos_w[:, :]
    rightfoot = env.robot.data.body_pos_w[:, env.feet_body_ids[1], :] - env.robot.data.root_link_pos_w[:, :]
    leftfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(env.robot.data.root_link_quat_w[:, :]), leftfoot)
    rightfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(env.robot.data.root_link_quat_w[:, :]), rightfoot)
    y_distance_b = torch.abs(leftfoot_b[:, 1] - rightfoot_b[:, 1] - distance)
    y_vel_flag = torch.abs(env.command_generator.command[:, 1]) < 0.1
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return y_distance_b * y_vel_flag * mask


def body_y_distance_speed(env: UltraMotionRunEnv, asset_cfg: SceneEntityCfg, distance: float = 0.2, style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize foot y-distance when running, to maintain a reasonable spacing."""
    leftfoot = env.robot.data.body_pos_w[:, asset_cfg.body_ids[0], :] - env.robot.data.root_link_pos_w[:, :]
    rightfoot = env.robot.data.body_pos_w[:, asset_cfg.body_ids[1], :] - env.robot.data.root_link_pos_w[:, :]
    leftfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(env.robot.data.root_link_quat_w[:, :]), leftfoot)
    rightfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(env.robot.data.root_link_quat_w[:, :]), rightfoot)
    y_distance_b = torch.abs(leftfoot_b[:, 1] - rightfoot_b[:, 1] - distance)
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    cmd_scale = torch.abs(env.command_generator.command[:, 0])
    return y_distance_b * cmd_scale * mask


def body_y_distance_in_range(
    env: UltraMotionRunEnv, 
    asset_cfg: SceneEntityCfg, 
    min_dist: float, 
    max_dist: float, 
    axis_index: int = 1,
    constant_reward: float = 0.1,
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    """Penalize inappropriate lateral distance between feet."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w
    root_quat = asset.data.root_quat_w
    asset_pos_world = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    # asset_pos_body = quat_rotate_inverse(yaw_quat(root_quat.unsqueeze(1)), asset_pos_world - root_pos.unsqueeze(1))
    yaw_quat_only = math_utils.yaw_quat(root_quat)
    yaw_quat_expanded = yaw_quat_only.unsqueeze(1).expand(-1, asset_pos_world.shape[1], -1)
    asset_pos_body = math_utils.quat_apply_inverse(yaw_quat_expanded, asset_pos_world - root_pos.unsqueeze(1))
    asset_dist = torch.abs(asset_pos_body[:, 0, axis_index] - asset_pos_body[:, 1, axis_index]).unsqueeze(1)

    dist = torch.where(
        asset_dist < min_dist, 
        torch.abs(asset_dist - min_dist), 
        torch.where(
            asset_dist > max_dist, 
            torch.abs(asset_dist - max_dist), 
            torch.zeros_like(asset_dist) - constant_reward
        )
    )
    penalty = torch.min(dist, dim=1).values
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return penalty * mask


# Periodic gait-based reward function
def gait_clock(phase, air_ratio, delta_t):
    """
    Generate periodic gait clock signals for foot swing and stance phases.

    This function constructs two phase-dependent signals:
    - `I_frc`: active during swing phase (used for penalizing ground force)
    - `I_spd`: active during stance phase (used for penalizing foot speed)

    Transitions between swing and stance are smoothed within a margin of `delta_t`
    to create differentiable transitions.

    Parameters
    ----------
    phase : torch.Tensor
        Normalized gait phase in [0, 1], shape: [num_envs].
    air_ratio : torch.Tensor
        Proportion of the gait cycle spent in swing phase, shape: [num_envs].
    delta_t : float
        Transition width around phase boundaries for smooth interpolation.

    Returns
    -------
    I_frc : torch.Tensor
        Gait-based swing-phase clock signal, range [0, 1], shape: [num_envs].
    I_spd : torch.Tensor
        Gait-based stance-phase clock signal, range [0, 1], shape: [num_envs].

    Notes
    -----
    - The transitions at the boundaries (e.g., swing→stance) are linear interpolations.
    - Used in reward shaping to associate expected behavior with gait phases.
    """
    swing_flag = (phase >= delta_t) & (phase <= (air_ratio - delta_t))
    stand_flag = (phase >= (air_ratio + delta_t)) & (phase <= (1 - delta_t))

    trans_flag1 = phase < delta_t
    trans_flag2 = (phase > (air_ratio - delta_t)) & (phase < (air_ratio + delta_t))
    trans_flag3 = phase > (1 - delta_t)

    I_frc = (
        1.0 * swing_flag
        + (0.5 + phase / (2 * delta_t)) * trans_flag1
        - (phase - air_ratio - delta_t) / (2.0 * delta_t) * trans_flag2
        + 0.0 * stand_flag
        + (phase - 1 + delta_t) / (2 * delta_t) * trans_flag3
    )
    I_spd = 1.0 - I_frc
    return I_frc, I_spd


def gait_feet_frc_perio(env: UltraMotionRunEnv, delta_t: float = 0.02, style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize foot force during the swing phase of the gait."""
    left_frc_swing_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[0]
    right_frc_swing_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[0]
    left_frc_score = left_frc_swing_mask * (torch.exp(-200 * torch.square(env.avg_feet_force_per_step[:, 0])))
    right_frc_score = right_frc_swing_mask * (torch.exp(-200 * torch.square(env.avg_feet_force_per_step[:, 1])))
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return (left_frc_score + right_frc_score) * mask


def gait_feet_spd_perio(env: UltraMotionRunEnv, delta_t: float = 0.02, style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize foot speed during the support phase of the gait."""
    left_spd_support_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[1]
    right_spd_support_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[1]
    left_spd_score = left_spd_support_mask * (torch.exp(-100 * torch.square(env.avg_feet_speed_per_step[:, 0])))
    right_spd_score = right_spd_support_mask * (torch.exp(-100 * torch.square(env.avg_feet_speed_per_step[:, 1])))
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return (left_spd_score + right_spd_score) * mask


def gait_feet_frc_support_perio(
    env: UltraMotionRunEnv, delta_t: float = 0.02, style_mask: list[int] | None = None
) -> torch.Tensor:
    """Reward that promotes proper support force during stance (support) phase."""
    left_frc_support_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[1]
    right_frc_support_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[1]
    left_frc_score = left_frc_support_mask * (1 - torch.exp(-10 * torch.square(env.avg_feet_force_per_step[:, 0])))
    right_frc_score = right_frc_support_mask * (1 - torch.exp(-10 * torch.square(env.avg_feet_force_per_step[:, 1])))
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return (left_frc_score + right_frc_score) * mask


def z_vel_run(env: BaseEnv | UltraMotionRunEnv, style_mask: list[int] | None = None) -> torch.Tensor:
    """Reward upward base velocity during stance-to-flight transition."""
    asset: Articulation = env.scene["robot"]
    phase_ratio_0 = torch.clamp(env.phase_ratio[:, 0], min=1e-6)
    phase_ratio_1 = torch.clamp(env.phase_ratio[:, 1], min=1e-6)

    s0 = env.gait_phase[:, 0] / phase_ratio_0 - 1.0
    s01 = s0 > 0
    s0 = s0 * (s0 > 0.0)
    denom_0 = (1.0 - phase_ratio_0) / phase_ratio_0
    denom_0 = torch.clamp(denom_0, min=1e-6)
    s0 = s0 / denom_0
    s0 = s01.float() * (s0 - 0.5)
    left_spd_score = s0 * asset.data.root_lin_vel_b[:, 2]

    s1 = env.gait_phase[:, 1] / phase_ratio_1 - 1.0
    s11 = s1 > 0
    s1 = s1 * (s1 > 0.0)
    denom_1 = (1.0 - phase_ratio_1) / phase_ratio_1
    denom_1 = torch.clamp(denom_1, min=1e-6)
    s1 = s1 / denom_1
    s1 = s11.float() * (s1 - 0.5)
    right_spd_score = s1 * asset.data.root_lin_vel_b[:, 2]

    reward = 5.0 * (left_spd_score + right_spd_score)
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return reward * mask

def gait_swing_symmetric(
    env: BaseEnv | UltraMotionRunEnv, delta_t: float = 0.02, style_mask: list[int] | None = None
) -> torch.Tensor:
    """Reward symmetric foot placement during double-air (both-swing) phase."""
    asset: Articulation = env.scene["robot"]
    left_frc_clock = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[0].unsqueeze(1)
    right_frc_clock = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[0].unsqueeze(1)

    left_foot_pos = asset.data.body_state_w[:, env.feet_body_ids[0], :3] - asset.data.root_state_w[:, :3]
    right_foot_pos = asset.data.body_state_w[:, env.feet_body_ids[1], :3] - asset.data.root_state_w[:, :3]

    root_quat = asset.data.root_state_w[:, 3:7]
    feet_distance = left_foot_pos - right_foot_pos
    feet_distance = math_utils.quat_apply_inverse(root_quat, feet_distance)

    db_air_flag = (left_frc_clock > 0.9) * (right_frc_clock > 0.9)

    # Lazy-init buffers on env
    if not hasattr(env, "feet_distance_buffer"):
        env.feet_distance_buffer = torch.zeros_like(feet_distance)
    if not hasattr(env, "last_feet_distance_buffer"):
        env.last_feet_distance_buffer = torch.zeros_like(feet_distance)

    score = torch.abs(feet_distance[:, 0:1] + env.last_feet_distance_buffer[:, 0:1])
    env.feet_distance_buffer = db_air_flag * feet_distance + (~db_air_flag) * env.feet_distance_buffer
    env.last_feet_distance_buffer = (~db_air_flag) * env.feet_distance_buffer + db_air_flag * env.last_feet_distance_buffer

    reward = (db_air_flag * torch.exp(-10 * score)).squeeze(1)
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return reward * mask


def knee_vel_swing_neg(
    env: BaseEnv | UltraMotionRunEnv,
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    """Penalize knee joint velocities during swing phase."""
    asset: Articulation = env.scene["robot"]
    phase_ratio_0 = torch.clamp(env.phase_ratio[:, 0], min=1e-6)
    phase_ratio_1 = torch.clamp(env.phase_ratio[:, 1], min=1e-6)
    s0 = env.gait_phase[:, 0] / phase_ratio_0
    s00 = s0 < 0.5
    s01 = (s0 > 0.5) & (s0 < 1.0)
    s1 = env.gait_phase[:, 1] / phase_ratio_1
    s10 = s1 < 0.5
    s11 = (s1 > 0.5) & (s1 < 1.0)

    knee_ids, _ = asset.find_joints(name_keys=["knee_pitch_l_joint", "knee_pitch_r_joint"], preserve_order=True)
    knee_vel_l = (asset.data.joint_vel[:, knee_ids[0]] * s00).clip(max=0.0) - (
        asset.data.joint_vel[:, knee_ids[0]] * s01
    ).clip(min=0.0)
    knee_vel_r = (asset.data.joint_vel[:, knee_ids[1]] * s10).clip(max=0.0) - (
        asset.data.joint_vel[:, knee_ids[1]] * s11
    ).clip(min=0.0)

    penalty = torch.abs(knee_vel_l + knee_vel_r)
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    cmd_scale = torch.abs(env.command_generator.command[:, 0])
    return penalty * cmd_scale * mask


def feet_swing_height_neg(
    env: BaseEnv | UltraMotionRunEnv,
    target_height_diff: float = 0.17,
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    """Penalize insufficient foot swing height during swing mid-phase."""
    asset: Articulation = env.scene["robot"]
    phase_ratio_0 = torch.clamp(env.phase_ratio[:, 0], min=1e-6)
    phase_ratio_1 = torch.clamp(env.phase_ratio[:, 1], min=1e-6)
    s0 = env.gait_phase[:, 0] / phase_ratio_0
    s01 = (s0 > 0.3) & (s0 < 0.6)
    s1 = env.gait_phase[:, 1] / phase_ratio_1
    s11 = (s1 > 0.3) & (s1 < 0.6)

    left_foot_pos = asset.data.body_state_w[:, env.feet_body_ids[0], 2]
    right_foot_pos = asset.data.body_state_w[:, env.feet_body_ids[1], 2]

    left_score = s01 * (left_foot_pos - right_foot_pos - target_height_diff).clip(max=0.0)
    right_score = s11 * (right_foot_pos - left_foot_pos - target_height_diff).clip(max=0.0)

    penalty = torch.abs(left_score + right_score)
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    cmd_scale = torch.abs(env.command_generator.command[:, 0]).clip(max=3.0)
    return penalty * cmd_scale * mask


def straighten_knee_neg(env: BaseEnv | UltraMotionRunEnv, style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize knees not straightened during stance phase."""
    asset: Articulation = env.scene["robot"]
    phase_ratio_0 = torch.clamp(env.phase_ratio[:, 0], min=1e-6)
    phase_ratio_1 = torch.clamp(env.phase_ratio[:, 1], min=1e-6)
    s0 = env.gait_phase[:, 0] / phase_ratio_0 - 0.5
    s0 = s0.clip(min=0.0, max=1.0) * (s0 < 0.6)
    s0 = s0.clip(min=0.0, max=0.5)
    s1 = env.gait_phase[:, 1] / phase_ratio_1 - 0.5
    s1 = s1.clip(min=0.0, max=1.0) * (s1 < 0.6)
    s1 = s1.clip(min=0.0, max=0.5)

    knee_ids, _ = asset.find_joints(name_keys=["knee_pitch_l_joint", "knee_pitch_r_joint"], preserve_order=True)
    left_score = torch.abs(asset.data.joint_pos[:, knee_ids[0]] - 0.3) * s0
    right_score = torch.abs(asset.data.joint_pos[:, knee_ids[1]] - 0.3) * s1
    penalty = left_score + right_score
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return penalty * mask


def feet_speed_z_diff(
    env: BaseEnv | UltraMotionRunEnv, sensor_cfg: SceneEntityCfg, style_mask: list[int] | None = None
) -> torch.Tensor:
    """Reward low foot vertical speed when near ground and not in contact."""
    asset: Articulation = env.scene["robot"]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    left_foot_height = asset.data.body_state_w[:, env.feet_body_ids[0], 2]
    right_foot_height = asset.data.body_state_w[:, env.feet_body_ids[1], 2]

    left_foot_v = asset.data.body_lin_vel_w[:, env.feet_body_ids[0], 2]
    right_foot_v = asset.data.body_lin_vel_w[:, env.feet_body_ids[1], 2]

    rew_left_fspd = torch.exp(-10.0 * torch.square(left_foot_v))
    rew_left_fspd *= left_foot_height < 0.05

    rew_right_fspd = torch.exp(-10.0 * torch.square(right_foot_v))
    rew_right_fspd *= right_foot_height < 0.05

    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    contact = contact_forces > 1.0

    # Lazy-init last contact buffer on env
    if not hasattr(env, "last_contacts_buffer"):
        env.last_contacts_buffer = torch.zeros_like(contact)

    contact_filt = torch.logical_or(contact, env.last_contacts_buffer)
    env.last_contacts_buffer = contact

    rew_left_fspd *= ~contact_filt[:, 0]
    rew_right_fspd *= ~contact_filt[:, 1]

    penalty = rew_left_fspd + rew_right_fspd
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return penalty * mask


def gait_feet_swing_neg(
    env: BaseEnv | UltraMotionRunEnv, delta_t: float = 0.02, style_mask: list[int] | None = None
) -> torch.Tensor:
    """Penalize foot vertical velocity during swing (near stance) phase."""
    asset: Articulation = env.scene["robot"]
    phase_ratio_0 = torch.clamp(env.phase_ratio[:, 0], min=1e-6)
    phase_ratio_1 = torch.clamp(env.phase_ratio[:, 1], min=1e-6)

    s0 = env.gait_phase[:, 0] / phase_ratio_0 - 0.5
    s0 = s0.clip(min=0.0, max=1.0) * (s0 < 0.6)
    s0 = s0.clip(min=0.0, max=0.5)
    s1 = env.gait_phase[:, 1] / phase_ratio_1 - 0.5
    s1 = s1.clip(min=0.0, max=1.0) * (s1 < 0.6)
    s1 = s1.clip(min=0.0, max=0.5)

    left_foot_v = asset.data.body_lin_vel_w[:, env.feet_body_ids[0], 2]
    right_foot_v = asset.data.body_lin_vel_w[:, env.feet_body_ids[1], 2]
    left_score = 16 * s0 * s0 * left_foot_v * left_foot_v
    right_score = 16 * s1 * s1 * right_foot_v * right_foot_v

    penalty = (left_score + right_score).clip(max=1.0)
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return penalty * mask


def gait_feet_distance(
    env: UltraMotionRunEnv,
    target_y: float = 0.15,
    lateral_cmd_relax_at: float = 0.5,
    speed_scale: float = 1.0,
    speed_max: float = 15.0,
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    """Penalize lateral foot spacing in robot body frame (root link frame).

    ``lateral_span = left_y - right_y`` in body frame (expected ~``2 * target_y``).
    Penalty is squared error: ``(lateral_span - 2 * target_y)^2``.
    Scales by ``(1 - clamp(|cmd_vy| / lateral_cmd_relax_at, 0, 1))`` so larger lateral
    velocity commands relax the penalty. Penalty also scales with forward command speed:
    ``speed_scale * |cmd_vx|``. Use with negative reward weight in cfg.
    """
    asset: Articulation = env.scene["robot"]
    left_foot_pos = asset.data.body_state_w[:, env.feet_body_ids[0], :3] - asset.data.root_state_w[:, :3]
    right_foot_pos = asset.data.body_state_w[:, env.feet_body_ids[1], :3] - asset.data.root_state_w[:, :3]

    root_quat = asset.data.root_quat_w
    left_foot_pos = math_utils.quat_apply_inverse(root_quat, left_foot_pos)
    right_foot_pos = math_utils.quat_apply_inverse(root_quat, right_foot_pos)

    lateral_span = left_foot_pos[:, 1] - right_foot_pos[:, 1]
    target_lateral = 2.0 * target_y
    penalty = torch.abs(lateral_span - target_lateral)

    cmd_vy = torch.abs(env.command_generator.command[:, 1])
    relax = (cmd_vy / lateral_cmd_relax_at).clamp(0.0, 1.0)
    penalty = penalty * (1.0 - relax)

    cmd_vx = torch.abs(env.command_generator.command[:, 0]).clamp(max=speed_max)
    penalty = penalty * speed_scale * cmd_vx

    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return penalty * mask


def gait_feet_y_distance_neg(
    env: UltraMotionRunEnv,
    command_name: str = "base_velocity",
    target_y: float = 0.15,
    speed_adaptive: bool = False,
    speed_scale: float = 0.02,
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    """Penalize feet y-distance from target.

    When ``speed_adaptive=True``, the target widens with forward speed:
    ``effective_target = target_y + speed_scale * |cmd_vx|``.
    This allows a wider stance at higher running speeds (biomechanically natural).
    """
    asset: Articulation = env.scene["robot"]
    left_foot_pos = asset.data.body_state_w[:, env.feet_body_ids[0], :3] - asset.data.root_state_w[:, :3]
    right_foot_pos = asset.data.body_state_w[:, env.feet_body_ids[1], :3] - asset.data.root_state_w[:, :3]

    root_quat = asset.data.root_state_w[:, 3:7]
    root_euler = quat_to_euler_xyz(root_quat)
    root_euler_no_yaw = torch.stack(
        [torch.zeros_like(root_euler[:, 0]), torch.zeros_like(root_euler[:, 1]), root_euler[:, 2]], dim=1
    )
    quat_without_yaw = euler_xyz_to_quat(root_euler_no_yaw)

    left_foot_pos = math_utils.quat_apply_inverse(quat_without_yaw, left_foot_pos)
    right_foot_pos = math_utils.quat_apply_inverse(quat_without_yaw, right_foot_pos)

    if speed_adaptive:
        cmd_vx = torch.abs(env.command_generator.command[:, 0]).clamp(max=4.0)
        effective_target = target_y + speed_scale * cmd_vx
    else:
        effective_target = target_y

    score_left = 2.0 * torch.abs(effective_target - left_foot_pos[:, 1])
    score_right = 2.0 * torch.abs(effective_target + right_foot_pos[:, 1])

    penalty = score_left + score_right
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    cmd_scale = 0.5 - torch.abs(env.command_generator.command[:, 1]).clip(max=0.5)
    return penalty * cmd_scale * mask


def knee_y_distance(
    env: UltraMotionRunEnv,
    asset_cfg: SceneEntityCfg,
    target_y: float = 0.15,
    speed_scale: float = 1.0,
    speed_max: float = 15.0,
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    """Penalize knee lateral spacing from target in robot body frame (root link frame).

    Left knee expected at ``+target_y``, right at ``-target_y``.
    Penalty scales with forward command speed: ``|cmd_vx|`` larger → stronger penalty.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    left_knee_pos = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3] - asset.data.root_state_w[:, :3]
    right_knee_pos = asset.data.body_state_w[:, asset_cfg.body_ids[1], :3] - asset.data.root_state_w[:, :3]

    root_quat = asset.data.root_quat_w
    left_knee_pos = math_utils.quat_apply_inverse(root_quat, left_knee_pos)
    right_knee_pos = math_utils.quat_apply_inverse(root_quat, right_knee_pos)

    score_left = 2.0 * torch.abs(target_y - left_knee_pos[:, 1])
    score_right = 2.0 * torch.abs(target_y + right_knee_pos[:, 1])
    penalty = score_left + score_right

    cmd_vx = torch.abs(env.command_generator.command[:, 0]).clamp(max=speed_max)
    penalty = penalty * speed_scale * cmd_vx

    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    cmd_scale = 0.5 - torch.abs(env.command_generator.command[:, 1]).clip(max=0.5)
    return penalty * cmd_scale * mask


def gait_feet_x_distance_neg(
    env: UltraMotionRunEnv, command_name: str = "base_velocity", style_mask: list[int] | None = None
) -> torch.Tensor:
    """Penalize feet x-distance during mid-air phase."""
    asset: Articulation = env.scene["robot"]
    left_foot_pos = asset.data.body_state_w[:, env.feet_body_ids[0], :3] - asset.data.root_state_w[:, :3]
    right_foot_pos = asset.data.body_state_w[:, env.feet_body_ids[1], :3] - asset.data.root_state_w[:, :3]

    root_quat = asset.data.root_state_w[:, 3:7]
    root_euler = quat_to_euler_xyz(root_quat)
    root_euler_no_yaw = torch.stack(
        [torch.zeros_like(root_euler[:, 0]), torch.zeros_like(root_euler[:, 1]), root_euler[:, 2]], dim=1
    )
    quat_without_yaw = euler_xyz_to_quat(root_euler_no_yaw)

    left_foot_pos = math_utils.quat_apply_inverse(quat_without_yaw, left_foot_pos)
    right_foot_pos = math_utils.quat_apply_inverse(quat_without_yaw, right_foot_pos)

    phase_ratio_0 = torch.clamp(env.phase_ratio[:, 0], min=1e-6)
    phase_ratio_1 = torch.clamp(env.phase_ratio[:, 1], min=1e-6)
    mid_air_flag = (torch.abs(env.gait_phase[:, 0] / phase_ratio_0 - 0.5) < 0.02) + (
        torch.abs(env.gait_phase[:, 1] / phase_ratio_1 - 0.5) < 0.02
    )

    penalty = (
        2.0
        * mid_air_flag.float()
        * (torch.abs(left_foot_pos[:, 0]) + torch.abs(right_foot_pos[:, 0]))
    )

    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    cmd_scale = 0.5 - torch.abs(env.command_generator.command[:, 0]).clip(max=0.5)
    return penalty * cmd_scale * mask


def gait_feet_distance_collision_neg(
    env: UltraMotionRunEnv,
    threshold_y: float = 0.14,
    threshold_x: float = 0.24,
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    """Penalize feet being too close together (collision risk)."""
    asset: Articulation = env.scene["robot"]
    left_foot_pos = asset.data.body_state_w[:, env.feet_body_ids[0], :3] - asset.data.root_state_w[:, :3]
    right_foot_pos = asset.data.body_state_w[:, env.feet_body_ids[1], :3] - asset.data.root_state_w[:, :3]

    root_quat = asset.data.root_state_w[:, 3:7]
    left_foot_pos = math_utils.quat_apply_inverse(root_quat, left_foot_pos)
    right_foot_pos = math_utils.quat_apply_inverse(root_quat, right_foot_pos)

    feet_distance_y = torch.abs(left_foot_pos[:, 1] - right_foot_pos[:, 1])
    feet_distance_x = torch.abs(left_foot_pos[:, 0] - right_foot_pos[:, 0])
    collision_flag = (feet_distance_y < threshold_y) * (feet_distance_x < threshold_x)
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return collision_flag.float() * mask


def feet_collision(
    env: UltraMotionRunEnv,
    threshold_y: float = 0.14,
    threshold_x: float = 0.24,
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    """Binary penalty when feet are too close in body frame (2026gym leg_collision)."""
    return gait_feet_distance_collision_neg(
        env, threshold_y=threshold_y, threshold_x=threshold_x, style_mask=style_mask
    )


def body_contact_neg(
    env: BaseEnv | UltraMotionRunEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    """Penalize contact on non-foot bodies (knee, hip, base, etc.)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    penalty = torch.sum(is_contact, dim=1).float()
    if style_mask is None:
        mask = torch.ones_like(env.style_ids, dtype=torch.float32)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return penalty * mask


def gait_feet_contact_vel_neg(
    env: UltraMotionRunEnv, style_mask: list[int] | None = None
) -> torch.Tensor:
    """Penalize foot velocity during contact transition."""
    asset: Articulation = env.scene["robot"]
    foot_v = torch.abs(asset.data.body_lin_vel_w[:, env.feet_body_ids, :])
    foot_v_norm = torch.norm(foot_v, dim=-1)

    s0 = env.gait_phase[:, :] - env.phase_ratio[:, :] > -0.1
    s1 = env.gait_phase[:, :] - env.phase_ratio[:, :] < 0.2

    penalty = s0 * s1 * foot_v_norm
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return torch.norm(penalty, dim=-1) * mask


def contact_impact_vel(
    env: UltraMotionRunEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    style_mask: list[int] | None = None,
    pre_contact_policy_steps: int = 2,
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize foot speed at touchdown and before landing (from force history).

    Touchdown: rising edge on the last two physics samples in ``net_forces_w_history``.

    Pre-landing (``pre_contact_policy_steps > 0``): foot still in air on the newest sample,
    and the trailing ``decimation * pre_contact_policy_steps`` physics samples are also in air
    (about to touch down within ~N policy steps). Excludes the toe-off pattern (oldest buffer
    slot in contact while the rest are in air). Optionally requires an earlier contact in the
    older part of the buffer (same swing) when history is long enough.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    feet_ids = asset_cfg.body_ids
    fz = contact_sensor.data.net_forces_w_history[:, :, feet_ids, 2]
    contact = fz > contact_force_threshold
    foot_spd = torch.norm(asset.data.body_lin_vel_w[:, feet_ids, :], dim=-1)

    if contact.shape[1] >= 2:
        rising_edge = contact[:, -1] & (~contact[:, -2])
    else:
        rising_edge = contact[:, -1]
    impact_score = torch.sum(rising_edge.float() * foot_spd, dim=1)

    pre_score = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    if pre_contact_policy_steps > 0:
        substeps = max(1, int(round(float(env.step_dt) / float(env.physics_dt))))
        pre_phys = substeps * pre_contact_policy_steps
        hist_len = contact.shape[1]
        recent_len = pre_phys + 1
        if hist_len >= recent_len:
            # Landing approach: in air now and over the recent physics window
            recent_air = (~contact[:, -recent_len:]).all(dim=1)
            # Toe-off (wrong phase): oldest buffered sample on ground, all newer in air
            toe_off_in_buffer = contact[:, 0] & (~contact[:, 1:]).all(dim=1)
            pre_contact = recent_air & ~toe_off_in_buffer
            if hist_len > recent_len:
                prior_touch_in_buf = contact[:, : hist_len - recent_len].any(dim=1)
                pre_contact = pre_contact & prior_touch_in_buf
            pre_score = torch.sum(pre_contact.float() * foot_spd, dim=1)

    score = impact_score + pre_score
    cmd_x = env.command_generator.command[:, 0]
    vmax = max(float(env.cfg.commands.ranges.lin_vel_x[1]), 1e-3)
    cmd_w = 0.5 + (cmd_x - 3.0).clamp(min=0.0, max=vmax) / vmax
    if style_mask is None:
        mask = torch.ones_like(env.style_ids, dtype=torch.float32)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return score * cmd_w * mask


def feet_descent_proximity_neg(
    env: BaseEnv | UltraMotionRunEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="ankle_pitch.*"),
    ground_z: float = 0.0,
    foot_sole_offset: float = 0.074,
    proximity_height_scale: float = 0.05,
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    """Penalize downward foot speed; weight grows as clearance to ground shrinks.

    Per foot: ``penalty = max(0, -v_z) * exp(-clearance / proximity_height_scale)``,
    where clearance is sole height above ground (feet ray scan if available, else
    ``foot_z - ground_z - foot_sole_offset``). Sum over feet.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    feet_ids = asset_cfg.body_ids
    foot_z = asset.data.body_pos_w[:, feet_ids, 2]

    if hasattr(env, "_get_measured_feet_ground_heights") and getattr(env, "feet_height_scanners", None):
        ground_h = env._get_measured_feet_ground_heights()
        clearance = (foot_z - ground_h - foot_sole_offset).clamp(min=0.0)
    else:
        clearance = (foot_z - ground_z - foot_sole_offset).clamp(min=0.0)

    prox_w = torch.exp(-clearance / max(proximity_height_scale, 1e-4))
    descent = (-asset.data.body_lin_vel_w[:, feet_ids, 2]).clamp(min=0.0)
    penalty = torch.sum(descent * prox_w, dim=-1)

    if style_mask is None:
        mask = torch.ones_like(env.style_ids, dtype=torch.float32)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return penalty * mask


def gait_feet_force_max_neg(
    env: UltraMotionRunEnv, sensor_cfg: SceneEntityCfg, max_force: float = 1500.0, style_mask: list[int] | None = None
) -> torch.Tensor:
    """Penalize excessive foot contact forces."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    extra_frc = (contact_forces - max_force).clip(min=0.0, max=100.0)
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return torch.sum(extra_frc, dim=-1) * mask

# --------------------------------------------------------------------------- #
# Ultra88p Rewards
# --------------------------------------------------------------------------- #

def high_speed_bonus(
    env: BaseEnv | UltraMotionRunEnv,
    std: float,
    std_speed_max: float | None = None,
    speed_walk: float = 1.0,
    speed_max: float = 15.0,
    speed_threshold: float = 3.0,
    normalize_speed: bool = True,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    """高速段（cmd_x > speed_threshold）的额外速度跟踪奖励.

    ``normalize_speed`` 时将 (cmd-threshold) 归一化到 [0,1]，避免 vmax=15 时单步奖励过大。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    err = torch.abs(env.command_generator.command[:, 0] - vel_yaw[:, 0])
    std_eff = _std_scaled_by_cmd_x(env, std, std_speed_max, speed_walk, speed_max)
    cmd_excess = (env.command_generator.command[:, 0] - speed_threshold).clip(min=0.0)
    if normalize_speed:
        denom = max(speed_max - speed_threshold, 1e-3)
        cmd_excess = cmd_excess / denom
    if style_mask is None:
        mask = torch.ones_like(env.style_ids, dtype=torch.float32)
    else:
        mask = torch.isin(env.style_ids,
                          torch.tensor(style_mask, device=env.style_ids.device)).float()
    return torch.exp(-err * err / (std_eff * std_eff)) * cmd_excess * mask

def body_orientation_speed_aware(
    env: BaseEnv | UltraMotionRunEnv,
    pitch_coef: float = 0.025,
    roll_coef: float = 0.07,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    """exp 形式 orientation 跟踪, 目标随命令自适应.
    高速时允许并鼓励适度前倾 (pitch_target = pitch_coef * cmd_x)
    转弯时允许侧倾 (roll_target = roll_coef * cmd_x * cmd_yaw)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    proj_g = math_utils.quat_apply_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W
    )
    cmd_x = env.command_generator.command[:, 0].clip(min=0.0)
    cmd_yaw = env.command_generator.command[:, 2]
    pitch_target = pitch_coef * cmd_x
    roll_target = roll_coef * torch.abs(cmd_x) * cmd_yaw
    err = (pitch_target - proj_g[:, 0]).square() + (roll_target - proj_g[:, 1]).square()
    if style_mask is None:
        mask = torch.ones_like(env.style_ids, dtype=torch.float32)
    else:
        mask = torch.isin(env.style_ids,
                          torch.tensor(style_mask, device=env.style_ids.device)).float()
    return torch.exp(-100.0 * err) * mask

def swing_decel(
    env: UltraMotionRunEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    window: float = 0.10,
    descent_threshold: float = 0.5,
    speed_threshold: float = 2.0,
    speed_weight_max: float = 8.0,
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    """Penalize swing-foot descent speed in the late-swing window before touchdown.

    Within ``[phase_ratio - window, phase_ratio)`` the penalty is the foot's downward
    velocity above ``descent_threshold``, multiplied by a speed-aware weight that grows
    from 0.5 at ``speed_threshold`` to 1.0 at ``speed_threshold + speed_weight_max``.
    Default ``speed_weight_max=8.0`` matches the legacy ultra88p behavior; tune lower
    (e.g. 4.0) for slower robots like omni so the weight saturates earlier.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    feet_ids = env.feet_cfg.body_ids
    # NOTE: env.phase_ratio is a per-env tensor (num_envs, 2), not a constant;
    #       [:, 0] = left swing ratio, [:, 1] = right; differs per stand/walk/run style.
    phase_l = env.gait_phase[:, 0]
    phase_r = env.gait_phase[:, 1]
    ratio_l = env.phase_ratio[:, 0]
    ratio_r = env.phase_ratio[:, 1]
    in_window_l = (phase_l > (ratio_l - window)) & (phase_l < ratio_l)
    in_window_r = (phase_r > (ratio_r - window)) & (phase_r < ratio_r)
    foot_vz = asset.data.body_lin_vel_w[:, feet_ids, 2]
    descent_l = (-foot_vz[:, 0] - descent_threshold).clip(min=0.0)
    descent_r = (-foot_vz[:, 1] - descent_threshold).clip(min=0.0)
    penalty = in_window_l.float() * descent_l + in_window_r.float() * descent_r
    cmd_x = env.command_generator.command[:, 0].abs()
    speed_w = 0.5 + 0.5 * (cmd_x - speed_threshold).clip(min=0.0, max=speed_weight_max) / speed_weight_max
    if style_mask is None:
        mask = torch.ones_like(env.style_ids, dtype=torch.float32)
    else:
        mask = torch.isin(env.style_ids,
                          torch.tensor(style_mask, device=env.style_ids.device)).float()
    return penalty * speed_w * mask

def swing_decel_xy(
    env: UltraMotionRunEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    window: float = 0.10,
    excess_threshold: float = 0.5,
    speed_threshold: float = 2.0,
    speed_weight_max: float = 8.0,
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    """Penalize horizontal foot-relative-to-base speed in the late-swing window.

    Complements :func:`swing_decel`, reducing horizontal touchdown shock. The
    ``speed_weight_max`` cap mirrors :func:`swing_decel` (default 8.0 = ultra88p).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    feet_ids = env.feet_cfg.body_ids
    # Same convention as swing_decel: env.phase_ratio is (num_envs, 2) per-env tensor.
    phase_l = env.gait_phase[:, 0]
    phase_r = env.gait_phase[:, 1]
    ratio_l = env.phase_ratio[:, 0]
    ratio_r = env.phase_ratio[:, 1]
    in_window_l = (phase_l > (ratio_l - window)) & (phase_l < ratio_l)
    in_window_r = (phase_r > (ratio_r - window)) & (phase_r < ratio_r)
    foot_vxy = asset.data.body_lin_vel_w[:, feet_ids, :2]
    base_vxy = asset.data.root_lin_vel_w[:, :2].unsqueeze(1)
    rel_speed = torch.norm(foot_vxy - base_vxy, dim=-1)  # (N, 2)
    excess_l = (rel_speed[:, 0] - excess_threshold).clip(min=0.0)
    excess_r = (rel_speed[:, 1] - excess_threshold).clip(min=0.0)
    penalty = in_window_l.float() * excess_l + in_window_r.float() * excess_r
    cmd_x = env.command_generator.command[:, 0].abs()
    speed_w = 0.5 + 0.5 * (cmd_x - speed_threshold).clip(min=0.0, max=speed_weight_max) / speed_weight_max
    if style_mask is None:
        mask = torch.ones_like(env.style_ids, dtype=torch.float32)
    else:
        mask = torch.isin(env.style_ids,
                          torch.tensor(style_mask, device=env.style_ids.device)).float()
    return penalty * speed_w * mask

def late_touchdown(
    env: UltraMotionRunEnv,
    sensor_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    """gait_phase 已超过 phase_ratio (应该着地了) 但 contact force ~ 0 的惩罚.
    平方形式, 越超时越贵, 迫使策略尽快落脚 -> 步频更紧 -> 速度上限抬高.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = torch.max(
        torch.norm(
            contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids], dim=-1
        ), dim=1
    )[0] > contact_threshold     # (N, 2)
    no_contact = (~contact).float()
    # env.phase_ratio: (num_envs, 2), 不是 (2,) 常量; per-env per-foot.
    late_l = (env.gait_phase[:, 0] - env.phase_ratio[:, 0]).clip(min=0.0)
    late_r = (env.gait_phase[:, 1] - env.phase_ratio[:, 1]).clip(min=0.0)
    penalty = late_l.square() * no_contact[:, 0] + late_r.square() * no_contact[:, 1]
    if style_mask is None:
        mask = torch.ones_like(env.style_ids, dtype=torch.float32)
    else:
        mask = torch.isin(env.style_ids,
                          torch.tensor(style_mask, device=env.style_ids.device)).float()
    return penalty * mask

def joint_torques_speed_scaled(
    env: UltraMotionRunEnv,
    speed_max: float = 8.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    """``sum(tau^2) * speed_scale`` style penalty that fades out at higher speeds.

    ``speed_scale = 0.2 * (speed_max - |cmd_x|).clip(min=0.0)`` so for small commands
    the penalty is full strength and at ``|cmd_x| >= speed_max`` it vanishes (so the
    policy may use large torques when actually running fast). Default ``speed_max=8.0``
    keeps the legacy ultra88p behavior; lower it (e.g. 4.0) for slower robots.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    sq = torch.sum(torch.square(asset.data.applied_torque), dim=1)
    cmd_x = env.command_generator.command[:, 0].abs()
    speed_scale = 0.2 * (speed_max - cmd_x).clip(min=0.0)
    if style_mask is None:
        mask = torch.ones_like(env.style_ids, dtype=torch.float32)
    else:
        mask = torch.isin(env.style_ids,
                          torch.tensor(style_mask, device=env.style_ids.device)).float()
    return sq * speed_scale * mask


# ── xhumanoid/run (2026gym) per-joint torque & velocity penalties ──
# Weights follow ``x_humanoid_run.py``; mapped by joint name (Isaac order agnostic).

_XHUMANOID_TORQUE_WEIGHTS: dict[str, float] = {
    "hip_yaw_l_joint": 1.5,
    "hip_yaw_r_joint": 1.5,
    "hip_roll_l_joint": 1.0,
    "hip_roll_r_joint": 1.0,
    "hip_pitch_l_joint": 0.6,
    "hip_pitch_r_joint": 0.6,
    "knee_pitch_l_joint": 0.6,
    "knee_pitch_r_joint": 0.6,
    "ankle_pitch_l_joint": 5.0,
    "ankle_pitch_r_joint": 5.0,
    "waist_yaw_joint": 3.0,
    "shoulder_pitch_l_joint": 2.0,
    "shoulder_pitch_r_joint": 2.0,
}

_XHUMANOID_VEL_WEIGHTS: dict[str, float] = {
    "hip_yaw_l_joint": 1.0,
    "hip_yaw_r_joint": 1.0,
    "hip_roll_l_joint": 1.0,
    "hip_roll_r_joint": 1.0,
    "hip_pitch_l_joint": 0.3,
    "hip_pitch_r_joint": 0.3,
    "knee_pitch_l_joint": 0.2,
    "knee_pitch_r_joint": 0.2,
    "ankle_pitch_l_joint": 1.0,
    "ankle_pitch_r_joint": 1.0,
    "waist_yaw_joint": 3.0,
    "shoulder_pitch_l_joint": 0.7,
    "shoulder_pitch_r_joint": 0.7,
}

# 2026gym x_humanoid_run ``action_smooth_first/second`` (Isaac joint names).
_XHUMANOID_ACTION_RATE_WEIGHTS: dict[str, float] = {
    "hip_yaw_l_joint": 3.0,
    "hip_yaw_r_joint": 3.0,
    "hip_roll_l_joint": 3.0,
    "hip_roll_r_joint": 3.0,
    "hip_pitch_l_joint": 1.0,
    "hip_pitch_r_joint": 1.0,
    "knee_pitch_l_joint": 1.0,
    "knee_pitch_r_joint": 1.0,
    "ankle_pitch_l_joint": 2.0,
    "ankle_pitch_r_joint": 2.0,
    "waist_yaw_joint": 2.0,
    "shoulder_pitch_l_joint": 2.0,
    "shoulder_pitch_r_joint": 2.0,
}


def _xhumanoid_fwd_cmd_max(env: UltraMotionRunEnv) -> float:
    m = float(env.cfg.commands.ranges.lin_vel_x[1])
    return m if m > 1e-3 else 1e-3


def _xhumanoid_lat_cmd_max(env: UltraMotionRunEnv) -> float:
    r = env.cfg.commands.ranges.lin_vel_y
    m = max(abs(float(r[0])), abs(float(r[1])))
    return m if m > 1e-3 else 1e-3


def _xhumanoid_yaw_cmd_max(env: UltraMotionRunEnv) -> float:
    r = env.cfg.commands.ranges.ang_vel_z
    m = max(abs(float(r[0])), abs(float(r[1])))
    return m if m > 1e-3 else 1e-3


def _xhumanoid_joint_weights(
    env: UltraMotionRunEnv, weight_by_name: dict[str, float], cache_attr: str
) -> torch.Tensor:
    cached = getattr(env, cache_attr, None)
    if cached is not None:
        return cached
    weights = [weight_by_name.get(str(n), 1.0) for n in env.robot.data.joint_names]
    w = torch.tensor(weights, device=env.device, dtype=torch.float32)
    setattr(env, cache_attr, w)
    return w


def torques_weighted_neg(
    env: UltraMotionRunEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    style_mask: list[int] | None = None,
    min_speed_scale: float = 0.4,
    max_speed_scale: float = 2.0,
    speed_relax: float = 3.0,
) -> torch.Tensor:
    """Per-joint weighted torque L2, scaled by forward command (2026gym ``_reward_torques``)."""
    asset: Articulation = env.scene[asset_cfg.name]
    w = _xhumanoid_joint_weights(env, _XHUMANOID_TORQUE_WEIGHTS, "_xhumanoid_torque_weights")
    speed_scale = _action_rate_speed_scale(env, min_speed_scale, max_speed_scale, speed_relax)
    penalty = torch.sum(w * torch.square(asset.data.applied_torque), dim=1) * speed_scale
    if style_mask is None:
        mask = torch.ones_like(env.style_ids, dtype=torch.float32)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return penalty * mask


def wholebody_vel_weighted_neg(
    env: UltraMotionRunEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    style_mask: list[int] | None = None,
    min_speed_scale: float = 0.4,
    max_speed_scale: float = 2.0,
    speed_relax: float = 3.0,
) -> torch.Tensor:
    """Per-joint weighted joint-velocity L2 (2026gym ``_reward_wholebody_vel``)."""
    asset: Articulation = env.scene[asset_cfg.name]
    w = _xhumanoid_joint_weights(env, _XHUMANOID_VEL_WEIGHTS, "_xhumanoid_vel_weights")
    speed_scale = _action_rate_speed_scale(env, min_speed_scale, max_speed_scale, speed_relax)
    penalty = torch.sum(w * torch.square(asset.data.joint_vel), dim=1) * speed_scale
    if style_mask is None:
        mask = torch.ones_like(env.style_ids, dtype=torch.float32)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return penalty * mask


def hip_yaw_vel_cmd_neg(
    env: UltraMotionRunEnv,
    style_mask: list[int] | None = None,
) -> torch.Tensor:
    """Hip-yaw velocity penalty; relaxes as |cmd_yaw| grows (2026gym ``_reward_hip_yaw_vel``)."""
    asset: Articulation = env.scene["robot"]
    vel_cost = torch.sum(torch.square(asset.data.joint_vel[:, env.hip_yaw_ids]), dim=1)
    m = _xhumanoid_yaw_cmd_max(env)
    cmd_ratio = (env.command_generator.command[:, 2].abs() / m).clip(0.0, 1.0)
    cmd_w = 1.0 - 0.8 * cmd_ratio
    if style_mask is None:
        mask = torch.ones_like(env.style_ids, dtype=torch.float32)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return vel_cost * cmd_w * mask