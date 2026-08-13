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
#
# This file contains reward functions migrated from ultra_run_baseline (IsaacGym).

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from legged_lab.envs.ultra_js_wb.ultra_motion_run_env import UltraMotionRunEnv


def _apply_style_mask(env, reward: torch.Tensor, style_mask: list[int] | None) -> torch.Tensor:
    if style_mask is None:
        mask = torch.ones_like(env.style_ids)
    else:
        mask = torch.isin(env.style_ids, torch.tensor(style_mask, device=env.style_ids.device)).float()
    return reward * mask


def quat_to_euler_xyz(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion (w, x, y, z) to Euler angles (XYZ order).
    
    Args:
        quat: Quaternion tensor of shape (..., 4) in (w, x, y, z) format.
    
    Returns:
        Euler angles tensor of shape (..., 3) in (x, y, z) order (roll, pitch, yaw).
    """
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
    """
    Convert Euler angles (XYZ order) to quaternion (w, x, y, z).
    
    Args:
        euler: Euler angles tensor of shape (..., 3) in (x, y, z) order (roll, pitch, yaw).
    
    Returns:
        Quaternion tensor of shape (..., 4) in (w, x, y, z) format.
    """
    roll, pitch, yaw = euler[..., 0], euler[..., 1], euler[..., 2]
    
    # Precompute trigonometric values
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    
    # Compute quaternion components
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return torch.stack([w, x, y, z], dim=-1)


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


def joint_pos_limits(env: UltraMotionRunEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize joint positions that exceed limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    pos_limits = asset.data.joint_pos_limits[:, asset_cfg.joint_ids]
    out_of_limits = -(joint_pos - pos_limits[:, :, 0]).clip(max=0.)  # lower limit
    out_of_limits += (joint_pos - pos_limits[:, :, 1]).clip(min=0.)  # upper limit
    return _apply_style_mask(env, torch.sum(out_of_limits, dim=1), style_mask)


def ankle_pos_limits_neg(env: UltraMotionRunEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize ankle joint positions that exceed limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    ankle_ids = env.ankle_joint_ids
    joint_pos = asset.data.joint_pos[:, ankle_ids]
    pos_limits = asset.data.joint_pos_limits[:, ankle_ids]
    out_of_limits = -(joint_pos - pos_limits[:, :, 0]).clip(max=0.)
    out_of_limits += (joint_pos - pos_limits[:, :, 1]).clip(min=0.)
    return _apply_style_mask(env, torch.sum(out_of_limits, dim=1), style_mask)


def dof_vel_limits_neg(env: UltraMotionRunEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), soft_limit: float = 1.0, style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize joint velocities that exceed limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    vel_limits = asset.data.joint_vel_limits[:, asset_cfg.joint_ids] * soft_limit
    return _apply_style_mask(env, 0.1 * torch.sum((joint_vel - vel_limits).clip(min=0., max=1.), dim=1), style_mask)


def torque_limits_neg(env: UltraMotionRunEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), soft_limit: float = 1.0, style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize torques that exceed limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    torques = asset.data.applied_torque[:, asset_cfg.joint_ids]
    torque_limits = asset.data.joint_effort_limits[:, asset_cfg.joint_ids] * soft_limit
    return _apply_style_mask(env, 0.005 * torch.sum((torch.square(torques) - torch.square(torque_limits)).clip(min=0.), dim=1), style_mask)


def dof_vel_neg(env: UltraMotionRunEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize joint velocities."""
    asset: Articulation = env.scene[asset_cfg.name]
    return _apply_style_mask(env, torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1), style_mask)


def dof_acc_neg(env: UltraMotionRunEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), dt: float = 0.02, style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize joint accelerations."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_acc = (env.last_joint_vel[:, asset_cfg.joint_ids] - asset.data.joint_vel[:, asset_cfg.joint_ids]) / dt
    return _apply_style_mask(env, torch.sum(torch.square(joint_acc), dim=1), style_mask)


def dof_acc_diff(env: UltraMotionRunEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), dt: float = 0.02, style_mask: list[int] | None = None) -> torch.Tensor:
    """Reward smooth joint accelerations."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_acc = (env.last_joint_vel[:, asset_cfg.joint_ids] - asset.data.joint_vel[:, asset_cfg.joint_ids]) / dt
    return _apply_style_mask(env, torch.exp(-0.0000001 * torch.sum(torch.square(joint_acc), dim=1)), style_mask)


def torques_neg(env: UltraMotionRunEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize all torques."""
    asset: Articulation = env.scene[asset_cfg.name]
    return _apply_style_mask(env, torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1), style_mask)


def knee_torques_neg(env: UltraMotionRunEnv, command_name: str = "base_velocity", style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize knee torques scaled by forward velocity."""
    asset: Articulation = env.scene["robot"]
    knee_ids, _ = asset.find_joints(name_keys=["knee_pitch_l_joint", "knee_pitch_r_joint"], preserve_order=True)
    cmd = env.command_generator.command
    return _apply_style_mask(env, torch.sum(torch.square(asset.data.applied_torque[:, knee_ids]), dim=1) * cmd[:, 0].clip(min=0., max=2.0), style_mask)


def ankle_pitch_torques_neg(env: UltraMotionRunEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize ankle pitch torques."""
    asset: Articulation = env.scene[asset_cfg.name]
    ankle_pitch_ids, _ = asset.find_joints(name_keys=["ankle_pitch_l_joint", "ankle_pitch_r_joint"], preserve_order=True)
    return _apply_style_mask(env, torch.sum(torch.square(asset.data.applied_torque[:, ankle_pitch_ids]), dim=1), style_mask)


def ankle_roll_torques_neg(env: UltraMotionRunEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize ankle roll torques."""
    asset: Articulation = env.scene[asset_cfg.name]
    ankle_roll_ids, _ = asset.find_joints(name_keys=["ankle_roll_l_joint", "ankle_roll_r_joint"], preserve_order=True)
    return _apply_style_mask(env, torch.sum(torch.square(asset.data.applied_torque[:, ankle_roll_ids]), dim=1), style_mask)


def ankle_pitch_vel_neg(env: UltraMotionRunEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize ankle pitch velocities."""
    asset: Articulation = env.scene[asset_cfg.name]
    ankle_pitch_ids, _ = asset.find_joints(name_keys=["ankle_pitch_l_joint", "ankle_pitch_r_joint"], preserve_order=True)
    return _apply_style_mask(env, torch.sum(torch.square(asset.data.joint_vel[:, ankle_pitch_ids]), dim=1), style_mask)


def joint_power_neg(env: UltraMotionRunEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize joint power (velocity * torque)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return _apply_style_mask(env, torch.sum(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1), style_mask)


def action_diff(env: UltraMotionRunEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), style_mask: list[int] | None = None) -> torch.Tensor:
    """Reward smooth action changes."""
    actions = env.action_buffer._circular_buffer.buffer[:, -1, :]
    last_actions = env.action_buffer._circular_buffer.buffer[:, -2, :]
    return _apply_style_mask(env, torch.exp(-0.05 * torch.norm(actions - last_actions, dim=-1)), style_mask)


def ankleaction_diff(env: UltraMotionRunEnv, style_mask: list[int] | None = None) -> torch.Tensor:
    """Reward smooth ankle action changes."""
    actions = env.action_buffer._circular_buffer.buffer[:, -1, :]
    last_actions = env.action_buffer._circular_buffer.buffer[:, -2, :]
    ankle_ids = env.ankle_joint_ids
    return _apply_style_mask(env, torch.exp(-0.05 * torch.norm(actions[:, ankle_ids] - last_actions[:, ankle_ids], dim=-1)), style_mask)


def upperbody_action_diff(env: UltraMotionRunEnv, style_mask: list[int] | None = None) -> torch.Tensor:
    """Reward smooth upper body action changes."""
    actions = env.action_buffer._circular_buffer.buffer[:, -1, :]
    last_actions = env.action_buffer._circular_buffer.buffer[:, -2, :]
    arm_ids = env.left_arm_ids + env.right_arm_ids
    return _apply_style_mask(env, torch.exp(-0.05 * torch.norm(actions[:, arm_ids] - last_actions[:, arm_ids], dim=-1)), style_mask)


def hiproll_action_diff_neg(env: UltraMotionRunEnv, style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize hip roll action changes."""
    asset: Articulation = env.scene["robot"]
    actions = env.action_buffer._circular_buffer.buffer[:, -1, :]
    last_actions = env.action_buffer._circular_buffer.buffer[:, -2, :]
    hip_roll_ids, _ = asset.find_joints(name_keys=["hip_roll_l_joint", "hip_roll_r_joint"], preserve_order=True)
    return _apply_style_mask(env, 1.0 * torch.norm(actions[:, hip_roll_ids] - last_actions[:, hip_roll_ids], dim=-1), style_mask)


def hiproll_action_diff_second_neg(env: UltraMotionRunEnv, style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize second-order hip roll action changes."""
    asset: Articulation = env.scene["robot"]
    actions = env.action_buffer._circular_buffer.buffer[:, -1, :]
    last_actions = env.action_buffer._circular_buffer.buffer[:, -2, :]
    last_last_actions = env.action_buffer._circular_buffer.buffer[:, -3, :]
    hip_roll_ids, _ = asset.find_joints(name_keys=["hip_roll_l_joint", "hip_roll_r_joint"], preserve_order=True)
    return _apply_style_mask(env, torch.norm(actions[:, hip_roll_ids] - 2 * last_actions[:, hip_roll_ids] + last_last_actions[:, hip_roll_ids], dim=-1), style_mask)


def hipyaw_action_diff_neg(env: UltraMotionRunEnv, style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize hip yaw action changes."""
    asset: Articulation = env.scene["robot"]
    actions = env.action_buffer._circular_buffer.buffer[:, -1, :]
    last_actions = env.action_buffer._circular_buffer.buffer[:, -2, :]
    hip_yaw_ids, _ = asset.find_joints(name_keys=["hip_yaw_l_joint", "hip_yaw_r_joint"], preserve_order=True)
    return _apply_style_mask(env, 1.0 * torch.norm(actions[:, hip_yaw_ids] - last_actions[:, hip_yaw_ids], dim=-1), style_mask)


def hipyaw_action_diff_second_neg(env: UltraMotionRunEnv, style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize second-order hip yaw action changes."""
    asset: Articulation = env.scene["robot"]
    actions = env.action_buffer._circular_buffer.buffer[:, -1, :]
    last_actions = env.action_buffer._circular_buffer.buffer[:, -2, :]
    last_last_actions = env.action_buffer._circular_buffer.buffer[:, -3, :]
    hip_yaw_ids, _ = asset.find_joints(name_keys=["hip_yaw_l_joint", "hip_yaw_r_joint"], preserve_order=True)
    return _apply_style_mask(env, torch.norm(actions[:, hip_yaw_ids] - 2 * last_actions[:, hip_yaw_ids] + last_last_actions[:, hip_yaw_ids], dim=-1), style_mask)


def hippitch_action_diff_neg(env: UltraMotionRunEnv, style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize hip pitch action changes."""
    asset: Articulation = env.scene["robot"]
    actions = env.action_buffer._circular_buffer.buffer[:, -1, :]
    last_actions = env.action_buffer._circular_buffer.buffer[:, -2, :]
    hip_pitch_ids, _ = asset.find_joints(name_keys=["hip_pitch_l_joint", "hip_pitch_r_joint"], preserve_order=True)
    return _apply_style_mask(env, 1.0 * torch.norm(actions[:, hip_pitch_ids] - last_actions[:, hip_pitch_ids], dim=-1), style_mask)


def hippitch_action_diff_second_neg(env: UltraMotionRunEnv, style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize second-order hip pitch action changes."""
    asset: Articulation = env.scene["robot"]
    actions = env.action_buffer._circular_buffer.buffer[:, -1, :]
    last_actions = env.action_buffer._circular_buffer.buffer[:, -2, :]
    last_last_actions = env.action_buffer._circular_buffer.buffer[:, -3, :]
    hip_pitch_ids, _ = asset.find_joints(name_keys=["hip_pitch_l_joint", "hip_pitch_r_joint"], preserve_order=True)
    return _apply_style_mask(env, torch.norm(actions[:, hip_pitch_ids] - 2 * last_actions[:, hip_pitch_ids] + last_last_actions[:, hip_pitch_ids], dim=-1), style_mask)


def hiproll_acc_neg(env: UltraMotionRunEnv, dt: float = 0.02, style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize hip roll acceleration."""
    asset: Articulation = env.scene["robot"]
    hip_roll_ids, _ = asset.find_joints(name_keys=["hip_roll_l_joint", "hip_roll_r_joint"], preserve_order=True)
    hip_roll_acc = (env.last_joint_vel[:, hip_roll_ids] - asset.data.joint_vel[:, hip_roll_ids]) / dt
    return _apply_style_mask(env, torch.sum(torch.square(hip_roll_acc), dim=1), style_mask)


def ankle_action_neg(env: UltraMotionRunEnv, style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize ankle actions."""
    asset: Articulation = env.scene["robot"]
    actions = env.action_buffer._circular_buffer.buffer[:, -1, :]
    ankle_pitch_ids, _ = asset.find_joints(name_keys=["ankle_pitch_l_joint", "ankle_pitch_r_joint"], preserve_order=True)
    return _apply_style_mask(env, torch.sum(torch.abs(actions[:, ankle_pitch_ids]), dim=1), style_mask)


def ankle_roll_action_neg(env: UltraMotionRunEnv, style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize ankle roll actions."""
    asset: Articulation = env.scene["robot"]
    actions = env.action_buffer._circular_buffer.buffer[:, -1, :]
    ankle_roll_ids, _ = asset.find_joints(name_keys=["ankle_roll_l_joint", "ankle_roll_r_joint"], preserve_order=True)
    return _apply_style_mask(env, torch.sum(torch.abs(actions[:, ankle_roll_ids]), dim=1), style_mask)


def arm_action_neg(env: UltraMotionRunEnv, style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize arm actions."""
    asset: Articulation = env.scene["robot"]
    actions = env.action_buffer._circular_buffer.buffer[:, -1, :]
    shoulder_ids, _ = asset.find_joints(name_keys=["shoulder_pitch_l_joint", "shoulder_pitch_r_joint"], preserve_order=True)
    return _apply_style_mask(env, torch.sum(torch.square(actions[:, shoulder_ids]), dim=1), style_mask)


def elbow_action_neg(env: UltraMotionRunEnv, style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize elbow actions."""
    asset: Articulation = env.scene["robot"]
    actions = env.action_buffer._circular_buffer.buffer[:, -1, :]
    elbow_ids, _ = asset.find_joints(name_keys=["elbow_pitch_l_joint", "elbow_pitch_r_joint"], preserve_order=True)
    return _apply_style_mask(env, torch.sum(torch.square(actions[:, elbow_ids] + 0.85), dim=1), style_mask)


def elbow_pos_neg(env: UltraMotionRunEnv, style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize elbow positions away from default."""
    asset: Articulation = env.scene["robot"]
    elbow_ids, _ = asset.find_joints(name_keys=["elbow_pitch_l_joint", "elbow_pitch_r_joint"], preserve_order=True)
    default_pos = asset.data.default_joint_pos[:, elbow_ids]
    joint_pos = asset.data.joint_pos[:, elbow_ids]
    return _apply_style_mask(env, torch.abs(joint_pos + 0.5).sum(dim=1), style_mask)


def hip_roll_action_neg(env: UltraMotionRunEnv, style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize hip roll actions."""
    asset: Articulation = env.scene["robot"]
    actions = env.action_buffer._circular_buffer.buffer[:, -1, :]
    hip_roll_ids, _ = asset.find_joints(name_keys=["hip_roll_l_joint", "hip_roll_r_joint"], preserve_order=True)
    return _apply_style_mask(env, torch.sum(torch.abs(actions[:, hip_roll_ids]), dim=1), style_mask)


def hip_roll_action_air_neg(env: UltraMotionRunEnv, delta_t: float = 0.02, command_name: str = "base_velocity", style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize hip roll positions during swing phase."""
    asset: Articulation = env.scene["robot"]
    left_frc_clock = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[0]
    right_frc_clock = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[0]
    hip_roll_ids, _ = asset.find_joints(name_keys=["hip_roll_l_joint", "hip_roll_r_joint"], preserve_order=True)
    cmd = env.command_generator.command
    return _apply_style_mask(
        env,
        (left_frc_clock * torch.abs(asset.data.joint_pos[:, hip_roll_ids[0]]) +
         right_frc_clock * torch.abs(asset.data.joint_pos[:, hip_roll_ids[1]])) * cmd[:, 0].clip(min=0., max=4.0),
        style_mask,
    )


def hip_yaw_action_neg(env: UltraMotionRunEnv, command_name: str = "base_velocity", style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize hip yaw positions based on yaw command."""
    asset: Articulation = env.scene["robot"]
    hip_yaw_ids, _ = asset.find_joints(name_keys=["hip_yaw_l_joint", "hip_yaw_r_joint"], preserve_order=True)
    cmd = env.command_generator.command
    return _apply_style_mask(env, torch.sum(torch.abs(asset.data.joint_pos[:, hip_yaw_ids]), dim=1) * (1.0 - torch.abs(cmd[:, 2]).clip(max=1.0)), style_mask)


def hip_yaw_vel_neg(env: UltraMotionRunEnv, command_name: str = "base_velocity", style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize hip yaw velocities based on commands."""
    asset: Articulation = env.scene["robot"]
    hip_yaw_ids, _ = asset.find_joints(name_keys=["hip_yaw_l_joint", "hip_yaw_r_joint"], preserve_order=True)
    cmd = env.command_generator.command
    return _apply_style_mask(env, torch.sum(torch.square(asset.data.joint_vel[:, hip_yaw_ids]), dim=1) * (1.0 - torch.abs(cmd[:, 2]).clip(max=1.0)) * torch.abs(cmd[:, 0]), style_mask)


def shoulder_pitch_vel_neg(env: UltraMotionRunEnv, style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize shoulder pitch velocities."""
    asset: Articulation = env.scene["robot"]
    shoulder_ids, _ = asset.find_joints(name_keys=["shoulder_pitch_l_joint", "shoulder_pitch_r_joint"], preserve_order=True)
    return _apply_style_mask(env, torch.sum(torch.square(asset.data.joint_vel[:, shoulder_ids]), dim=1), style_mask)


def shoulder_pitch_acc_neg(env: UltraMotionRunEnv, dt: float = 0.02, style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize shoulder pitch accelerations."""
    asset: Articulation = env.scene["robot"]
    shoulder_ids, _ = asset.find_joints(name_keys=["shoulder_pitch_l_joint", "shoulder_pitch_r_joint"], preserve_order=True)
    shoulder_acc = (env.last_joint_vel[:, shoulder_ids] - asset.data.joint_vel[:, shoulder_ids]) / dt
    return _apply_style_mask(env, torch.sum(torch.square(shoulder_acc), dim=1), style_mask)


def elbow_pitch_vel_neg(env: UltraMotionRunEnv, style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize elbow pitch velocities."""
    asset: Articulation = env.scene["robot"]
    elbow_ids, _ = asset.find_joints(name_keys=["elbow_pitch_l_joint", "elbow_pitch_r_joint"], preserve_order=True)
    return _apply_style_mask(env, torch.sum(torch.square(asset.data.joint_vel[:, elbow_ids]), dim=1), style_mask)


def arm_vel_ang_yaw_neg(env: UltraMotionRunEnv, command_name: str = "base_velocity", style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize arm velocities during yaw commands."""
    asset: Articulation = env.scene["robot"]
    arm_ids = env.left_arm_ids + env.right_arm_ids
    cmd = env.command_generator.command
    return _apply_style_mask(env, torch.sum(torch.square(asset.data.joint_vel[:, arm_ids]), dim=1) * torch.abs(cmd[:, 2]) * torch.abs(cmd[:, 0]).clip(max=2.0), style_mask)


def hip_roll_pos_air_ang_yaw_neg(env: UltraMotionRunEnv, delta_t: float = 0.02, command_name: str = "base_velocity", style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize hip roll positions during swing phase when yawing."""
    asset: Articulation = env.scene["robot"]
    left_frc_clock = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[0]
    right_frc_clock = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[0]
    hip_roll_ids, _ = asset.find_joints(name_keys=["hip_roll_l_joint", "hip_roll_r_joint"], preserve_order=True)
    cmd = env.command_generator.command
    return _apply_style_mask(
        env,
        ((left_frc_clock * torch.abs(asset.data.joint_pos[:, hip_roll_ids[0]]) +
          right_frc_clock * torch.abs(asset.data.joint_pos[:, hip_roll_ids[1]])) *
         (0.5 - torch.abs(cmd[:, 1]).clip(max=0.5)) * torch.abs(cmd[:, 2]) * torch.abs(cmd[:, 0])),
        style_mask,
    )


def yaw_vel_neg(env: UltraMotionRunEnv, style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize yaw velocity when opposite to command."""
    asset: Articulation = env.scene["robot"]
    cmd = env.command_generator.command
    ang_vel_z = asset.data.root_ang_vel_b[:, 2]
    score = (ang_vel_z * cmd[:, 2]).clip(max=0.) * torch.abs(cmd[:, 0])
    reward = torch.square(score)
    return _apply_style_mask(env, reward, style_mask)


def stumble_x_neg(env: UltraMotionRunEnv, sensor_cfg: SceneEntityCfg, style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize feet hitting vertical surfaces (stumbling)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    # Check if horizontal forces exceed vertical forces (indicating stumbling)
    horizontal_forces = torch.abs(contact_forces[:, :, 0])
    vertical_forces = torch.abs(contact_forces[:, :, 2])
    return _apply_style_mask(env, torch.any(horizontal_forces > 0.5 * vertical_forces, dim=1).float(), style_mask)


def episode_length(env: UltraMotionRunEnv, style_mask: list[int] | None = None) -> torch.Tensor:
    """Reward for longer episodes."""
    return _apply_style_mask(env, 0.1 * env.episode_length_buf.float() * env.step_dt, style_mask)


def base_height_run_neg(env: UltraMotionRunEnv, base_height_target: float = 0.85, style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize base height away from target for running."""
    asset: Articulation = env.scene["robot"]
    # Get base height relative to terrain (simplified - may need terrain height)
    base_height = asset.data.root_state_w[:, 2]
    score = torch.abs((base_height - base_height_target).clip(max=0., min=-0.5))
    return _apply_style_mask(env, score, style_mask)


def straighten_knee_neg(env: UltraMotionRunEnv, command_name: str = "base_velocity", style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize knees not straightened during stance phase."""
    asset: Articulation = env.scene["robot"]
    # Avoid division by zero when phase_ratio is 0 (standing state)
    phase_ratio_0 = torch.clamp(env.phase_ratio[:, 0], min=1e-6)
    phase_ratio_1 = torch.clamp(env.phase_ratio[:, 1], min=1e-6)
    s0 = env.gait_phase[:, 0] / phase_ratio_0 - 0.5
    s0 = s0.clip(min=0., max=1.) * (s0 < 0.6)
    s0 = s0.clip(min=0., max=0.5)
    s1 = env.gait_phase[:, 1] / phase_ratio_1 - 0.5
    s1 = s1.clip(min=0., max=1.) * (s1 < 0.6)
    s1 = s1.clip(min=0., max=0.5)
    
    knee_ids, _ = asset.find_joints(name_keys=["knee_pitch_l_joint", "knee_pitch_r_joint"], preserve_order=True)
    left_score = torch.abs(asset.data.joint_pos[:, knee_ids[0]] - 0.3) * s0
    right_score = torch.abs(asset.data.joint_pos[:, knee_ids[1]] - 0.3) * s1
    return _apply_style_mask(env, left_score + right_score, style_mask)


def knee_vel_swing_neg(env: UltraMotionRunEnv, command_name: str = "base_velocity", style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize knee velocities during swing phase."""
    asset: Articulation = env.scene["robot"]
    # Avoid division by zero when phase_ratio is 0 (standing state)
    phase_ratio_0 = torch.clamp(env.phase_ratio[:, 0], min=1e-6)
    phase_ratio_1 = torch.clamp(env.phase_ratio[:, 1], min=1e-6)
    s0 = env.gait_phase[:, 0] / phase_ratio_0
    s00 = (s0 < 0.5)
    s01 = (s0 > 0.5) & (s0 < 1.0)
    s1 = env.gait_phase[:, 1] / phase_ratio_1
    s10 = (s1 < 0.5)
    s11 = (s1 > 0.5) & (s1 < 1.0)
    
    knee_ids, _ = asset.find_joints(name_keys=["knee_pitch_l_joint", "knee_pitch_r_joint"], preserve_order=True)
    knee_vel_l = (asset.data.joint_vel[:, knee_ids[0]] * s00).clip(max=0.) - (asset.data.joint_vel[:, knee_ids[0]] * s01).clip(min=0.)
    knee_vel_r = (asset.data.joint_vel[:, knee_ids[1]] * s10).clip(max=0.) - (asset.data.joint_vel[:, knee_ids[1]] * s11).clip(min=0.)
    
    cmd = env.command_generator.command
    return _apply_style_mask(env, torch.abs((knee_vel_l + knee_vel_r) * cmd[:, 0].clip(min=0., max=4.0)), style_mask)


def foot_swing_height_neg(env: UltraMotionRunEnv, command_name: str = "base_velocity", style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize feet not high enough during swing phase."""
    asset: Articulation = env.scene["robot"]
    # Avoid division by zero when phase_ratio is 0 (standing state)
    phase_ratio_0 = torch.clamp(env.phase_ratio[:, 0], min=1e-6)
    phase_ratio_1 = torch.clamp(env.phase_ratio[:, 1], min=1e-6)
    s0 = env.gait_phase[:, 0] / phase_ratio_0
    s01 = (s0 > 0.3) & (s0 < 0.6)
    s1 = env.gait_phase[:, 1] / phase_ratio_1
    s11 = (s1 > 0.3) & (s1 < 0.6)
    
    left_foot_pos = asset.data.body_state_w[:, env.feet_body_ids[0], 2]
    right_foot_pos = asset.data.body_state_w[:, env.feet_body_ids[1], 2]
    
    left_score = s01 * (left_foot_pos - right_foot_pos - 0.17).clip(max=0.)
    right_score = s11 * (right_foot_pos - left_foot_pos - 0.17).clip(max=0.)
    
    cmd = env.command_generator.command
    flag = cmd[:, 0].clip(min=0., max=4.0) * (cmd[:, 0] > 1.0)
    return _apply_style_mask(env, torch.abs((left_score + right_score) * flag), style_mask)


def feet_speed_z_diff_new_neg(env: UltraMotionRunEnv, sensor_cfg: SceneEntityCfg, style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize feet moving too fast downward."""
    asset: Articulation = env.scene["robot"]
    left_foot_v = asset.data.body_lin_vel_w[:, env.feet_body_ids[0], 2]
    right_foot_v = asset.data.body_lin_vel_w[:, env.feet_body_ids[1], 2]
    rew_left_fspd = left_foot_v < -2.0
    rew_right_fspd = right_foot_v < -2.0
    return _apply_style_mask(env, rew_left_fspd.float() + rew_right_fspd.float(), style_mask)


def gait_swing_neg(env: UltraMotionRunEnv, delta_t: float = 0.02, style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize foot downward velocity during swing phase."""
    asset: Articulation = env.scene["robot"]
    # Avoid division by zero when phase_ratio is 0 (standing state)
    phase_ratio_0 = torch.clamp(env.phase_ratio[:, 0], min=1e-6)
    phase_ratio_1 = torch.clamp(env.phase_ratio[:, 1], min=1e-6)
    s0 = env.gait_phase[:, 0] / phase_ratio_0 - 0.5
    s0 = s0.clip(min=0., max=1.) * (s0 < 0.6)
    s0 = s0.clip(min=0., max=0.5)
    s1 = env.gait_phase[:, 1] / phase_ratio_1 - 0.5
    s1 = s1.clip(min=0., max=1.) * (s1 < 0.6)
    s1 = s1.clip(min=0., max=0.5)
    
    left_foot_v = asset.data.body_lin_vel_w[:, env.feet_body_ids[0], 2]
    right_foot_v = asset.data.body_lin_vel_w[:, env.feet_body_ids[1], 2]
    
    left_score = 16 * s0 * s0 * left_foot_v * left_foot_v
    right_score = 16 * s1 * s1 * right_foot_v * right_foot_v
    return _apply_style_mask(env, (left_score + right_score).clip(max=1.), style_mask)


def gait_feet_contact_neg(env: UltraMotionRunEnv, sensor_cfg: SceneEntityCfg, delta_t: float = 0.02, style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize foot velocity during contact transition."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene["robot"]
    
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    contact = contact_forces > 1.0
    
    foot_v = torch.abs(asset.data.body_lin_vel_w[:, env.feet_body_ids, :])
    foot_v_norm = torch.norm(foot_v, dim=-1)
    
    s0 = env.gait_phase[:, :] - env.phase_ratio[:, :] > -0.1
    s1 = env.gait_phase[:, :] - env.phase_ratio[:, :] < 0.2
    
    score = s0 * s1 * foot_v_norm
    return _apply_style_mask(env, torch.norm(score, dim=-1), style_mask)


def foot_flip_neg(env: UltraMotionRunEnv, sensor_cfg: SceneEntityCfg, style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize foot angular velocity when in contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene["robot"]
    
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    contact = contact_forces > 1.0
    
    foot_ang_vel = asset.data.body_ang_vel_w[:, env.feet_body_ids, :]
    foot_ang_vel_norm = torch.norm(foot_ang_vel, dim=-1)
    
    rew = foot_ang_vel_norm * contact.float()
    return _apply_style_mask(env, torch.sum(rew, dim=1), style_mask)


def gait_feet_force_max_neg(env: UltraMotionRunEnv, sensor_cfg: SceneEntityCfg, max_force: float = 1500.0, style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize excessive foot contact forces."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    extra_frc = (contact_forces - max_force).clip(min=0., max=100)
    return _apply_style_mask(env, torch.sum(extra_frc, dim=-1), style_mask)


def gait_feet_distance_collision_neg(env: UltraMotionRunEnv, style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize feet being too close together (collision risk)."""
    asset: Articulation = env.scene["robot"]
    left_foot_pos = asset.data.body_state_w[:, env.feet_body_ids[0], :3] - asset.data.root_state_w[:, :3]
    right_foot_pos = asset.data.body_state_w[:, env.feet_body_ids[1], :3] - asset.data.root_state_w[:, :3]
    
    # Transform to body frame
    root_quat = asset.data.root_state_w[:, 3:7]
    left_foot_pos = math_utils.quat_apply_inverse(root_quat, left_foot_pos)
    right_foot_pos = math_utils.quat_apply_inverse(root_quat, right_foot_pos)
    
    feet_distance_y = torch.abs(left_foot_pos[:, 1] - right_foot_pos[:, 1])
    feet_distance_x = torch.abs(left_foot_pos[:, 0] - right_foot_pos[:, 0])
    collision_flag = (feet_distance_y < 0.14) * (feet_distance_x < 0.24)
    return _apply_style_mask(env, collision_flag.float(), style_mask)


def z_vel_run(env: UltraMotionRunEnv, style_mask: list[int] | None = None) -> torch.Tensor:
    """Reward upward velocity during stance-to-flight transition."""
    asset: Articulation = env.scene["robot"]
    # Avoid division by zero when phase_ratio is 0 (standing state)
    phase_ratio_0 = torch.clamp(env.phase_ratio[:, 0], min=1e-6)
    phase_ratio_1 = torch.clamp(env.phase_ratio[:, 1], min=1e-6)
    s0 = env.gait_phase[:, 0] / phase_ratio_0 - 1.0
    s01 = s0 > 0
    s0 = s0 * (s0 > 0.)
    # Avoid division by zero: (1.0 - phase_ratio) / phase_ratio
    denom_0 = (1.0 - phase_ratio_0) / phase_ratio_0
    denom_0 = torch.clamp(denom_0, min=1e-6)  # Ensure denominator is not too small
    s0 = s0 / denom_0
    s0 = s01.float() * (s0 - 0.5)
    left_spd_score = s0 * asset.data.root_lin_vel_b[:, 2]
    
    s1 = env.gait_phase[:, 1] / phase_ratio_1 - 1.0
    s11 = s1 > 0
    s1 = s1 * (s1 > 0.)
    # Avoid division by zero: (1.0 - phase_ratio) / phase_ratio
    denom_1 = (1.0 - phase_ratio_1) / phase_ratio_1
    denom_1 = torch.clamp(denom_1, min=1e-6)  # Ensure denominator is not too small
    s1 = s1 / denom_1
    s1 = s11.float() * (s1 - 0.5)
    right_spd_score = s1 * asset.data.root_lin_vel_b[:, 2]
    
    return _apply_style_mask(env, 5.0 * (left_spd_score + right_spd_score), style_mask)


def x_vel_bonus(env: UltraMotionRunEnv, style_mask: list[int] | None = None) -> torch.Tensor:
    """Reward for tracking forward velocity with bonus scaling by command magnitude."""
    asset: Articulation = env.scene["robot"]
    cmd = env.command_generator.command
    base_lin_vel = asset.data.root_lin_vel_b
    
    flag = torch.abs(cmd[:, 0]) < 0.1
    x_vel_error = (flag * torch.abs(cmd[:, 0] - base_lin_vel[:, 0]) +
                   ~flag * torch.abs(cmd[:, 0] - env.avg_x_vel))
    reward = torch.exp(-2 * x_vel_error) * torch.abs(cmd[:, 0])
    return _apply_style_mask(env, reward, style_mask)


def x_vel_neg(env: UltraMotionRunEnv, style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize forward velocity tracking error."""
    asset: Articulation = env.scene["robot"]
    cmd = env.command_generator.command
    base_lin_vel = asset.data.root_lin_vel_b
    
    flag = torch.abs(cmd[:, 0]) < 0.1
    x_vel_error = (flag * torch.abs(cmd[:, 0] - base_lin_vel[:, 0]) +
             ~flag * torch.abs(cmd[:, 0] - env.avg_x_vel))
    return _apply_style_mask(env, x_vel_error, style_mask)


def ang_vel_bonus(env: UltraMotionRunEnv, style_mask: list[int] | None = None) -> torch.Tensor:
    """Reward for tracking angular velocity with bonus scaling by forward command."""
    asset: Articulation = env.scene["robot"]
    cmd = env.command_generator.command
    base_ang_vel = asset.data.root_ang_vel_b
    
    flag = torch.abs(cmd[:, 2]) < 0.1
    ang_vel_error = (flag * torch.abs(cmd[:, 2] - base_ang_vel[:, 2]) +
                     ~flag * torch.abs(cmd[:, 2] - env.avg_yaw_vel))
    reward = torch.exp(-3 * ang_vel_error) * torch.abs(cmd[:, 0])
    return _apply_style_mask(env, reward, style_mask)


def ang_vel_neg(env: UltraMotionRunEnv, style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize angular velocity tracking error."""
    asset: Articulation = env.scene["robot"]
    cmd = env.command_generator.command
    base_ang_vel = asset.data.root_ang_vel_b
    
    flag = torch.abs(cmd[:, 2]) < 0.1
    ang_vel_error = (flag * torch.abs(cmd[:, 2] - base_ang_vel[:, 2]) +
                     ~flag * torch.abs(cmd[:, 2] - env.avg_yaw_vel))
    return _apply_style_mask(env, ang_vel_error, style_mask)


def ang_vel_roll_neg(env: UltraMotionRunEnv, style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize roll angular velocity."""
    asset: Articulation = env.scene["robot"]
    cmd = env.command_generator.command
    ang_vel_error = torch.square(asset.data.root_ang_vel_b[:, 0]).clip(max=1.0)
    reward = ang_vel_error * (1 + torch.abs(cmd[:, 0])).clip(max=2.0) * (1 + torch.abs(cmd[:, 2]))
    return _apply_style_mask(env, reward, style_mask)


def hipyaw_perio(env: UltraMotionRunEnv, delta_t: float = 0.02, style_mask: list[int] | None = None) -> torch.Tensor:
    """Reward periodic hip yaw motion during stance phase."""
    asset: Articulation = env.scene["robot"]
    left_frc_clock = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[0]
    right_frc_clock = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[0]
    
    left_yaw_stance = (env.gait_phase[:, 0] - env.phase_ratio[:, 0])
    left_yaw_stance = left_yaw_stance.clip(min=0., max=1.) * env.cfg.gait.gait_cycle
    right_yaw_stance = (env.gait_phase[:, 1] - env.phase_ratio[:, 1])
    right_yaw_stance = right_yaw_stance.clip(min=0., max=1.) * env.cfg.gait.gait_cycle
    
    cmd = env.command_generator.command
    action_scale = env.action_scale
    
    hip_yaw_ids, _ = asset.find_joints(name_keys=["hip_yaw_l_joint", "hip_yaw_r_joint"], preserve_order=True)
    default_pos = asset.data.default_joint_pos[:, hip_yaw_ids]
    
    actions = env.action_buffer._circular_buffer.buffer[:, -1, :]
    left_yaw_score = (left_frc_clock * torch.exp(-10.0 * torch.abs(actions[:, hip_yaw_ids[0]] - default_pos[:, 0])) +
                        (1 - left_frc_clock) * torch.exp(-10.0 * torch.abs(actions[:, hip_yaw_ids[0]] + left_yaw_stance * cmd[:, 2] / action_scale)))
    right_yaw_score = (right_frc_clock * torch.exp(-10.0 * torch.abs(actions[:, hip_yaw_ids[1]] - default_pos[:, 1])) +
                        (1 - right_frc_clock) * torch.exp(-10.0 * torch.abs(actions[:, hip_yaw_ids[1]] + right_yaw_stance * cmd[:, 2] / action_scale)))
    
    return _apply_style_mask(env, left_yaw_score + right_yaw_score, style_mask)


def gait_swing_symmetric(env: UltraMotionRunEnv, delta_t: float = 0.02, style_mask: list[int] | None = None) -> torch.Tensor:
    """Reward symmetric foot placement during double-air phase."""
    asset: Articulation = env.scene["robot"]
    left_frc_clock = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[0].unsqueeze(1)
    right_frc_clock = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[0].unsqueeze(1)
    
    left_foot_pos = asset.data.body_state_w[:, env.feet_body_ids[0], :3] - asset.data.root_state_w[:, :3]
    right_foot_pos = asset.data.body_state_w[:, env.feet_body_ids[1], :3] - asset.data.root_state_w[:, :3]
    
    root_quat = asset.data.root_state_w[:, 3:7]
    feet_distance = left_foot_pos - right_foot_pos
    feet_distance = math_utils.quat_apply_inverse(root_quat, feet_distance)
    
    db_air_flag = (left_frc_clock > 0.9) * (right_frc_clock > 0.9)
    
    # Initialize feet_distance buffer if not exists
    if not hasattr(env, 'feet_distance_buffer'):
        env.feet_distance_buffer = torch.zeros_like(feet_distance)
    if not hasattr(env, 'last_feet_distance_buffer'):
        env.last_feet_distance_buffer = torch.zeros_like(feet_distance)
    
    score = torch.abs(feet_distance[:, 0:1] + env.last_feet_distance_buffer[:, 0:1])
    env.feet_distance_buffer = db_air_flag * feet_distance + (~db_air_flag) * env.feet_distance_buffer
    env.last_feet_distance_buffer = (~db_air_flag) * env.feet_distance_buffer + db_air_flag * env.last_feet_distance_buffer
    
    return _apply_style_mask(env, (3.3 * db_air_flag * torch.exp(-10 * score)).squeeze(1), style_mask)


def gait_feet_distance(env: UltraMotionRunEnv, style_mask: list[int] | None = None) -> torch.Tensor:
    """Reward proper foot distance during stance."""
    asset: Articulation = env.scene["robot"]
    left_foot_pos = asset.data.body_state_w[:, env.feet_body_ids[0], :3] - asset.data.root_state_w[:, :3]
    right_foot_pos = asset.data.body_state_w[:, env.feet_body_ids[1], :3] - asset.data.root_state_w[:, :3]
    
    # Get root orientation without yaw
    root_quat = asset.data.root_state_w[:, 3:7]
    root_euler = quat_to_euler_xyz(root_quat)
    root_euler_no_yaw = torch.stack([
        torch.zeros_like(root_euler[:, 0]),
        torch.zeros_like(root_euler[:, 1]),
        root_euler[:, 2]
    ], dim=1)
    quat_without_yaw = euler_xyz_to_quat(root_euler_no_yaw)
    
    left_foot_pos = math_utils.quat_apply_inverse(quat_without_yaw, left_foot_pos)
    right_foot_pos = math_utils.quat_apply_inverse(quat_without_yaw, right_foot_pos)
    
    feet_distance_x = torch.abs(left_foot_pos[:, 0] - right_foot_pos[:, 0])
    feet_distance_y_left = torch.abs(left_foot_pos[:, 1] - 0.12)
    feet_distance_y_right = torch.abs(right_foot_pos[:, 1] + 0.12)
    
    return _apply_style_mask(env, 0.5 * torch.exp(-10 * feet_distance_x) + 0.5 * torch.exp(-10 * (feet_distance_y_left + feet_distance_y_right)), style_mask)


def gait_feet_y_distance_neg(env: UltraMotionRunEnv, command_name: str = "base_velocity", style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize feet y-distance from target."""
    asset: Articulation = env.scene["robot"]
    left_foot_pos = asset.data.body_state_w[:, env.feet_body_ids[0], :3] - asset.data.root_state_w[:, :3]
    right_foot_pos = asset.data.body_state_w[:, env.feet_body_ids[1], :3] - asset.data.root_state_w[:, :3]
    
    root_quat = asset.data.root_state_w[:, 3:7]
    root_euler = quat_to_euler_xyz(root_quat)
    root_euler_no_yaw = torch.stack([
        torch.zeros_like(root_euler[:, 0]),
        torch.zeros_like(root_euler[:, 1]),
        root_euler[:, 2]
    ], dim=1)
    quat_without_yaw = euler_xyz_to_quat(root_euler_no_yaw)
    
    left_foot_pos = math_utils.quat_apply_inverse(quat_without_yaw, left_foot_pos)
    right_foot_pos = math_utils.quat_apply_inverse(quat_without_yaw, right_foot_pos)
    
    cmd = env.command_generator.command
    score_left = 2.0 * torch.abs(0.12 - left_foot_pos[:, 1]) * (0.5 - torch.abs(cmd[:, 1]).clip(max=0.5))
    score_right = 2.0 * torch.abs(0.12 + right_foot_pos[:, 1]) * (0.5 - torch.abs(cmd[:, 1]).clip(max=0.5))
    
    return _apply_style_mask(env, score_left + score_right, style_mask)


def gait_feet_x_distance_neg(env: UltraMotionRunEnv, command_name: str = "base_velocity", style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize feet x-distance during mid-air phase."""
    asset: Articulation = env.scene["robot"]
    left_foot_pos = asset.data.body_state_w[:, env.feet_body_ids[0], :3] - asset.data.root_state_w[:, :3]
    right_foot_pos = asset.data.body_state_w[:, env.feet_body_ids[1], :3] - asset.data.root_state_w[:, :3]
    
    root_quat = asset.data.root_state_w[:, 3:7]
    root_euler = quat_to_euler_xyz(root_quat)
    root_euler_no_yaw = torch.stack([
        torch.zeros_like(root_euler[:, 0]),
        torch.zeros_like(root_euler[:, 1]),
        root_euler[:, 2]
    ], dim=1)
    quat_without_yaw = euler_xyz_to_quat(root_euler_no_yaw)
    
    left_foot_pos = math_utils.quat_apply_inverse(quat_without_yaw, left_foot_pos)
    right_foot_pos = math_utils.quat_apply_inverse(quat_without_yaw, right_foot_pos)
    
    # Avoid division by zero when phase_ratio is 0 (standing state)
    phase_ratio_0 = torch.clamp(env.phase_ratio[:, 0], min=1e-6)
    phase_ratio_1 = torch.clamp(env.phase_ratio[:, 1], min=1e-6)
    mid_air_flag = (torch.abs(env.gait_phase[:, 0] / phase_ratio_0 - 0.5) < 0.02) + \
                   (torch.abs(env.gait_phase[:, 1] / phase_ratio_1 - 0.5) < 0.02)
    
    cmd = env.command_generator.command
    score = 2.0 * mid_air_flag.float() * (torch.abs(left_foot_pos[:, 0]) + torch.abs(right_foot_pos[:, 0])) * (0.5 - torch.abs(cmd[:, 0]).clip(max=0.5))
    
    return _apply_style_mask(env, score, style_mask)


def feet_speed_z_diff(env: UltraMotionRunEnv, sensor_cfg: SceneEntityCfg, style_mask: list[int] | None = None) -> torch.Tensor:
    """Reward low foot vertical speed when near ground and not in contact."""
    asset: Articulation = env.scene["robot"]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Get foot heights (simplified - using body z position)
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
    
    # Initialize last_contacts if not exists
    if not hasattr(env, 'last_contacts_buffer'):
        env.last_contacts_buffer = torch.zeros_like(contact)
    
    contact_filt = torch.logical_or(contact, env.last_contacts_buffer)
    env.last_contacts_buffer = contact
    
    rew_left_fspd *= ~contact_filt[:, 0]
    rew_right_fspd *= ~contact_filt[:, 1]
    
    return _apply_style_mask(env, rew_left_fspd + rew_right_fspd, style_mask)


# ============================================================================
# Gait rewards
# ============================================================================

def gait_feet_frc_perio(env: UltraMotionRunEnv, delta_t: float = 0.02, style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize foot force during the swing phase of the gait."""
    left_frc_swing_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[0]
    right_frc_swing_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[0]
    left_frc_score = left_frc_swing_mask * (torch.exp(-200 * torch.square(env.avg_feet_force_per_step[:, 0])))
    right_frc_score = right_frc_swing_mask * (torch.exp(-200 * torch.square(env.avg_feet_force_per_step[:, 1])))
    return _apply_style_mask(env, left_frc_score + right_frc_score, style_mask)


def gait_feet_spd_perio(env: UltraMotionRunEnv, delta_t: float = 0.02, style_mask: list[int] | None = None) -> torch.Tensor:
    """Penalize foot speed during the support phase of the gait."""
    left_spd_support_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[1]
    right_spd_support_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[1]
    left_spd_score = left_spd_support_mask * (torch.exp(-10 * torch.square(env.avg_feet_speed_per_step[:, 0])))
    right_spd_score = right_spd_support_mask * (torch.exp(-10 * torch.square(env.avg_feet_speed_per_step[:, 1])))
    return _apply_style_mask(env, left_spd_score + right_spd_score, style_mask)


def gait_feet_frc_support_perio(env: UltraMotionRunEnv, delta_t: float = 0.02, style_mask: list[int] | None = None) -> torch.Tensor:
    """Reward that promotes proper support force during stance (support) phase."""
    left_frc_support_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[1]
    right_frc_support_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[1]
    left_frc_score = left_frc_support_mask * (1 - torch.exp(-10 * torch.square(env.avg_feet_force_per_step[:, 0])))
    right_frc_score = right_frc_support_mask * (1 - torch.exp(-10 * torch.square(env.avg_feet_force_per_step[:, 1])))
    return _apply_style_mask(env, left_frc_score + right_frc_score, style_mask)


# ============================================================================
# Velocity tracking rewards
# ============================================================================

def x_vel_exp(env: UltraMotionRunEnv, style_mask: list[int] | None = None) -> torch.Tensor:
    """Reward for tracking forward velocity."""
    asset: Articulation = env.scene["robot"]
    cmd = env.command_generator.command
    base_lin_vel = asset.data.root_lin_vel_b
    
    flag = torch.abs(cmd[:, 0]) < 0.1
    x_vel_error = (flag * torch.abs(cmd[:, 0] - base_lin_vel[:, 0]) +
                   ~flag * torch.abs(cmd[:, 0] - env.avg_x_vel))
    reward = torch.exp(-2 * x_vel_error)
    return _apply_style_mask(env, reward, style_mask)


def y_vel_exp(env: UltraMotionRunEnv, style_mask: list[int] | None = None) -> torch.Tensor:
    """Reward for tracking lateral velocity."""
    cmd = env.command_generator.command
    reward = torch.exp(-10 * torch.abs(cmd[:, 1] - env.avg_y_vel))
    return _apply_style_mask(env, reward, style_mask)


def z_vel_exp(env: UltraMotionRunEnv, style_mask: list[int] | None = None) -> torch.Tensor:
    """Reward for penalizing vertical velocity."""
    asset: Articulation = env.scene["robot"]
    base_lin_vel = asset.data.root_lin_vel_b
    reward = torch.exp(-2 * torch.abs(base_lin_vel[:, 2]))
    return _apply_style_mask(env, reward, style_mask)


def ang_vel_exp(env: UltraMotionRunEnv, style_mask: list[int] | None = None) -> torch.Tensor:
    """Reward for tracking angular velocity."""
    asset: Articulation = env.scene["robot"]
    cmd = env.command_generator.command
    base_ang_vel = asset.data.root_ang_vel_b
    
    flag = torch.abs(cmd[:, 2]) < 0.1
    ang_vel_error = (flag * torch.abs(cmd[:, 2] - base_ang_vel[:, 2]) +
                     ~flag * torch.abs(cmd[:, 2] - env.avg_yaw_vel))
    reward = torch.exp(-3 * ang_vel_error)
    return _apply_style_mask(env, reward, style_mask)


def ang_vel_xy_exp(env: UltraMotionRunEnv, style_mask: list[int] | None = None) -> torch.Tensor:
    """Reward for penalizing roll and pitch angular velocities."""
    asset: Articulation = env.scene["robot"]
    base_ang_vel = asset.data.root_ang_vel_b
    ang_vel_error = torch.sum(torch.square(base_ang_vel[:, :2]), dim=-1)
    reward = torch.exp(-2 * ang_vel_error)
    return _apply_style_mask(env, reward, style_mask)


def orientation_exp(env: UltraMotionRunEnv, style_mask: list[int] | None = None) -> torch.Tensor:
    """Reward for maintaining proper orientation."""
    asset: Articulation = env.scene["robot"]
    cmd = env.command_generator.command
    projected_gravity = asset.data.projected_gravity_b
    error = torch.square(0.03 * cmd[:, 0].clip(min=0.) - projected_gravity[:, 0])
    error += torch.square(projected_gravity[:, 1])
    return _apply_style_mask(env, torch.exp(-100 * error), style_mask)

