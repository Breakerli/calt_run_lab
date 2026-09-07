from __future__ import annotations

from typing import Optional, Tuple

import torch

from rsl_rl.env import VecEnv


def _to_list(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return value.tolist()
    return list(value)


def _swap_pairs(tensor: torch.Tensor, left: list[int], right: list[int], offset: int = 0) -> None:
    if not left or not right or len(left) != len(right):
        return
    left_idx = [offset + i for i in left]
    right_idx = [offset + i for i in right]
    temp = tensor[:, left_idx].clone()
    tensor[:, left_idx] = tensor[:, right_idx]
    tensor[:, right_idx] = temp


def _negate_indices(tensor: torch.Tensor, indices: list[int], offset: int = 0) -> None:
    if not indices:
        return
    idx = [offset + i for i in indices]
    tensor[:, idx] *= -1.0


def _get_joint_mirror_ids(env: VecEnv) -> tuple[list[int], list[int], list[int]]:
    """Build (left, right, negate) lists for the left/right body-symmetry transform.

    - left/right: joints that get column-swapped left↔right. Built from leg + arm.
      Waist 不在 swap 集合 (中线单链), 仅参与 negate.
    - negate: 关节角在镜像后需要变号的 roll / yaw 轴. 优先按 *named* id 字段查 (env
      __init__ 用 find_joints 显式拿到), 这样不依赖 left_leg_ids / left_arm_ids 的
      内部排序; 如果 env 没暴露这些字段, 退化到旧的位置硬编码 (假设腿 = [hip_roll,
      hip_yaw, hip_pitch, knee, ankle_pitch, ankle_roll]).
    """
    # 完整镜像用 env.left/right_mirror_ids（覆盖整机所有左右关节）；缺省时退化到
    # left_leg + left_arm（ARM 特征关节，AMP 布局要求 arm 只含 1 个 pitch 关节）。
    left = _to_list(getattr(env, "left_mirror_ids", None))
    right = _to_list(getattr(env, "right_mirror_ids", None))
    if not left or not right:
        left_leg = _to_list(getattr(env, "left_leg_ids", None))
        right_leg = _to_list(getattr(env, "right_leg_ids", None))
        left_arm = _to_list(getattr(env, "left_arm_ids", None))
        right_arm = _to_list(getattr(env, "right_arm_ids", None))
        left = left_leg + left_arm
        right = right_leg + right_arm

    # 优先用按名字注册的 id (env.{hip_roll,hip_yaw,ankle_roll,...}_ids).
    # ─── 腿 ───
    hip_roll = _to_list(getattr(env, "hip_roll_ids", None))
    hip_yaw = _to_list(getattr(env, "hip_yaw_ids", None))
    ankle_roll = _to_list(getattr(env, "ankle_roll_ids", None))
    # ─── 手臂 (仅当 robot 有 roll/yaw 自由度时才存在; 2-DOF 全 pitch 的不会暴露) ───
    shoulder_roll = _to_list(getattr(env, "shoulder_roll_ids", None))
    shoulder_yaw = _to_list(getattr(env, "shoulder_yaw_ids", None))
    elbow_yaw = _to_list(getattr(env, "elbow_yaw_ids", None))
    wrist_roll = _to_list(getattr(env, "wrist_roll_ids", None))
    wrist_yaw = _to_list(getattr(env, "wrist_yaw_ids", None))
    # ─── 腰 (中线; 仅 yaw / roll 在镜像下变号) ───
    waist_roll = _to_list(getattr(env, "waist_roll_ids", None))
    waist_yaw = _to_list(getattr(env, "waist_yaw_ids", None))

    # 只要任一按名字字段存在就走 named 分支 (3 个新 env 都满足).
    use_named = bool(hip_roll or hip_yaw or ankle_roll
                     or shoulder_roll or shoulder_yaw or elbow_yaw
                     or wrist_roll or wrist_yaw
                     or waist_roll or waist_yaw)

    if use_named:
        negate = (
            list(hip_roll) + list(hip_yaw) + list(ankle_roll)
            + list(shoulder_roll) + list(shoulder_yaw) + list(elbow_yaw)
            + list(wrist_roll) + list(wrist_yaw)
            + list(waist_roll) + list(waist_yaw)
        )
    else:
        # legacy fallback: 假设 left_leg = [hip_roll, hip_yaw, hip_pitch, knee, ankle_pitch, ankle_roll].
        # 注意: 这里不会处理 arm/waist 的 roll/yaw, 留给后续把这些 env 也补上 named id 字段.
        left_leg = _to_list(getattr(env, "left_leg_ids", None))
        right_leg = _to_list(getattr(env, "right_leg_ids", None))
        negate = []
        if len(left_leg) >= 6:
            negate.extend([left_leg[0], left_leg[1], left_leg[5]])
        if len(right_leg) >= 6:
            negate.extend([right_leg[0], right_leg[1], right_leg[5]])
    return left, right, negate


def _get_actor_base_dim(env: VecEnv) -> int:
    num_actions = int(getattr(env, "num_actions"))
    # ang(3)+grav(3)+cmd(3) + dof_pos/vel/action (no gait phase, no style one-hot)
    return 9 + num_actions * 3


def _get_critic_extra_dim(env: VecEnv) -> int:
    num_actions = int(getattr(env, "num_actions", 13))
    return int(getattr(env, "critic_extra_dim", 15 + 3 * num_actions))


def _get_critic_base_dim(env: VecEnv) -> int:
    return _get_actor_base_dim(env) + _get_critic_extra_dim(env)


def _mirror_critic_extra(obs: torch.Tensor, env: VecEnv, offset: int) -> None:
    """Mirror nogait-style critic extras after actor block."""
    num_actions = int(getattr(env, "num_actions"))
    left, right, negate = _get_joint_mirror_ids(env)

    o = offset
    # root_lin_vel_yaw: flip y
    obs[:, o + 2] *= -1.0
    # feet_vel_z, foot_force_z, contact, foot_clearance: swap L/R
    for seg_start in (o + 4, o + 6, o + 8, o + 11):
        _swap_pairs(obs, [0], [1], offset=seg_start)
    # domain_rand: friction, restitution invariant; then p, d, motor per joint
    d0 = o + 13
    for seg in (d0 + 2, d0 + 2 + num_actions, d0 + 2 + 2 * num_actions):
        _swap_pairs(obs, left, right, offset=seg)
        _negate_indices(obs, negate, offset=seg)


def _split_history(obs: torch.Tensor, history_len: int, base_dim: int) -> tuple[torch.Tensor, Optional[torch.Tensor], int, int]:
    obs_dim = obs.shape[1]
    if history_len <= 0:
        history_len = 1
    history_dim = history_len * base_dim
    if history_dim > obs_dim:
        if obs_dim % history_len == 0:
            base_dim = obs_dim // history_len
            history_dim = obs_dim
        else:
            history_len = 1
            base_dim = obs_dim
            history_dim = obs_dim
    extra = obs[:, history_dim:] if history_dim < obs_dim else None
    history = obs[:, :history_dim].reshape(obs.shape[0], history_len, base_dim)
    return history, extra, history_len, base_dim


def _mirror_actor_current(obs: torch.Tensor, env: VecEnv) -> torch.Tensor:
    obs_m = obs.clone()
    num_actions = int(getattr(env, "num_actions"))

    # ang_vel (x,z) sign flip
    obs_m[:, 0] *= -1.0
    obs_m[:, 2] *= -1.0
    # projected_gravity (y) sign flip
    obs_m[:, 4] *= -1.0
    # command: lin_vel_y, ang_vel_z sign flip
    obs_m[:, 7] *= -1.0
    obs_m[:, 8] *= -1.0

    # joint segments
    jpos_start = 9
    jvel_start = jpos_start + num_actions
    act_start = jvel_start + num_actions

    left, right, negate = _get_joint_mirror_ids(env)
    _swap_pairs(obs_m, left, right, offset=jpos_start)
    _swap_pairs(obs_m, left, right, offset=jvel_start)
    _swap_pairs(obs_m, left, right, offset=act_start)

    _negate_indices(obs_m, negate, offset=jpos_start)
    _negate_indices(obs_m, negate, offset=jvel_start)
    _negate_indices(obs_m, negate, offset=act_start)
    return obs_m


def _mirror_critic_current(obs: torch.Tensor, env: VecEnv) -> torch.Tensor:
    obs_m = obs.clone()
    actor_dim = _get_actor_base_dim(env)
    if obs_m.shape[1] < actor_dim:
        return obs_m

    obs_m[:, :actor_dim] = _mirror_actor_current(obs_m[:, :actor_dim], env)
    extra_dim = _get_critic_extra_dim(env)
    if obs_m.shape[1] >= actor_dim + extra_dim:
        _mirror_critic_extra(obs_m, env, actor_dim)
    return obs_m


def _mirror_height_scan_extra(extra: torch.Tensor, env: VecEnv) -> torch.Tensor:
    """Mirror flattened base height scan (flip grid y, nogait body symmetry)."""
    perm = getattr(env, "height_scan_mirror_indices", None)
    if perm is None or extra.shape[1] != perm.numel():
        return extra.clone()
    return extra[:, perm]


def _mirror_actions(actions: torch.Tensor, env: VecEnv) -> torch.Tensor:
    act_m = actions.clone()
    left, right, negate = _get_joint_mirror_ids(env)
    _swap_pairs(act_m, left, right, offset=0)
    _negate_indices(act_m, negate, offset=0)
    return act_m


@torch.no_grad()
def data_augmentation_func(
    obs: Optional[torch.Tensor] = None,
    actions: Optional[torch.Tensor] = None,
    env: VecEnv = None,
    obs_type: str = "policy",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetry-based data augmentation.

    Args:
        obs: Observation tensor of shape [B, obs_dim].
        actions: Action tensor of shape [B, act_dim].
        env: Vectorized env instance.
        obs_type: "policy" or "critic".

    Returns:
        (augmented_obs, augmented_actions)
    """

    if obs is None:
        actions_mirror = _mirror_actions(actions, env)
        aug_actions = torch.cat([actions, actions_mirror], dim=0)
        return None, aug_actions

    if obs_type == "critic":
        history_len = int(getattr(env.cfg.robot, "critic_obs_history_length", 1))
        base_dim = _get_critic_base_dim(env)
        history, extra, history_len, base_dim = _split_history(obs, history_len, base_dim)
        flat = history.reshape(-1, base_dim)
        mirrored = _mirror_critic_current(flat, env).reshape(obs.shape[0], history_len, base_dim)
        obs_mirror = mirrored.reshape(obs.shape[0], history_len * base_dim)

        if actions is not None:
            next_history, next_extra, _, _ = _split_history(actions, history_len, base_dim)
            next_flat = next_history.reshape(-1, base_dim)
            next_mirrored = _mirror_critic_current(next_flat, env).reshape(obs.shape[0], history_len, base_dim)
            next_obs_mirror = next_mirrored.reshape(obs.shape[0], history_len * base_dim)
            if next_extra is not None:
                next_obs_mirror = torch.cat([next_obs_mirror, next_extra], dim=-1)
            if extra is not None:
                obs_mirror = torch.cat([obs_mirror, extra], dim=-1)
            aug_obs = torch.cat([obs, obs_mirror], dim=0)
            aug_next = torch.cat([actions, next_obs_mirror], dim=0)
            return aug_obs, aug_next
    elif obs_type == "policy":
        history_len = int(getattr(env.cfg.robot, "actor_obs_history_length", 1))
        base_dim = _get_actor_base_dim(env)
        history, extra, history_len, base_dim = _split_history(obs, history_len, base_dim)
        flat = history.reshape(-1, base_dim)
        mirrored = _mirror_actor_current(flat, env).reshape(obs.shape[0], history_len, base_dim)
        obs_mirror = mirrored.reshape(obs.shape[0], history_len * base_dim)
    else:
        raise ValueError(f"Invalid obs_type: {obs_type}")

    if extra is not None:
        if obs_type == "critic":
            extra = _mirror_height_scan_extra(extra, env)
        obs_mirror = torch.cat([obs_mirror, extra], dim=-1)

    actions_mirror = _mirror_actions(actions, env) if actions is not None else None
    aug_obs = torch.cat([obs, obs_mirror], dim=0)
    aug_actions = torch.cat([actions, actions_mirror], dim=0) if actions is not None else None
    return aug_obs, aug_actions
