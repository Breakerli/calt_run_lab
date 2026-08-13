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

from collections.abc import Sequence

import torch
import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg


class CustomVelocityCommand(UniformVelocityCommand):
    """自定义速度命令：站/走/跑分桶采样；run 高速段；阶跃/斜坡混合；线速度加速度上限。"""

    cfg: UniformVelocityCommandCfg

    def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.cmd_target_b = torch.zeros(self.num_envs, 3, device=self.device)

    def _cmd_cfg(self):
        env_cfg = getattr(self._env, "cfg", None)
        return getattr(env_cfg, "commands", None) if env_cfg is not None else None

    def _style_thresholds(self) -> tuple[float, float, float]:
        style_thr = getattr(getattr(self._env, "cfg", None), "style_from_command", None)
        stand_enter = float(getattr(style_thr, "stand_enter", 0.08))
        stand_exit = float(getattr(style_thr, "stand_exit", 0.12))
        run_enter = float(getattr(style_thr, "run_enter", 1.0))
        return stand_enter, stand_exit, run_enter

    def _apply_near_zero_as_stand(self, env_ids_t: torch.Tensor) -> None:
        """vx、vy、vyaw 均接近 0 时强制站定（与站桶一致）。"""
        stand_enter, _, _ = self._style_thresholds()
        cmd = self.vel_command_b[env_ids_t]
        near_zero = (
            (torch.abs(cmd[:, 0]) < stand_enter)
            & (torch.abs(cmd[:, 1]) < stand_enter)
            & (torch.abs(cmd[:, 2]) < stand_enter)
        )
        if not torch.any(near_zero):
            return
        stand_ids = env_ids_t[near_zero]
        self.vel_command_b[stand_ids, :] = 0.0
        self.is_standing_env[stand_ids] = True
        self.is_heading_env[stand_ids] = False

    def _apply_cmd_accel_limit(self, dt: float) -> None:
        """将 vel_command_b 以 max_lin_accel 向 cmd_target_b 逼近（仅非 heading 的 vz）。"""
        cmd_cfg = self._cmd_cfg()
        max_accel = float(getattr(cmd_cfg, "max_lin_accel", 0.0))
        if max_accel <= 0.0:
            return

        max_dv = max_accel * dt
        for dim in (0, 1):
            delta = self.cmd_target_b[:, dim] - self.vel_command_b[:, dim]
            self.vel_command_b[:, dim] += torch.clamp(delta, -max_dv, max_dv)

        non_heading = ~self.is_heading_env
        if torch.any(non_heading):
            delta_z = self.cmd_target_b[:, 2] - self.vel_command_b[:, 2]
            self.vel_command_b[non_heading, 2] += torch.clamp(delta_z[non_heading], -max_dv, max_dv)

    def _sample_run_bucket_vx(
        self,
        count: int,
        lo_run: float,
        vmax: float,
        threshold_frac: float,
        high_frac: float,
    ) -> torch.Tensor:
        """run 桶：vx ~ U[lo_run, vmax]；其中 high_frac 落在 [threshold*vmax, vmax]。"""
        if count <= 0:
            return torch.empty(0, device=self.device)
        lo_run = float(lo_run)
        vmax = float(vmax)
        v_high = max(threshold_frac * vmax, lo_run)

        if vmax <= lo_run + 1e-6:
            return torch.full((count,), vmax, device=self.device)
        if v_high >= vmax - 1e-6:
            return torch.empty(count, device=self.device).uniform_(lo_run, vmax)

        high_mask = torch.rand(count, device=self.device) < high_frac
        vx = torch.empty(count, device=self.device)
        n_high = int(high_mask.sum().item())
        n_low = count - n_high
        if n_high > 0:
            vx[high_mask] = torch.empty(n_high, device=self.device).uniform_(v_high, vmax)
        if n_low > 0:
            vx[~high_mask] = torch.empty(n_low, device=self.device).uniform_(lo_run, v_high)
        return vx

    def _resample_bucket_lin_vel_x(self, env_ids_t: torch.Tensor) -> None:
        """站/走/跑 1:2:7 分桶采样。

        - 站桶：三轴 0
        - 走桶：vx ~ U[walk_lo, walk_hi]
        - 跑桶：vx ~ U[run_min, vmax]，50% 高速段
        - 采样后若 vx、vy、vyaw 均 < stand_enter → 置站
        """
        cmd_cfg = self._cmd_cfg()
        probs = getattr(cmd_cfg, "cmd_bucket_probs", None)
        if probs is None:
            return

        vmin = float(self.cfg.ranges.lin_vel_x[0])
        vmax = float(self.cfg.ranges.lin_vel_x[1])
        walk_lo, walk_hi = getattr(cmd_cfg, "cmd_bucket_walk_lin_vel_x", (-0.5, 1.0))
        run_min = float(getattr(cmd_cfg, "cmd_bucket_run_lin_vel_x_min", 1.0))

        p = torch.tensor(probs, device=self.device, dtype=torch.float32)
        p = p / p.sum()
        n = len(env_ids_t)
        bucket = torch.multinomial(p, n, replacement=True)

        vx = torch.zeros(n, device=self.device)
        is_standing = bucket == 0

        walk_mask = bucket == 1
        if torch.any(walk_mask):
            count = int(walk_mask.sum().item())
            lo = max(float(walk_lo), vmin)
            hi = min(float(walk_hi), vmax)
            if hi <= lo + 1e-6:
                vx[walk_mask] = lo
            else:
                vx[walk_mask] = torch.empty(count, device=self.device).uniform_(lo, hi)

        run_mask = bucket == 2
        if torch.any(run_mask):
            count = int(run_mask.sum().item())
            threshold_frac = float(getattr(cmd_cfg, "run_high_speed_threshold_frac", 0.7))
            high_frac = float(getattr(cmd_cfg, "run_high_speed_sample_frac", 0.5) or 0.5)
            lo_run = max(run_min, vmin)
            hi_run = vmax
            if hi_run <= lo_run + 1e-6:
                vx[run_mask] = hi_run
            else:
                vx[run_mask] = self._sample_run_bucket_vx(count, lo_run, hi_run, threshold_frac, high_frac)

        self.vel_command_b[env_ids_t, 0] = vx
        self.is_standing_env[env_ids_t] = is_standing
        if torch.any(is_standing):
            stand_ids = env_ids_t[is_standing]
            self.vel_command_b[stand_ids, :] = 0.0
            self.is_heading_env[stand_ids] = False

    def _resample_run_high_speed_commands(self, env_ids: Sequence[int]) -> None:
        """run style: 部分 env 采样 [threshold*vmax, vmax]；run env 会强制 is_standing_env=False."""
        cmd_cfg = self._cmd_cfg()
        if getattr(cmd_cfg, "cmd_bucket_probs", None) is not None:
            return
        sample_frac = getattr(cmd_cfg, "run_high_speed_sample_frac", None)
        if sample_frac is None:
            return

        threshold_frac = getattr(cmd_cfg, "run_high_speed_threshold_frac", 0.7)
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).view(-1)
        style_thr = getattr(getattr(self._env, "cfg", None), "style_from_command", None)
        run_enter = float(getattr(style_thr, "run_enter", 1.05))
        run_mask = torch.abs(self.cmd_target_b[env_ids_t, 0]) >= run_enter
        if not torch.any(run_mask):
            return

        run_env_ids = env_ids_t[run_mask]
        self.is_standing_env[run_env_ids] = False

        vmax = float(self.cfg.ranges.lin_vel_x[1])
        if vmax <= 0.0:
            return
        v_high = threshold_frac * vmax
        if v_high >= vmax:
            self.vel_command_b[run_env_ids, 0] = vmax
            return

        rand = torch.rand(len(run_env_ids), device=self.device)
        high_mask = rand < sample_frac
        if not torch.any(high_mask):
            return
        high_ids = run_env_ids[high_mask]
        self.vel_command_b[high_ids, 0] = torch.empty(len(high_ids), device=self.device).uniform_(v_high, vmax)

    def _resample_non_bucket_commands(self, env_ids_t: torch.Tensor) -> None:
        """vy / wz / heading 采样（与 Isaac Lab UniformVelocityCommand 一致）。"""
        r = torch.empty(len(env_ids_t), device=self.device)
        self.vel_command_b[env_ids_t, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        self.vel_command_b[env_ids_t, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        if self.cfg.heading_command:
            self.heading_target[env_ids_t] = r.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids_t] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs

    def _force_lateral_or_yaw_command(self, env_ids_t: torch.Tensor) -> None:
        """Ensure selected walk samples request meaningful lateral or yaw motion."""
        if len(env_ids_t) == 0:
            return
        _, stand_exit, _ = self._style_thresholds()
        y_lo, y_hi = self.cfg.ranges.lin_vel_y
        w_lo, w_hi = self.cfg.ranges.ang_vel_z
        min_abs = min(
            max(stand_exit, 0.25 * min(max(abs(float(y_lo)), abs(float(y_hi))), max(abs(float(w_lo)), abs(float(w_hi))))),
            max(max(abs(float(y_lo)), abs(float(y_hi))), max(abs(float(w_lo)), abs(float(w_hi)))),
        )
        cmd = self.vel_command_b[env_ids_t]
        active = (torch.abs(cmd[:, 1]) >= min_abs) | (torch.abs(cmd[:, 2]) >= min_abs)
        if torch.all(active):
            return

        inactive_ids = env_ids_t[~active]
        choose_y = torch.rand(len(inactive_ids), device=self.device) < 0.5
        signs = torch.where(torch.rand(len(inactive_ids), device=self.device) < 0.5, -1.0, 1.0)

        if torch.any(choose_y):
            ids = inactive_ids[choose_y]
            max_y = max(abs(float(y_lo)), abs(float(y_hi)))
            if max_y > 0.0:
                lo_abs = min(min_abs, max_y)
                val = torch.empty(len(ids), device=self.device).uniform_(lo_abs, max_y) * signs[choose_y]
                self.vel_command_b[ids, 1] = torch.clamp(val, min=float(y_lo), max=float(y_hi))

        if torch.any(~choose_y):
            ids = inactive_ids[~choose_y]
            max_w = max(abs(float(w_lo)), abs(float(w_hi)))
            if max_w > 0.0:
                lo_abs = min(min_abs, max_w)
                val = torch.empty(len(ids), device=self.device).uniform_(lo_abs, max_w) * signs[~choose_y]
                self.vel_command_b[ids, 2] = torch.clamp(val, min=float(w_lo), max=float(w_hi))

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).view(-1)
        prev_cmd = self.vel_command_b[env_ids_t].clone()
        cmd_cfg = self._cmd_cfg()

        if getattr(cmd_cfg, "cmd_bucket_probs", None) is not None:
            self._resample_bucket_lin_vel_x(env_ids_t)
            move_mask = ~self.is_standing_env[env_ids_t]
            if torch.any(move_mask):
                move_ids = env_ids_t[move_mask]
                self._resample_non_bucket_commands(move_ids)
                lateral_yaw_frac = float(getattr(cmd_cfg, "cmd_bucket_walk_lateral_yaw_frac", 0.0))
                if lateral_yaw_frac > 0.0:
                    run_min = float(getattr(cmd_cfg, "cmd_bucket_run_lin_vel_x_min", 1.0))
                    walk_ids = move_ids[torch.abs(self.vel_command_b[move_ids, 0]) < run_min]
                    lateral_yaw_mask = torch.rand(len(walk_ids), device=self.device) < lateral_yaw_frac
                    lateral_yaw_ids = walk_ids[lateral_yaw_mask]
                    if len(lateral_yaw_ids) > 0:
                        self.is_heading_env[lateral_yaw_ids] = False
                        self._force_lateral_or_yaw_command(lateral_yaw_ids)
            self._apply_near_zero_as_stand(env_ids_t)
        else:
            super()._resample_command(env_ids)
            self._resample_run_high_speed_commands(env_ids)

        self.cmd_target_b[env_ids_t] = self.vel_command_b[env_ids_t].clone()

        step_frac = float(getattr(cmd_cfg, "cmd_step_sample_frac", 0.5))
        if step_frac < 1.0:
            ramp_mask = torch.rand(len(env_ids_t), device=self.device) >= step_frac
            if torch.any(ramp_mask):
                ramp_ids = env_ids_t[ramp_mask]
                self.vel_command_b[ramp_ids] = prev_cmd[ramp_mask]

    def _update_command(self) -> None:
        dt = float(self._env.step_dt)
        self._apply_cmd_accel_limit(dt)
        super()._update_command()
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        if len(standing_env_ids) > 0:
            self.cmd_target_b[standing_env_ids] = 0.0

    def _debug_vis_callback(self, event):
        """绿=指令、蓝=实际速度；箭头沿 yaw 系水平方向，随机器人航向转动。"""
        if not self.robot.is_initialized:
            return
        if not hasattr(self, "goal_vel_visualizer"):
            return

        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5

        yaw_q = math_utils.yaw_quat(self.robot.data.root_quat_w)
        vel_yaw_xy = math_utils.quat_apply_inverse(yaw_q, self.robot.data.root_lin_vel_w)[:, :2]

        vel_des_scale, vel_des_quat = self._resolve_xy_velocity_to_arrow_yaw(self.command[:, :2], yaw_q)
        vel_scale, vel_quat = self._resolve_xy_velocity_to_arrow_yaw(vel_yaw_xy, yaw_q)

        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_quat, vel_des_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_quat, vel_scale)

    def _resolve_xy_velocity_to_arrow_yaw(
        self, xy_velocity: torch.Tensor, yaw_q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """yaw 系平面速度 -> 世界系水平箭头（仅绕 z，随机器人转向）。"""
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 1.0

        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        arrow_quat = math_utils.quat_mul(yaw_q, arrow_quat)
        return arrow_scale, arrow_quat
