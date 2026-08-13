# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Joint-state CSV logging (shared by play.py and sim2sim_ultraYaw.py)."""

from __future__ import annotations

import atexit
import csv
import os
from typing import IO, TYPE_CHECKING

import numpy as np
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env import BaseEnv
    from legged_lab.envs.ultra_js_wb.ultra_motion_run_env import UltraMotionRunEnv


def joint_log_csv_header(joint_names: list[str]) -> list[str]:
    """CSV columns aligned with plot_sim2sim_ultra_yaw_csv.py."""
    header = [
        "time",
        "base_vx",
        "base_vy",
        "base_vz",
        "base_wx",
        "base_wy",
        "base_wz",
        "base_z",
        "cmd_vx",
        "cmd_vy",
        "cmd_wz",
    ]
    for name in joint_names:
        header.extend(
            [
                f"q_{name}",
                f"qd_{name}",
                f"tau_{name}",
                f"q_min_{name}",
                f"q_max_{name}",
                f"qd_min_{name}",
                f"qd_max_{name}",
                f"tau_min_{name}",
                f"tau_max_{name}",
            ]
        )
    return header


def root_vel_yaw_frame_np(robot) -> tuple[np.ndarray, np.ndarray]:
    """Root linear/angular velocity in yaw frame (env 0), same as LCM / track_lin_vel_*."""
    root_quat = robot.data.root_quat_w[0:1]
    yaw_q = yaw_quat(root_quat)
    lin_yaw = quat_apply_inverse(yaw_q, robot.data.root_lin_vel_w[0:1])[0].detach().cpu().numpy()
    ang_yaw = quat_apply_inverse(yaw_q, robot.data.root_ang_vel_w[0:1])[0].detach().cpu().numpy()
    return lin_yaw, ang_yaw


def _play_command_np(env) -> np.ndarray:
    """Active velocity command for env 0 (matches play.py vel_command_b pinning)."""
    cg = env.command_generator
    if hasattr(cg, "vel_command_b"):
        return cg.vel_command_b[0].detach().cpu().numpy()
    return cg.command[0].detach().cpu().numpy()


class PlayJointStateLogger:
    """Per-step joint state CSV during Isaac Lab play (format matches sim2sim --log-torques).

    Rows are appended and flushed immediately so data survives Ctrl+C / window close.
    """

    def __init__(self, env: BaseEnv | UltraMotionRunEnv, output_path: str) -> None:
        self.env = env
        self.output_path = os.path.abspath(output_path)
        self.robot = env.robot
        self.joint_names = [str(n) for n in self.robot.data.joint_names]
        self._row_count = 0
        self._closed = False
        self._file: IO[str] | None = None
        self._writer: csv.writer | None = None
        self._init_limits()

        out_dir = os.path.dirname(self.output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        self._file = open(self.output_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(joint_log_csv_header(self.joint_names))
        self._file.flush()
        atexit.register(self.close)

    def _init_limits(self) -> None:
        n = len(self.joint_names)
        pos_lim = self.robot.data.joint_pos_limits[0].detach().cpu().numpy()
        self._q_min = pos_lim[:, 0].astype(np.float64)
        self._q_max = pos_lim[:, 1].astype(np.float64)

        vel_lim = getattr(self.robot.data, "joint_vel_limits", None)
        if vel_lim is not None:
            vel_lim = vel_lim[0].detach().cpu().numpy()
            self._qd_min = (-np.abs(vel_lim)).astype(np.float64)
            self._qd_max = np.abs(vel_lim).astype(np.float64)
        else:
            self._qd_min = np.full(n, -999.0, dtype=np.float64)
            self._qd_max = np.full(n, 999.0, dtype=np.float64)

        eff_lim = getattr(self.robot.data, "joint_effort_limits", None)
        if eff_lim is not None:
            eff_lim = eff_lim[0].detach().cpu().numpy()
            self._tau_min = (-np.abs(eff_lim)).astype(np.float64)
            self._tau_max = np.abs(eff_lim).astype(np.float64)
        else:
            self._tau_min = np.full(n, -999.0, dtype=np.float64)
            self._tau_max = np.full(n, 999.0, dtype=np.float64)

        if n != self._q_min.shape[0]:
            raise RuntimeError(
                f"joint_names len {n} != limits len {self._q_min.shape[0]}; check articulation."
            )

    def _build_row(self) -> list[float]:
        lin_yaw, ang_yaw = root_vel_yaw_frame_np(self.robot)
        cmd = _play_command_np(self.env)
        sim_t = float(self.env.episode_length_buf[0].item()) * float(self.env.step_dt)

        q = self.robot.data.joint_pos[0].detach().cpu().numpy()
        qd = self.robot.data.joint_vel[0].detach().cpu().numpy()
        tau = self.robot.data.applied_torque[0].detach().cpu().numpy()
        base_z = float(self.robot.data.root_pos_w[0, 2].detach().cpu().item())

        row: list[float] = [
            sim_t,
            float(lin_yaw[0]),
            float(lin_yaw[1]),
            float(lin_yaw[2]),
            float(ang_yaw[0]),
            float(ang_yaw[1]),
            float(ang_yaw[2]),
            base_z,
            float(cmd[0]),
            float(cmd[1]),
            float(cmd[2]),
        ]
        for i in range(len(self.joint_names)):
            row.extend(
                [
                    float(q[i]),
                    float(qd[i]),
                    float(tau[i]),
                    float(self._q_min[i]),
                    float(self._q_max[i]),
                    float(self._qd_min[i]),
                    float(self._qd_max[i]),
                    float(self._tau_min[i]),
                    float(self._tau_max[i]),
                ]
            )
        return row

    def record(self) -> None:
        """Append one row after env.step (uses env 0); flushed to disk immediately."""
        if self._closed or self._writer is None or self._file is None:
            return
        self._writer.writerow(self._build_row())
        self._file.flush()
        self._row_count += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
        self._writer = None
        if self._row_count > 0:
            print(f"[INFO] Wrote {self._row_count} joint-state samples to {self.output_path}")
        else:
            print(
                f"[WARN] Joint log has 0 samples (no env.step before exit?): {self.output_path}"
            )

    def save(self) -> None:
        """Alias for close(); kept for play.py finally / sim2sim naming."""
        self.close()
