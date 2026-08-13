#!/usr/bin/env python3
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Collect AMP expert motion from play (policy rollout in Isaac Lab).

Run trained policy, record ``env.get_amp_obs_for_expert_trans()`` each step,
save JSON motion files for Multi-AMP. Command handling matches play.py: set
``--vx/--vy/--wz`` (or edit defaults below), pin every step after ``env.step``.

Usage (single line — do not use ``\\\\`` line break, or bash drops ``--duration``):

  python legged_lab/scripts/collect_play_motion_data.py --task ultra_game_yaw_run --duration 15

  python legged_lab/scripts/collect_play_motion_data.py --task ultra_game_yaw_run --headless --duration 20 --vx 10 --single-file -o legged_lab/envs/ultra_game_yaw/datasets/motion_amp_expert/play_run_10mps.txt
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher

from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.utils import task_registry

import legged_lab.utils.cli_args as cli_args  # isort: skip

# Same defaults as play.py (override via --vx/--vy/--wz)
DEFAULT_CMD_VX = 15.0
DEFAULT_CMD_VY = 0.0
DEFAULT_CMD_WZ = 0.0
DEFAULT_DURATION_S = 10.0
DEFAULT_WARMUP_STEPS = 100


def _strip_argv_value(flag: str) -> float | int | None:
    """Pop ``--flag VALUE`` or ``--flag=VALUE`` from sys.argv (before AppLauncher).

    Isaac Lab / shell line breaks often drop trailing args; stripping early matches
    2026gym collect_omnidirectional_run_data and avoids ``--duration`` being lost.
    """
    for i, arg in enumerate(list(sys.argv)):
        if arg == flag:
            if i + 1 >= len(sys.argv):
                raise SystemExit(f"[ERROR] {flag} requires a value")
            val = sys.argv[i + 1]
            del sys.argv[i : i + 2]
            return val
        prefix = flag + "="
        if arg.startswith(prefix):
            val = arg[len(prefix) :]
            del sys.argv[i]
            return val
    return None


# Parse before AppLauncher (same rationale as 2026gym collect script)
_duration_raw = _strip_argv_value("--duration")
_warmup_raw = _strip_argv_value("--warmup")
COLLECT_DURATION_S = float(_duration_raw) if _duration_raw is not None else DEFAULT_DURATION_S
COLLECT_WARMUP_STEPS = int(_warmup_raw) if _warmup_raw is not None else DEFAULT_WARMUP_STEPS

parser = argparse.ArgumentParser(
    description="Collect AMP motion data during play (policy rollout).",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("--task", type=str, default="ultra_game_yaw_run", help="Task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs (default: 1).")
parser.add_argument("--seed", type=int, default=None, help="Env seed.")
parser.add_argument("--duration", type=float, default=COLLECT_DURATION_S, help="Collection time in seconds after warmup.")
parser.add_argument("--warmup", type=int, default=COLLECT_WARMUP_STEPS, help="Warmup steps before recording.")
parser.add_argument(
    "-o",
    "--output",
    type=str,
    default=None,
    help="Output .txt path (single file) or directory (multi-env). "
    "Default: <task>/datasets/motion_amp_expert/collected_<timestamp>/",
)
parser.add_argument(
    "--single-file",
    action="store_true",
    help="Save env-0 trajectory only (use with -o ending in .txt).",
)
parser.add_argument(
    "--motion-weight",
    type=float,
    default=0.5,
    help="MotionWeight field in saved JSON (default: 0.5).",
)
parser.add_argument("--vx", type=float, default=DEFAULT_CMD_VX, help="Command vx in yaw frame (m/s).")
parser.add_argument("--vy", type=float, default=DEFAULT_CMD_VY, help="Command vy in yaw frame (m/s).")
parser.add_argument("--wz", type=float, default=DEFAULT_CMD_WZ, help="Command yaw rate (rad/s).")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _hydra_args = parser.parse_known_args()
if _duration_raw is not None:
    args_cli.duration = COLLECT_DURATION_S
if _warmup_raw is not None:
    args_cli.warmup = COLLECT_WARMUP_STEPS
print(f"[INFO] collect duration={args_cli.duration:.1f}s  warmup={args_cli.warmup} steps")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs.ui import ViewportCameraController  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from legged_lab.envs import *  # noqa: F401, F403, E402
from legged_lab.utils.amp_motion_io import save_amp_motion_txt  # noqa: E402
from legged_lab.utils.cli_args import update_rsl_rl_cfg  # noqa: E402
from legged_lab.utils.play_viewport_camera import (  # noqa: E402
    apply_viewport_camera_patch,
    setup_camera_follow_toggle,
)
from rsl_rl.runners import AmpOnPolicyRunner, MultiAmpOnPolicyRunner, OnPolicyRunner  # noqa: E402

apply_viewport_camera_patch()


def pin_play_command(env, vx: float, vy: float, wz: float) -> None:
    """Pin velocity command for all envs (same as play.py after env.step)."""
    env.command_generator.vel_command_b[:, 0] = vx
    env.command_generator.vel_command_b[:, 1] = vy
    env.command_generator.vel_command_b[:, 2] = wz
    if hasattr(env.command_generator, "cmd_target_b"):
        env.command_generator.cmd_target_b[:, 0] = vx
        env.command_generator.cmd_target_b[:, 1] = vy
        env.command_generator.cmd_target_b[:, 2] = wz


def motion_filename_stem(env_idx: int, vx: float, vy: float, wz: float) -> str:
    stem = f"env{env_idx:03d}_vx{vx:+.2f}_vy{vy:+.2f}_w{wz:+.2f}"
    return stem.replace(".", "p").replace("+", "p").replace("-", "n")


def default_output_dir(task_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if "ultra_game_yaw" in task_name:
        base = Path(LEGGED_LAB_ROOT_DIR) / "legged_lab/envs/ultra_game_yaw/datasets/motion_amp_expert"
    else:
        base = Path(LEGGED_LAB_ROOT_DIR) / "datasets" / "motion_amp_expert"
    return base / f"collected_{stamp}"


def configure_play_env(env_cfg, args: argparse.Namespace) -> None:
    """Play-like overrides: plane, no DR noise, fixed command ranges."""
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.domain_rand.events.randomize_rigid_body_inertia = None
    env_cfg.domain_rand.events.add_base_mass = None
    env_cfg.domain_rand.events.add_others_mass = None
    env_cfg.domain_rand.events.randomize_rigid_body_com = None
    env_cfg.domain_rand.events.randomize_joint_armature = None
    env_cfg.domain_rand.events.randomize_joint_friction = None
    env_cfg.domain_rand.events.randomize_actuator_friction = None
    env_cfg.domain_rand.events.randomize_motor_strength = None
    env_cfg.domain_rand.events.randomize_actuator_gains = None
    env_cfg.domain_rand.events.physics_material = None
    env_cfg.scene.max_episode_length_s = max(1000.0, args.duration + 30.0)
    env_cfg.scene.num_envs = 1 if args.num_envs is None else args.num_envs
    env_cfg.scene.env_spacing = 2.5
    env_cfg.commands.debug_vis = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.cmd_bucket_probs = None
    env_cfg.commands.use_curriculum = False
    env_cfg.commands.run_high_speed_sample_frac = None
    env_cfg.commands.resampling_time_range = (0.1, 0.1)
    env_cfg.commands.ranges.lin_vel_x = (args.vx, args.vx)
    env_cfg.commands.ranges.lin_vel_y = (args.vy, args.vy)
    env_cfg.commands.ranges.ang_vel_z = (args.wz, args.wz)
    env_cfg.commands.ranges.heading = (0.0, 0.0)
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)
    env_cfg.scene.height_scanner.enable_height_scan = False
    env_cfg.scene.height_scanner.enable_feet_height_scan = False
    env_cfg.scene.terrain_generator = None
    env_cfg.scene.terrain_type = "plane"


def rollout_step(env, policy, obs: torch.Tensor, vx: float, vy: float, wz: float) -> torch.Tensor:
    with torch.inference_mode():
        actions = policy(obs)
        obs, _, _, _ = env.step(actions)
    pin_play_command(env, vx, vy, wz)
    return obs


def collect() -> None:
    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)
    configure_play_env(env_cfg, args_cli)

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    vx, vy, wz = args_cli.vx, args_cli.vy, args_cli.wz
    num_envs = env_cfg.scene.num_envs
    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)

    camera = None
    camera_follow_sub = None
    if not args_cli.headless:
        camera = ViewportCameraController(env, env_cfg.viewer)  # noqa: F841
        camera_follow_sub = setup_camera_follow_toggle(default_on=False)  # noqa: F841

    if not hasattr(env, "get_amp_obs_for_expert_trans"):
        raise RuntimeError(
            f"Task {env_class_name} env has no get_amp_obs_for_expert_trans(); "
            "use a Multi-AMP motion task (e.g. ultra_game_yaw_run)."
        )

    log_root_path = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    print(f"[INFO] Checkpoint: {resume_path}")

    runner_class = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.device)

    frame_dt = float(env.step_dt)
    num_collect = int(args_cli.duration / frame_dt)

    print(f"\n{'=' * 72}")
    print("Collect play motion (AMP expert format)")
    print(f"{'=' * 72}")
    print(f"Task:        {env_class_name}")
    print(f"Envs:        {num_envs}")
    print(f"Command:     vx={vx:+.2f}  vy={vy:+.2f}  wz={wz:+.2f}")
    print(f"Frame dt:    {frame_dt:.4f}s  ({1.0 / frame_dt:.1f} Hz)")
    print(f"Warmup:      {args_cli.warmup} steps")
    print(f"Collect:     {num_collect} steps ({args_cli.duration:.1f}s)")
    print(f"AMP dim:     {env.get_amp_obs_for_expert_trans().shape[-1]}")
    print(f"{'=' * 72}\n")

    obs, _ = env.get_observations()
    pin_play_command(env, vx, vy, wz)
    terminate_counts = 0

    print(f"[Warmup] {args_cli.warmup} steps...")
    for _ in range(args_cli.warmup):
        obs = rollout_step(env, policy, obs, vx, vy, wz)
        if env.reset_buf.any():
            terminate_counts += int(env.reset_buf.sum().item())

    env_frames = [[] for _ in range(num_envs)]
    print(f"[Collect] {num_collect} steps...")
    for step in range(num_collect):
        obs = rollout_step(env, policy, obs, vx, vy, wz)
        amp_obs = env.get_amp_obs_for_expert_trans()
        for idx in range(num_envs):
            env_frames[idx].append(amp_obs[idx].detach().cpu().numpy())

        if env.reset_buf.any():
            terminate_counts += int(env.reset_buf.sum().item())

        if (step + 1) % 500 == 0:
            print(f"  step {step + 1}/{num_collect}  terminates={terminate_counts}")

    print(f"[Collect] Done. terminations={terminate_counts}")

    if args_cli.single_file or (args_cli.output and args_cli.output.endswith(".txt")):
        out_path = Path(args_cli.output) if args_cli.output else default_output_dir(env_class_name) / "motion.txt"
        frames = np.stack(env_frames[0], axis=0)
        save_amp_motion_txt(out_path, frames, frame_dt, motion_weight=args_cli.motion_weight)
        print(f"[Save] {out_path}  frames={len(frames)}  dur={len(frames) * frame_dt:.2f}s")
        return

    if args_cli.output:
        out_dir = Path(args_cli.output)
    else:
        out_dir = default_output_dir(env_class_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Save] {out_dir}")
    for idx in range(num_envs):
        if not env_frames[idx]:
            print(f"  [{idx:3d}] skip (no frames)")
            continue
        frames = np.stack(env_frames[idx], axis=0)
        stem = motion_filename_stem(idx, vx, vy, wz)
        out_path = out_dir / f"{stem}.txt"
        save_amp_motion_txt(out_path, frames, frame_dt, motion_weight=args_cli.motion_weight)
        print(
            f"  [{idx:3d}] vx={vx:+.2f} vy={vy:+.2f} w={wz:+.2f} | "
            f"frames={len(frames)} dur={len(frames) * frame_dt:.1f}s -> {out_path.name}"
        )

    print(f"\n{'=' * 72}")
    print(f"Saved {num_envs} motion file(s) under {out_dir}")
    print("Add paths to motion_run_cfg.py amp_motion_files_dict as needed.")
    print(f"{'=' * 72}")


def main() -> None:
    try:
        collect()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
