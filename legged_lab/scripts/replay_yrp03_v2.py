"""Replay a YRP03-v2 AMP motion txt file inside Isaac Lab.

Adapted from ``replay_ultra_game_yaw.py`` for the YRP03-v2 model. The motion
files are the same 38-dim schema (root pos/rot + 13-joint dof pos/vel, see
``Yrp03V2MotionRunEnv.visualize_motion``), so the replay logic is unchanged;
only the default task and motion file are switched to YRP03-v2.

Example:
    python legged_lab/scripts/replay_yrp03_v2.py \
        --input_file legged_lab/envs/yrp03_v2/datasets/motion_visualization/run_14mps_play_newultra.txt
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TASK = "yrp03_v2_run"
DEFAULT_MOTION_FILE = "legged_lab/envs/yrp03_v2/datasets/motion_visualization/run_14mps_play_newultra.txt"
# 38-dim schema: root_pos(3) + euler(3) + dof_pos(13) + lin_vel(3) + ang_vel(3) + dof_vel(13).
# YRP03-v2 的两个 ankle_roll 关节不在此 schema 内(保持默认 0), 与 Ultra 13 关节布局一致。
EXPECTED_FRAME_DIM = 38


@dataclass(frozen=True)
class MotionMetadata:
    frame_duration: float
    num_frames: int
    frame_dim: int


def resolve_motion_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.suffix != ".txt":
        path = path.with_suffix(".txt")
    return path.expanduser().resolve()


def load_motion_metadata(path_str: str | Path) -> MotionMetadata:
    motion_path = resolve_motion_path(str(path_str))
    if not motion_path.is_file():
        raise FileNotFoundError(f"Motion file not found: {motion_path}")

    with motion_path.open("r", encoding="utf-8") as stream:
        motion_json = json.load(stream)

    frames = motion_json.get("Frames")
    if not isinstance(frames, list) or len(frames) == 0:
        raise ValueError(f"Motion file has no Frames entries: {motion_path}")

    first_frame = frames[0]
    if not isinstance(first_frame, list) or len(first_frame) == 0:
        raise ValueError(f"Motion file has an invalid first frame: {motion_path}")

    frame_duration = float(motion_json["FrameDuration"])
    if frame_duration <= 0.0:
        raise ValueError(f"FrameDuration must be positive: {frame_duration}")

    return MotionMetadata(
        frame_duration=frame_duration,
        num_frames=len(frames),
        frame_dim=len(first_frame),
    )


def _build_parser(app_launcher_cls) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a YRP03-v2 AMP motion txt file.")
    parser.add_argument("--input_file", type=str, default=DEFAULT_MOTION_FILE, help="Path to the txt motion file.")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Registered YRP03-v2 task name.")
    parser.add_argument("--fps", type=float, default=None, help="Override playback FPS stored in the motion file.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument("--loop", action="store_true", help="Loop the motion instead of holding the last frame.")
    parser.add_argument(
        "--camera_follow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Follow the robot with the camera.",
    )
    app_launcher_cls.add_app_launcher_args(parser)
    return parser


def _configure_env(env_cfg, args_cli, motion_path: Path):
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 2.5
    env_cfg.scene.terrain_generator = None
    env_cfg.scene.terrain_type = "plane"
    env_cfg.commands.debug_vis = False
    env_cfg.amp_motion_files_display = [str(motion_path)]
    env_cfg.device = args_cli.device


def _run_replay(simulation_app, env, metadata: MotionMetadata, args_cli):
    import numpy as np

    frame_duration = 1.0 / args_cli.fps if args_cli.fps is not None else metadata.frame_duration
    frame_duration /= max(args_cli.speed, 1.0e-6)

    frame_idx = 0
    last_frame_wall_time = time.perf_counter()

    while simulation_app.is_running():
        if frame_idx >= metadata.num_frames:
            if args_cli.loop:
                frame_idx = 0
            else:
                frame_idx = metadata.num_frames - 1

        env.visualize_motion(frame_idx=frame_idx)

        if args_cli.camera_follow:
            lookat = env.robot.data.root_state_w[0, :3].detach().cpu().numpy()
            env.sim.set_camera_view(lookat + np.array([2.0, 2.0, 0.8]), lookat)

        frame_idx += 1

        elapsed = time.perf_counter() - last_frame_wall_time
        sleep_time = frame_duration - elapsed
        if sleep_time > 0.0:
            time.sleep(sleep_time)
        last_frame_wall_time = time.perf_counter()


def main():
    from isaaclab.app import AppLauncher

    parser = _build_parser(AppLauncher)
    args_cli = parser.parse_args()

    motion_path = resolve_motion_path(args_cli.input_file)
    metadata = load_motion_metadata(motion_path)
    if metadata.frame_dim != EXPECTED_FRAME_DIM:
        raise ValueError(
            f"Unsupported YRP03-v2 frame dimension: {metadata.frame_dim}. "
            f"Expected {EXPECTED_FRAME_DIM} for this replay script."
        )

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import legged_lab.envs  # noqa: F401
    from legged_lab.utils.task_registry import task_registry

    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)
    env_cfg.scene.seed = agent_cfg.seed
    _configure_env(env_cfg, args_cli, motion_path)

    env_class = task_registry.get_task_class(args_cli.task)
    env = env_class(env_cfg, args_cli.headless)

    fps = 1.0 / metadata.frame_duration
    print(f"[INFO] Loading motion: {motion_path}")
    print(f"[INFO] Motion frames: {metadata.num_frames}, frame_dim: {metadata.frame_dim}, fps: {fps:.2f}")

    try:
        _run_replay(simulation_app, env, metadata, args_cli)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
