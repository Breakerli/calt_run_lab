"""Replay a YRP03-v2 AMP expert motion txt file inside Isaac Lab.

The ``motion_amp_expert`` files are stored in the 42-dim AMP feature layout that
``Yrp03V2MotionRunEnv.get_amp_obs_for_expert_trans`` produces:

    pos[0:15]  = right_arm(1), left_arm(1), right_leg(6), left_leg(6), waist(1)
    vel[15:30] = same order (velocities)
    end[30:42] = left_hand(3), right_hand(3), left_foot(3), right_foot(3)

Unlike the ``motion_visualization`` files (root_pos + euler + dof_pos + lin/ang
vel), this schema has no root trajectory. The replay therefore drives the joints
directly and reconstructs the root height from the feet (so the lowest foot
touches the ground), holding identity orientation and zero velocity.

Example:
    python legged_lab/scripts/replay_yrp03_v2.py \
        --input_file legged_lab/envs/yrp03_v2/datasets/motion_amp_expert/run_17200_9mps.txt
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TASK = "yrp03_v2_run"
DEFAULT_MOTION_FILE = "legged_lab/envs/yrp03_v2/datasets/motion_amp_expert/run_17200_9mps.txt"
# 42-dim AMP expert feature layout (see module docstring). ankle_roll is zero-padded
# by convert_amp_expert_to_yrp03_v2.py but is still present as the 6th leg dof.
EXPECTED_FRAME_DIM = 42

# The ankle-roll link origin sits above the foot sole; lift the reconstructed root
# so the sole (not the link origin) rests on the ground. Matches
# ``CriticPrivilegeCfg.foot_height_offset`` / ``feet_descent_proximity_neg``.
FOOT_SOLE_OFFSET = 0.074


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


def load_motion(path_str: str | Path):
    """Load a motion txt file, returning ``(metadata, frames_np)``.

    ``frames_np`` is a ``(num_frames, frame_dim)`` float32 numpy array.
    """
    import numpy as np

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

    metadata = MotionMetadata(
        frame_duration=frame_duration,
        num_frames=len(frames),
        frame_dim=len(first_frame),
    )
    return metadata, np.asarray(frames, dtype=np.float32)


def _build_parser(app_launcher_cls) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a YRP03-v2 AMP expert motion txt file.")
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


def _apply_amp_frame(env, frame):
    """Write one 42-dim AMP expert frame into the sim.

    Args:
        env: ``Yrp03V2MotionRunEnv`` instance.
        frame: ``(42,)`` float32 tensor on ``env.device``.
    """
    import torch

    num_envs = env.num_envs
    device = env.device

    dof_pos = torch.zeros((num_envs, env.robot.num_joints), device=device)
    dof_vel = torch.zeros_like(dof_pos)

    # pos block (15): right_arm, left_arm, right_leg(6), left_leg(6), waist.
    dof_pos[:, env.right_arm_ids] = frame[0]
    dof_pos[:, env.left_arm_ids] = frame[1]
    dof_pos[:, env.right_leg_ids] = frame[2:8]
    dof_pos[:, env.left_leg_ids] = frame[8:14]
    dof_pos[:, env.waist_yaw_ids] = frame[14]

    # vel block (15): same order.
    dof_vel[:, env.right_arm_ids] = frame[15]
    dof_vel[:, env.left_arm_ids] = frame[16]
    dof_vel[:, env.right_leg_ids] = frame[17:23]
    dof_vel[:, env.left_leg_ids] = frame[23:29]
    dof_vel[:, env.waist_yaw_ids] = frame[29]

    # root state: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]. The AMP
    # expert schema has no root trajectory, so hold identity orientation + zero
    # velocity and reconstruct z from the feet so the lowest foot touches ground.
    foot_z = frame[36:42].reshape(2, 3)[:, 2]  # left_foot z, right_foot z (root frame)
    root_z = float(-foot_z.min().item()) + FOOT_SOLE_OFFSET

    root_state = torch.zeros((num_envs, 13), device=device)
    root_state[:, 2] = root_z
    root_state[:, 3] = 1.0  # identity quaternion (w=1)

    env_ids = torch.arange(num_envs, device=device)
    env.robot.write_joint_position_to_sim(dof_pos)
    env.robot.write_joint_velocity_to_sim(dof_vel)
    env.robot.write_root_state_to_sim(root_state, env_ids)
    env.sim.render()
    env.scene.update(dt=env.step_dt)


def _run_replay(simulation_app, env, metadata: MotionMetadata, frames, args_cli):
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

        _apply_amp_frame(env, frames[frame_idx])

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
    import torch
    from isaaclab.app import AppLauncher

    parser = _build_parser(AppLauncher)
    args_cli = parser.parse_args()

    motion_path = resolve_motion_path(args_cli.input_file)
    metadata, frames_np = load_motion(motion_path)
    if metadata.frame_dim != EXPECTED_FRAME_DIM:
        raise ValueError(
            f"Unsupported YRP03-v2 AMP frame dimension: {metadata.frame_dim}. "
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

    frames = torch.tensor(frames_np, device=env.device, dtype=torch.float32)

    fps = 1.0 / metadata.frame_duration
    print(f"[INFO] Loading motion: {motion_path}")
    print(f"[INFO] Motion frames: {metadata.num_frames}, frame_dim: {metadata.frame_dim}, fps: {fps:.2f}")

    try:
        _run_replay(simulation_app, env, metadata, frames, args_cli)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
