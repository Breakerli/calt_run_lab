#!/usr/bin/env python3
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Convert a Multi-AMP expert motion txt into a ``motion_visualization`` txt.

Background
----------
The two dataset folders hold the SAME JSON container but different per-frame
schemas (both are 38-wide for the current Ultra 13-dof layout, so they cannot
be mixed — this is why the folders are kept separate):

* ``motion_amp_expert`` (training, read by ``AMPLoader``): the discriminator
  features, all expressed in the ROOT frame. Two layouts are supported:
  - ``ultra13`` (Ultra 13-DoF, 38-col expert): column layout matches
    ``UltraGameYawMotionRunEnv.get_amp_obs_for_expert_trans()``:
    [0:13]  joint pos : right_arm, left_arm, right_leg(yaw,roll,pitch,knee,ankle),
                        left_leg(....), waist_yaw
    [13:26] joint vel : same order
    [26:38] end-effector pos (root frame) : l_hand, r_hand, l_foot, r_foot
  - ``humanoid15`` (yrp03_v2 / unitree_g1 15 feature joints, 42-col expert):
    [0:15]  joint pos : shoulder_r, shoulder_l, right_leg(yaw,roll,pitch,knee,
                        ankle_pitch,ankle_roll), left_leg(...6...), waist_yaw
    [15:30] joint vel : same order
    [30:42] end-effector pos (root frame) : l_hand, r_hand, l_foot, r_foot
  Global root pose/velocity are NOT stored (they are unnecessary for the
  discriminator). See ``rsl_rl/rsl_rl/utils/motion_loader.py``.

* ``motion_visualization`` (display, read by ``AMPLoaderDisplay`` and driven by
  ``UltraGameYawMotionRunEnv.visualize_motion()``): full-body state so the whole
  robot (root + joints) can be driven. Layout consumed by ``visualize_motion``:
    [0:3]   root_pos (x,y,z)
    [3:6]   root euler XYZ
    [6:16]  joint pos : left_leg(5), right_leg(5), waist_yaw, arm_l, arm_r
    [19:22] root lin_vel (in this root frame)
    [22:25] root ang_vel
    [25:38] joint vel : left_leg(5), right_leg(5), waist_yaw, arm_l, arm_r

What this script does / cannot do
---------------------------------
It remaps the 13 joint pos + 13 joint vel from an expert file into the visual
schema. The ee columns (26:38) are redundant for display (pure FK output) and
are dropped.

The expert file does NOT contain the global root pose/velocity, so those 12
columns cannot be recovered from it. They are filled with constants
(``--root_xyz --root_euler``, zero root velocity) which reproduces the style of
the existing ``run_*_newultra`` clips: the robot replays the recorded joint/gait
motion in place at the origin. If you need the REAL global trajectory (forward
displacement, heading, pitch profile), re-record it from the original source —
i.e. play the policy in Isaac (or keep the full state of the teleop recording)
and dump per-frame root_state + joint pos/vel, as was done for the original
``run_*_newultra`` files. ``--vx/--vy`` can inject a synthetic forward motion.

Example:
    python legged_lab/scripts/utils/amp_expert_to_visual.py \\
        --files legged_lab/envs/ultra_game_yaw/datasets/motion_amp_expert/stand_ultra_yaw2.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from legged_lab.utils.amp_motion_io import save_amp_motion_txt

# ── Column blocks (see module docstring) ────────────────────────────────────
# Expert joint order within the pos block / vel block (same order in both).
# Offsets are relative to the START of the joint block (0 for pos, span for vel).
# humanoid15 腿只取前 5 个 (yaw,roll,pitch,knee,ankle_pitch), ankle_roll 无 vis 列
# (visualize_motion 里 ankle_roll 保持 0), expert 中该列本身也是零填充。
LAYOUTS = {
    "ultra13": {
        "expert_dim": 26,  # pos13 + vel13; ee(12) dropped (expert file is 38-col)
        "joint_span": 13,
        "root_xyz_default": (0.0, 0.0, 1.15),
        "blocks": {"arm_r": (0, 1), "arm_l": (1, 1), "leg_r": (2, 5),
                   "leg_l": (7, 5), "waist": (12, 1)},
    },
    "humanoid15": {
        "expert_dim": 30,  # pos15 + vel15; ee(12) dropped (expert file is 42-col)
        "joint_span": 15,
        "root_xyz_default": (0.0, 0.0, 0.793),  # G1 站立 pelvis 高度
        "blocks": {"arm_r": (0, 1), "arm_l": (1, 1), "leg_r": (2, 5),
                   "leg_l": (8, 5), "waist": (14, 1)},
    },
}

# Visual joint layout, output col of each block's start within its 13-joint span
# (visual joint-pos cols 6:19, joint-vel cols 25:38).
VIS_POS_START = 6  # after root_pos(3)+euler(3)
VIS_VEL_START = 25  # after lin_vel(3)+ang_vel(3)
VIS_BLOCK_ORDER = ["leg_l", "leg_r", "waist", "arm_l", "arm_r"]


def load_expert_frames(path: str | Path, layout: dict) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        motion = json.load(f)
    frames = np.asarray(motion["Frames"], dtype=np.float64)
    dim = layout["expert_dim"]
    if frames.ndim != 2 or frames.shape[1] < dim:
        raise ValueError(
            f"Expected >={dim}-dim AMP expert frames, got {frames.shape}. "
            f"Check that --layout matches the expert file ({dim + 12}-col container)."
        )
    return {
        "frames": frames[:, :dim],  # joint pos + vel (same span); drop ee(12)
        "frame_duration": float(motion["FrameDuration"]),
        "motion_weight": float(motion.get("MotionWeight", 0.5)),
        "loop_mode": str(motion.get("LoopMode", "Wrap")),
    }


def _place_block(out: np.ndarray, src: np.ndarray, blocks: dict, block_name: str, src_base: int, vis_base: int) -> None:
    src_off, n = blocks[block_name]
    vis_off = 0
    for name in VIS_BLOCK_ORDER:
        if name == block_name:
            out[:, vis_base + vis_off : vis_base + vis_off + n] = src[:, src_base + src_off : src_base + src_off + n]
            return
        vis_off += blocks[name][1]
    raise KeyError(block_name)


def expert_to_visual(
    frames: np.ndarray,
    layout: dict,
    root_xyz=(0.0, 0.0, 1.15),
    root_euler=(0.0, 0.0, 0.0),
    vx: float = 0.0,
    vy: float = 0.0,
    frame_duration: float = 0.01,
) -> np.ndarray:
    """Map expert [pos|vel] frames to the 38-col visual schema."""
    n = frames.shape[0]
    span = layout["joint_span"]
    blocks = layout["blocks"]
    out = np.zeros((n, 38), dtype=np.float64)

    # root pose + euler
    out[:, 0:3] = root_xyz
    out[:, 3:6] = root_euler

    # joints
    for name in ("leg_l", "leg_r", "waist", "arm_l", "arm_r"):
        _place_block(out, frames, blocks, name, 0, VIS_POS_START)
        _place_block(out, frames, blocks, name, span, VIS_VEL_START)

    # root velocities (not recoverable from expert data)
    if vx != 0.0 or vy != 0.0:
        t = np.arange(n, dtype=np.float64) * frame_duration
        out[:, 0] = vx * t
        out[:, 1] = vy * t
        out[:, 19] = vx
        out[:, 20] = vy
    return out


def default_output_path(input_path: Path) -> Path:
    """Write next to the source file's env dir, under ``motion_visualization``."""
    src_dir = input_path.parent  # .../<env>/datasets/motion_amp_expert
    if src_dir.name == "motion_amp_expert":
        out_dir = src_dir.with_name("motion_visualization")
    else:
        out_dir = src_dir
    return out_dir / f"{input_path.stem}_vis.txt"


def _parse_argv_float3(value: str, label: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"{label} must be 'a,b,c'")
    return tuple(float(p) for p in parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="AMP expert .txt motion file(s).",
    )
    parser.add_argument(
        "--layout",
        choices=sorted(LAYOUTS),
        default="ultra13",
        help="Expert joint layout: ultra13 (Ultra 38-col expert) or "
        "humanoid15 (yrp03_v2 / unitree_g1 42-col expert, 6-DoF legs, "
        "ankle_roll dropped for display).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file/dir. Default: <env>/datasets/motion_visualization/<stem>_vis.txt per input.",
    )
    parser.add_argument(
        "--root_xyz",
        type=str,
        default=None,
        help="Root position (x,y,z) to stamp on every frame. Default: layout-specific "
        "(ultra13: 0,0,1.15; humanoid15: 0,0,0.793). visualize_motion() adds +1.5 to z "
        "at playback, so tune to your model.",
    )
    parser.add_argument(
        "--root_euler",
        type=str,
        default="0.0,0.0,0.0",
        help="Root euler XYZ (rad) stamped on every frame (roll,pitch,yaw).",
    )
    parser.add_argument(
        "--vx",
        type=float,
        default=0.0,
        help="Synthetic forward speed (m/s): integrates root x and sets lin_vel_x. "
        "Useful to fake global translation for a run clip whose command speed is known.",
    )
    parser.add_argument(
        "--vy",
        type=float,
        default=0.0,
        help="Synthetic lateral speed (m/s).",
    )
    args = parser.parse_args()

    layout = LAYOUTS[args.layout]
    root_xyz = (_parse_argv_float3(args.root_xyz, "--root_xyz") if args.root_xyz is not None
                else layout["root_xyz_default"])
    root_euler = _parse_argv_float3(args.root_euler, "--root_euler")

    for file in args.files:
        in_path = Path(file)
        data = load_expert_frames(in_path, layout)
        frames = data["frames"]
        vis = expert_to_visual(
            frames,
            layout,
            root_xyz=root_xyz,
            root_euler=root_euler,
            vx=args.vx,
            vy=args.vy,
            frame_duration=data["frame_duration"],
        )

        if args.output is not None:
            out_path = args.output if args.output.suffix == ".txt" else args.output / f"{in_path.stem}_vis.txt"
        else:
            out_path = default_output_path(in_path)

        save_amp_motion_txt(
            out_path,
            vis,
            frame_duration=data["frame_duration"],
            motion_weight=data["motion_weight"],
            loop_mode=data["loop_mode"],
        )
        print(
            f"[OK] {in_path} ({frames.shape[0]} frames) -> {out_path}\n"
            f"     To preview: set env_cfg.amp_motion_files_display = [{str(out_path)!r}] and run replay."
        )


if __name__ == "__main__":
    main()
