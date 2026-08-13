# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""Convert ultra_GameYaw (13-DoF) AMP expert motion files to yrp03-v2 (15-DoF).

yrp03-v2 adds two ankle_roll joints (one per leg). Those joints do not exist in
the original expert retargeting, so per the design they are zero-padded, inserted
at the correct slot inside each leg block so the AMP feature order stays aligned
with UltraGameYawMotionRunEnv.get_amp_obs_for_expert_trans / AMPLoaderYrp.

Original 38-col AMP frame layout (per env + AMPLoader):
    pos[0:13]  = [shoulder_r, shoulder_l,
                  right_leg(hip_yaw,hip_roll,hip_pitch,knee,ankle_pitch),
                  left_leg(...5...), waist_yaw]
    vel[13:26] = same order, velocities
    end[26:38] = [left_hand(3), right_hand(3), left_foot(3), right_foot(3)]

Target 42-col frame: ankle_roll (=0) inserted after each ankle_pitch in the pos
and vel blocks -> right_leg becomes 6 dof, left_leg becomes 6 dof.

Pure JSON/number manipulation; no Isaac / torch dependency.

Usage:
    python legged_lab/scripts/utils/convert_amp_expert_to_yrp03_v2.py
"""

from __future__ import annotations

import json
import os

SRC_DIR = "legged_lab/envs/ultra_game_yaw/datasets/motion_amp_expert"
DST_DIR = "legged_lab/envs/yrp03_v2/datasets/motion_amp_expert"

# Files referenced by the yrp03_v2 run cfg (stand / walk / run x2).
FILES = [
    "stand_ultra_yaw2.txt",
    "walk_ultra_yaw2.txt",
    "run_17200_9mps.txt",
    "collected_20260602_160257/env000_vxp15p00_vyp0p00_wp0p00.txt",
]

SRC_COLS = 38
DST_COLS = 42


def convert_frame(frame: list[float]) -> list[float]:
    """Insert ankle_roll=0 after each ankle_pitch in the pos and vel blocks."""
    if len(frame) != SRC_COLS:
        raise ValueError(f"expected {SRC_COLS} cols, got {len(frame)}")
    new = (
        frame[0:7]        # pos: shoulder_r, shoulder_l, right_leg[hy,hr,hp,knee,ankle_pitch]
        + [0.0]           # -> ankle_roll_r (pos)
        + frame[7:12]     # pos: left_leg[hy,hr,hp,knee,ankle_pitch]
        + [0.0]           # -> ankle_roll_l (pos)
        + frame[12:20]    # waist_yaw(pos) + vel: shoulder_r,shoulder_l,right_leg[...ankle_pitch]
        + [0.0]           # -> ankle_roll_r (vel)
        + frame[20:25]    # vel: left_leg[hy,hr,hp,knee,ankle_pitch]
        + [0.0]           # -> ankle_roll_l (vel)
        + frame[25:38]    # waist_yaw(vel) + end-effector(12)
    )
    if len(new) != DST_COLS:
        raise ValueError(f"produced {len(new)} cols, expected {DST_COLS}")
    return new


def convert_file(src_path: str, dst_path: str) -> None:
    with open(src_path) as f:
        motion = json.load(f)
    frames = motion["Frames"]
    motion["Frames"] = [convert_frame(list(fr)) for fr in frames]
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w") as f:
        json.dump(motion, f, indent=2)
    print(f"[convert] {src_path} ({len(frames)} frames) -> {dst_path} [{SRC_COLS}->{DST_COLS} cols]")


def main() -> None:
    for rel in FILES:
        convert_file(os.path.join(SRC_DIR, rel), os.path.join(DST_DIR, rel))
    print("[convert] done.")


if __name__ == "__main__":
    main()
