# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Read/write AMP expert motion JSON (.txt) files."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_amp_motion_txt(
    save_path: str | Path,
    frames_np: np.ndarray,
    frame_duration: float,
    motion_weight: float = 0.5,
    loop_mode: str = "Wrap",
) -> None:
    """Write AMP motion JSON txt (same format as play_amp_animation / 2026gym collect)."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    frames_np = np.asarray(frames_np, dtype=np.float64)
    num_frames = frames_np.shape[0]
    print(f"Saving {num_frames} frames (dim={frames_np.shape[1]}) to {save_path} ...")
    with save_path.open("w", encoding="utf-8") as f:
        f.write("{\n")
        f.write(f'"LoopMode": "{loop_mode}",\n')
        f.write(f'"FrameDuration": {frame_duration:.6f},\n')
        f.write('"EnableCycleOffsetPosition": true,\n')
        f.write('"EnableCycleOffsetRotation": true,\n')
        f.write(f'"MotionWeight": {motion_weight:.6f},\n\n')
        f.write('"Frames":\n[\n')
        for i in range(num_frames):
            row = ", ".join(f"{v:.6f}" for v in frames_np[i])
            if i < num_frames - 1:
                f.write(f"  [{row}],\n")
            else:
                f.write(f"  [{row}]\n")
            if i % 1000 == 0 and i > 0:
                print(f"  write progress: {i}/{num_frames}")
        f.write("]\n}")
