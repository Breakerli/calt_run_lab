# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""RGB frame capture for play.py and sim2sim_ultraYaw.py (--video)."""

from __future__ import annotations

import os

import numpy as np


def setup_rgb_capture(env) -> None:
    """Attach a Replicator RGB annotator to the viewer camera (once per env)."""
    if hasattr(env, "_play_rgb_annotator"):
        return

    import omni.replicator.core as rep

    from isaaclab.sim import SimulationContext

    viewer_cfg = env.cfg.viewer
    if env.sim.render_mode.value < SimulationContext.RenderMode.PARTIAL_RENDERING.value:
        raise RuntimeError(
            "Cannot record video: simulation render mode is too low. "
            "Use --enable_cameras (set automatically with --video) and avoid a non-rendering kit."
        )

    env._play_render_product = rep.create.render_product(viewer_cfg.cam_prim_path, viewer_cfg.resolution)
    env._play_rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    env._play_rgb_annotator.attach([env._play_render_product])


def capture_rgb_frame(env) -> np.ndarray:
    """Return (H, W, 3) uint8 RGB. Calls sim.render() when needed (e.g. headless play)."""
    setup_rgb_capture(env)

    if not env.sim.has_rtx_sensors():
        env.sim.render()

    rgb_data = env._play_rgb_annotator.get_data()
    rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
    if rgb_data.size == 0:
        w, h = env.cfg.viewer.resolution
        return np.zeros((h, w, 3), dtype=np.uint8)
    return rgb_data[:, :, :3]


def save_frames_to_mp4(frames: list[np.ndarray], path: str, fps: int) -> None:
    """Write frames to an mp4 (same backend as gymnasium RecordVideo)."""
    if len(frames) == 0:
        print("[WARN] No frames captured; video not saved.")
        return

    try:
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    except ImportError as e:
        raise ImportError('MoviePy is required for --video. Install: pip install "gymnasium[other]"') from e

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(path, logger=None)
    del clip
    print(f"[INFO] Saved video ({len(frames)} frames @ {fps} fps): {os.path.abspath(path)}")


class PlayVideoRecorder:
    """Collect rgb_array frames during play and export one mp4."""

    def __init__(self, env, log_dir: str, video_length: int, name_prefix: str = "rl-video-step-0"):
        self.env = env
        self.video_length = video_length
        self.frames: list[np.ndarray] = []
        self.fps = max(1, int(round(1.0 / float(env.step_dt))))
        self.video_folder = os.path.join(log_dir, "videos", "play")
        self.output_path = os.path.join(self.video_folder, f"{name_prefix}.mp4")
        self._saved = False
        os.makedirs(self.video_folder, exist_ok=True)
        print(f"[INFO] Recording video: up to {video_length} steps -> {self.output_path} ({self.fps} fps)")

    def capture_step(self) -> bool:
        """Capture one frame after env.step(). Returns True when recording is complete."""
        self.frames.append(capture_rgb_frame(self.env))
        return len(self.frames) >= self.video_length

    def save(self) -> None:
        if self._saved:
            return
        self._saved = True
        save_frames_to_mp4(self.frames, self.output_path, fps=self.fps)


def capture_mujoco_viewer_rgb(viewer) -> np.ndarray:
    """Read the current GLFW framebuffer after mujoco_viewer.render()."""
    import glfw
    import mujoco

    width, height = glfw.get_framebuffer_size(viewer.window)
    viewer.viewport.width, viewer.viewport.height = width, height
    img = np.zeros((height, width, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(img, None, viewer.viewport, viewer.ctx)
    return np.flipud(img)


class Sim2SimVideoRecorder:
    """Collect MuJoCo viewer frames; one frame per policy step (after viewer.render())."""

    def __init__(
        self,
        viewer,
        video_length: int,
        output_path: str,
        policy_dt: float,
        name_prefix: str = "rl-video-step-0",
    ):
        self.viewer = viewer
        self.video_length = video_length
        self.frames: list[np.ndarray] = []
        self.fps = max(1, int(round(1.0 / float(policy_dt))))
        self.video_folder = os.path.dirname(os.path.abspath(output_path))
        base = os.path.basename(output_path)
        if not base.endswith(".mp4"):
            base = f"{name_prefix}.mp4"
        self.output_path = os.path.join(self.video_folder, base)
        self._saved = False
        os.makedirs(self.video_folder, exist_ok=True)
        print(f"[INFO] Recording video: up to {video_length} policy steps -> {self.output_path} ({self.fps} fps)")

    def capture_step(self) -> bool:
        self.frames.append(capture_mujoco_viewer_rgb(self.viewer))
        return len(self.frames) >= self.video_length

    def save(self) -> None:
        if self._saved:
            return
        self._saved = True
        save_frames_to_mp4(self.frames, self.output_path, fps=self.fps)
