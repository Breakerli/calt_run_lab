# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Viewport camera follow + F-key toggle (shared by play.py and collect_play_motion_data.py)."""

from __future__ import annotations

import time

import numpy as np
from isaaclab.envs.ui import ViewportCameraController

_camera_follow_enabled = [False]
_camera_orbit_start_t: list[float | None] = [None]
CAMERA_ORBIT_PERIOD_S = 50.0


def _reset_camera_orbit_clock() -> None:
    _camera_orbit_start_t[0] = time.perf_counter()


def _camera_orbit_yaw() -> float:
    if _camera_orbit_start_t[0] is None:
        _reset_camera_orbit_clock()
    elapsed = time.perf_counter() - _camera_orbit_start_t[0]
    return (2.0 * np.pi / CAMERA_ORBIT_PERIOD_S) * elapsed % (2.0 * np.pi)


def _set_camera_view_with_orbit(self) -> None:
    base_eye = np.asarray(self.default_cam_eye, dtype=float)
    base_lookat = np.asarray(self.default_cam_lookat, dtype=float)
    offset = base_eye - base_lookat

    yaw = _camera_orbit_yaw()
    c, s = np.cos(yaw), np.sin(yaw)
    rot_offset = np.array(
        [c * offset[0] - s * offset[1], s * offset[0] + c * offset[1], offset[2]],
        dtype=float,
    )

    origin = self.viewer_origin.detach().cpu().numpy()
    cam_target = origin + base_lookat
    cam_eye = cam_target + rot_offset
    self._env.sim.set_camera_view(eye=cam_eye, target=cam_target)


def _safe_update_tracking_callback(self, event):
    if not _camera_follow_enabled[0]:
        return
    cfg = self.cfg
    if cfg.origin_type not in ("asset_root", "asset_body"):
        return
    if cfg.asset_name is None:
        return
    try:
        asset = self._env.scene[cfg.asset_name]
    except (KeyError, AttributeError):
        return
    if getattr(asset, "root_physx_view", None) is None:
        return

    if cfg.origin_type == "asset_root":
        self.viewer_origin = self._env.scene[cfg.asset_name].data.root_pos_w[cfg.env_index]
    elif cfg.origin_type == "asset_body" and cfg.body_name is not None:
        body_id, _ = asset.find_bodies(cfg.body_name)
        self.viewer_origin = asset.data.body_pos_w[cfg.env_index, body_id[0]].view(3)

    _set_camera_view_with_orbit(self)


def apply_viewport_camera_patch() -> None:
    """Safe follow callback + F-key toggle gate (call once after AppLauncher)."""
    ViewportCameraController._update_tracking_callback = _safe_update_tracking_callback


def setup_camera_follow_toggle(default_on: bool = False):
    """Subscribe F key to toggle camera follow. Caller must keep the returned handle."""
    import carb.input
    import omni.appwindow

    _camera_follow_enabled[0] = default_on
    if default_on:
        _reset_camera_orbit_clock()
    print(
        f"[Camera] follow = {'ON' if default_on else 'OFF'}  (press F to toggle; "
        f"ON: orbit {CAMERA_ORBIT_PERIOD_S:.0f}s/rev)"
    )

    appwindow = omni.appwindow.get_default_app_window()
    keyboard = appwindow.get_keyboard()
    input_iface = carb.input.acquire_input_interface()

    def _on_key(event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS and event.input.name == "F":
            _camera_follow_enabled[0] = not _camera_follow_enabled[0]
            if _camera_follow_enabled[0]:
                _reset_camera_orbit_clock()
            print(
                f"[Camera] follow = {'ON' if _camera_follow_enabled[0] else 'OFF'}"
                + (f"  (orbit {CAMERA_ORBIT_PERIOD_S:.0f}s/rev)" if _camera_follow_enabled[0] else "")
            )
        return True

    return input_iface.subscribe_to_keyboard_events(keyboard, _on_key)
