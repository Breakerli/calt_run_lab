"""在 IsaacSim 中运行 ultra_GameYaw 策略。

核心功能：
1. 默认使用手柄输入控制机器人 body-frame 速度命令。
2. 使用 ``--use_vel_cmd`` 时切换到预设速度命令：5m/s、10m/s、15m/s 各持续 4s。
3. 使用 ``--plot_joints`` 时退出后自动绘制并保存 base 速度、T-N 散点图、关节位置/限位、速度和力矩曲线。
4. 使用 ``--record_pkl`` 时记录 env0 的关节/各 body 世界系轨迹，退出时保存为 pkl
   （字段与 play.py 的 cleaned npz 一致：joint_pos/joint_vel/body_*_w/contact_mask）。
5. 非 headless 模式下 viewport 默认固定距离跟踪机器人，鼠标 Orbit 旋转后仍保持跟踪。

运行示例：
    python legged_lab/scripts/sim2isaac_joy.py --policy Exported_policy/policy.pt
    python legged_lab/scripts/sim2isaac_joy.py --use_implicit_motor --use_vel_cmd
    python legged_lab/scripts/sim2isaac_joy.py --use_vel_cmd --plot_joints
    python legged_lab/scripts/sim2isaac_joy.py --use_vel_cmd --plot_joints --no_show_plots
    python legged_lab/scripts/sim2isaac_joy.py --use_vel_cmd --record_pkl --record_fps 100
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
import signal
import sys
import math
import numpy as np
from isaaclab.app import AppLauncher


# --- Argument Parsing ---
def parse_args() -> argparse.Namespace:
    """Parse script and IsaacLab launcher arguments."""
    parser = argparse.ArgumentParser(description="Run ultra_GameYaw policy.pt in IsaacSim with joystick commands.")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Registered IsaacLab task name.")
    parser.add_argument("--policy", type=str, default=DEFAULT_POLICY, help="Path to TorchScript policy.pt.")
    parser.add_argument("--duration", type=float, default=0.0, help="Sim seconds; <=0 runs until IsaacSim exits.")
    parser.add_argument(
        "--plot_joints",
        action="store_true",
        help="Plot and save all exit diagnostics: base velocity, T-N scatter, joint positions, velocity and torque.",
    )
    parser.add_argument("--plot_dir", type=str, default=DEFAULT_PLOT_DIR, help="Directory to save plot PNGs when --plot_joints is enabled.")
    parser.add_argument("--no_show_plots", action="store_true", help="Save plot PNGs without opening matplotlib windows.")
    parser.add_argument(
        "--use_vel_cmd",
        "--use-vel-cmd",
        action="store_true",
        help=f"Use predefined velocity commands instead of joystick input: {DEFAULT_VEL_CMD_SEQUENCE_HELP}.",
    )
    parser.add_argument(
        "--use_implicit_motor",
        "--use-implicit-motor",
        action="store_true",
        help="Override custom delayed/T-N actuators with IsaacLab ImplicitActuatorCfg for sim2isaac.",
    )
    parser.add_argument(
        "--record_pkl",
        action="store_true",
        help="Record env0 joint/body world-frame trajectory and save a cleaned-style pkl on exit.",
    )
    parser.add_argument(
        "--record_fps",
        type=int,
        default=100,
        help="Target FPS stored in the recorded pkl (frames are downsampled from the policy rate).",
    )
    parser.add_argument(
        "--record_pkl_file",
        type=str,
        default=None,
        help=f"Output path for the recorded pkl. Defaults to a timestamped file under {DEFAULT_RECORD_DIR}.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _hydra_args = parser.parse_known_args()
    legacy_plot_args = {"--plot_TN", "--plot-TN", "--plot_joints_pos", "--plot-joints-pos"}
    used_legacy_plot_args = sorted(arg for arg in _hydra_args if arg in legacy_plot_args)
    if used_legacy_plot_args:
        parser.error(
            f"{', '.join(used_legacy_plot_args)} 已取消；现在统一使用 --plot_joints 自动绘制并保存所有诊断图。"
        )
    return args_cli

# --- Constants & Config ---
LEGGED_LAB_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
RSL_RL_ROOT_DIR = os.path.join(LEGGED_LAB_ROOT_DIR, "rsl_rl")
if RSL_RL_ROOT_DIR not in sys.path:
    sys.path.insert(0, RSL_RL_ROOT_DIR)

DEFAULT_TASK = "ultra_game_yaw_run"
DEFAULT_POLICY = os.path.join(LEGGED_LAB_ROOT_DIR, "Exported_policy", "policy.pt")
DEFAULT_PLOT_DIR = os.path.join(LEGGED_LAB_ROOT_DIR, "logs", "sim2isaac_joy_plots")
DEFAULT_RECORD_DIR = os.path.join(LEGGED_LAB_ROOT_DIR, "logs", "sim2isaac_records")
DEFAULT_VEL_CMD_SEQUENCE = (
    (100.0, (15.0, 0.0, 0.0)),
    (1.0, (10.0, 0.0, 0.0)),
    (100.0, (0.0, 0.0, 0.0)),
)
DEFAULT_VEL_CMD_SEQUENCE_HELP = " -> ".join(
    f"({cmd[0]:g}, {cmd[1]:g}, {cmd[2]:g}) m/s for {duration:g}s"
    for duration, cmd in DEFAULT_VEL_CMD_SEQUENCE
)
MAX_CMD = np.array([20.0, 0.5, 0.5], dtype=np.float64)
MAX_CMD_VX_NEG = 0.5
CMD_DEADZONE = 0.05
# DEPLOY_INIT_BASE_POS = (0.0, 0.0, 1.23)
DEPLOY_INIT_BASE_POS = (0.0, 0.0, 1.23)

# 初始站立姿态
DEPLOY_INIT_JOINT_POS = {
    "knee_pitch_l_joint": math.radians(30.0),
    "knee_pitch_r_joint": math.radians(30.0),
    "hip_pitch_l_joint": math.radians(-15.0),
    "hip_pitch_r_joint": math.radians(-15.0),
    "ankle_pitch_l_joint": math.radians(-15.0),
    "ankle_pitch_r_joint": math.radians(-15.0),
    "hip_roll_l_joint": 0.0,
    "hip_roll_r_joint": 0.0,
    "hip_yaw_l_joint": 0.0,
    "hip_yaw_r_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "shoulder_pitch_l_joint": 0.5,
    "shoulder_pitch_r_joint": 0.5,
}

# 准备跑姿势
# DEPLOY_INIT_JOINT_POS = {
#     "hip_yaw_l_joint": 0.41250,
#     "hip_roll_l_joint": -0.29305,
#     "hip_pitch_l_joint": -0.41659,
#     "knee_pitch_l_joint": 0.62464,
#     "ankle_pitch_l_joint": -0.56119,
#     "hip_yaw_r_joint": -0.05288,
#     "hip_roll_r_joint": -0.06540,
#     "hip_pitch_r_joint": -1.46116,
#     "knee_pitch_r_joint": 0.78403,
#     "ankle_pitch_r_joint": 0.01416,
#     "waist_yaw_joint": -0.24905,
#     "shoulder_pitch_l_joint": -1.01854,
#     "shoulder_pitch_r_joint": -0.10722,
# }

DEFAULT_CAMERA_DISTANCE = 5.0
DEFAULT_CAMERA_TARGET_HEIGHT = 0.0
CAMERA_OFFSET_EPS = 1e-6

GAMEPAD_PROFILES = {
    "xbox": (0, 1, 3),
    "betop": (0, 1, 2),
    "ps": (0, 1, 2),
}


def detect_gamepad_profile(name: str) -> str:
    """Infer the axis mapping profile from a joystick device name.

    Args:
        name: Device name reported by pygame.

    Returns:
        The profile key used by ``GAMEPAD_PROFILES``.
    """
    name_lower = name.lower()
    if any(k in name_lower for k in ("betop", "北通", "bt", "gamepad")):
        return "betop"
    if any(k in name_lower for k in ("xbox", "xinput", "045e")):
        return "xbox"
    if any(k in name_lower for k in ("sony", "dualshock", "dualsense", "ps4", "ps5", "054c")):
        return "ps"
    return "xbox"


def apply_deadzone(val: float, threshold: float = CMD_DEADZONE) -> float:
    """Apply a centered joystick deadzone and rescale the remaining range.

    Args:
        val: Raw normalized joystick axis value.
        threshold: Absolute value below which the axis is treated as zero.

    Returns:
        The deadzone-filtered axis value in ``[-1, 1]``.
    """
    if abs(val) < threshold:
        return 0.0
    sign = 1.0 if val >= 0.0 else -1.0
    return sign * (abs(val) - threshold) / (1.0 - threshold)


def map_gamepad_axes_to_command(left_x: float, left_y: float, right_x: float) -> np.ndarray:
    """Convert normalized joystick axes to a velocity command vector.

    Args:
        left_x: Left stick horizontal axis.
        left_y: Left stick vertical axis after sign correction.
        right_x: Right stick horizontal axis.

    Returns:
        Body-frame command ``[vx, vy, yaw_rate]``.
    """
    vx_axis = apply_deadzone(left_y)
    vy_axis = apply_deadzone(left_x)
    yaw_axis = apply_deadzone(right_x)

    if vx_axis >= 0.0:
        vx = vx_axis * MAX_CMD[0]
    else:
        vx = vx_axis * MAX_CMD_VX_NEG
    return np.array(
        [
            vx,
            vy_axis * MAX_CMD[1],
            yaw_axis * MAX_CMD[2],
        ],
        dtype=np.float64,
    )


def format_vel_cmd_sequence(sequence=DEFAULT_VEL_CMD_SEQUENCE) -> str:
    """Format a scripted velocity command sequence for logs and CLI help.

    Args:
        sequence: Iterable of ``(duration_s, command)`` pairs.

    Returns:
        A compact human-readable sequence description.
    """
    return " -> ".join(
        f"({cmd[0]:g}, {cmd[1]:g}, {cmd[2]:g}) m/s for {duration:g}s"
        for duration, cmd in sequence
    )


def predefined_velocity_command(elapsed_time_s: float) -> np.ndarray:
    """Select the scripted velocity command for elapsed simulation time.

    Args:
        elapsed_time_s: Seconds elapsed since entering the policy loop.

    Returns:
        Body-frame command ``[vx, vy, yaw_rate]``. The command returns to zero
        after all configured segments finish.
    """
    elapsed_time_s = max(0.0, elapsed_time_s)
    segment_start_s = 0.0
    for duration_s, command in DEFAULT_VEL_CMD_SEQUENCE:
        segment_end_s = segment_start_s + duration_s
        if elapsed_time_s < segment_end_s:
            return np.asarray(command, dtype=np.float64)
        segment_start_s = segment_end_s
    return np.zeros(3, dtype=np.float64)


class GamepadCommandSource:
    """Read body-frame velocity commands from the first available gamepad.

    Attributes:
        pygame: Imported pygame module, or ``None`` when unavailable.
        joystick: First initialized joystick, or ``None`` when no device exists.
        profile: Axis mapping profile selected from ``GAMEPAD_PROFILES``.
    """

    def __init__(self):
        """Initialize an empty gamepad command source."""
        self.pygame = None
        self.joystick = None
        self.profile = "xbox"

    def setup(self) -> bool:
        """Initialize pygame and select the first connected joystick.

        The command source remains valid when pygame or a joystick is missing;
        subsequent reads simply return a zero command.
        """
        try:
            import pygame
        except ImportError:
            print("[WARNING] pygame not installed. Install with: pip install pygame")
            return False
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            print("[WARNING] No gamepad detected. Command will stay at zero.")
            self.pygame = pygame
            return False
        self.pygame = pygame
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.profile = detect_gamepad_profile(self.joystick.get_name())
        print(f"[INFO] Gamepad detected: {self.joystick.get_name()} -> profile={self.profile}")
        return True

    def read_command_target(self) -> np.ndarray:
        """Read the current joystick axes and return the target command.

        Returns:
            Body-frame command ``[vx, vy, yaw_rate]``. Returns zeros when no
            joystick is available.
        """
        if self.pygame is None or self.joystick is None:
            return np.zeros(3, dtype=np.float64)
        self.pygame.event.pump()
        axis_left_x, axis_left_y, axis_right_x = GAMEPAD_PROFILES.get(
            self.profile, GAMEPAD_PROFILES["xbox"]
        )
        left_x = self.joystick.get_axis(axis_left_x)
        left_y = -self.joystick.get_axis(axis_left_y)
        right_x = self.joystick.get_axis(axis_right_x) if self.joystick.get_numaxes() > axis_right_x else 0.0
        return map_gamepad_axes_to_command(left_x=left_x, left_y=left_y, right_x=right_x)

    def close(self) -> None:
        """Release pygame resources if they were initialized."""
        if self.pygame is not None:
            self.pygame.quit()


def _clear_event_if_present(events, name: str) -> None:
    """Disable a named event config when it exists on the event container.

    Args:
        events: Domain-randomization event config container.
        name: Event attribute name to clear.
    """
    if hasattr(events, name):
        setattr(events, name, None)


def apply_deploy_init_state(robot_cfg) -> None:
    """Apply script-level deploy initial base position and motor joint positions."""
    base_pos = tuple(float(v) for v in DEPLOY_INIT_BASE_POS)
    if len(base_pos) != 3 or not np.all(np.isfinite(base_pos)):
        raise ValueError(f"DEPLOY_INIT_BASE_POS must contain 3 values, got {DEPLOY_INIT_BASE_POS!r}")
    if not DEPLOY_INIT_JOINT_POS:
        raise ValueError("DEPLOY_INIT_JOINT_POS is empty; provide at least one motor joint initial position.")
    joint_pos = {}
    for joint_name, value in DEPLOY_INIT_JOINT_POS.items():
        if not isinstance(joint_name, str) or not joint_name:
            raise ValueError(f"Invalid deploy init joint name: {joint_name!r}")
        joint_pos[joint_name] = float(value)
        if not np.isfinite(joint_pos[joint_name]):
            raise ValueError(f"Invalid deploy init joint position for {joint_name!r}: {value!r}")
    robot_cfg.init_state.pos = base_pos
    robot_cfg.init_state.joint_pos = joint_pos
    robot_cfg.init_state.joint_vel = {".*": 0.0}


def use_implicit_motor_actuators(robot_cfg) -> list[str]:
    """Replace robot actuator configs with IsaacLab implicit motors for sim2isaac."""
    from isaaclab.actuators import ImplicitActuatorCfg

    actuators = getattr(robot_cfg, "actuators", None)
    if not isinstance(actuators, dict) or len(actuators) == 0:
        raise ValueError("robot_cfg.actuators must be a non-empty dict before enabling --use_implicit_motor.")

    implicit_actuators = {}
    for name, actuator_cfg in actuators.items():
        joint_names_expr = getattr(actuator_cfg, "joint_names_expr", None)
        stiffness = getattr(actuator_cfg, "stiffness", None)
        damping = getattr(actuator_cfg, "damping", None)
        if joint_names_expr is None:
            raise ValueError(f"Actuator {name!r} has no joint_names_expr; cannot convert to implicit motor.")
        if stiffness is None or damping is None:
            raise ValueError(f"Actuator {name!r} must define stiffness and damping for implicit motor conversion.")
        if isinstance(joint_names_expr, tuple):
            joint_names_expr = list(joint_names_expr)

        implicit_actuators[name] = ImplicitActuatorCfg(
            joint_names_expr=joint_names_expr,
            effort_limit_sim=getattr(actuator_cfg, "effort_limit_sim", None),
            velocity_limit_sim=getattr(actuator_cfg, "velocity_limit_sim", None),
            stiffness=stiffness,
            damping=damping,
            armature=getattr(actuator_cfg, "armature", None),
            friction=getattr(actuator_cfg, "friction", None),
            dynamic_friction=getattr(actuator_cfg, "dynamic_friction", None),
            viscous_friction=getattr(actuator_cfg, "viscous_friction", None),
        )

    robot_cfg.actuators = implicit_actuators
    return list(implicit_actuators.keys())


def apply_deploy_initial_sim_state(env) -> None:
    """Teleport the created env to the deterministic deploy initial state."""
    import torch

    robot = env.robot
    env_ids = torch.arange(env.num_envs, device=env.device)
    root_state = robot.data.default_root_state.clone()
    if hasattr(env.scene, "env_origins"):
        root_state[:, :3] += env.scene.env_origins
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(robot.data.default_joint_vel)
    if hasattr(env, "joint_pos_bias"):
        env.joint_pos_bias.zero_()

    robot.write_root_state_to_sim(root_state, env_ids)
    robot.write_joint_position_to_sim(joint_pos, env_ids=env_ids)
    robot.write_joint_velocity_to_sim(joint_vel, env_ids=env_ids)
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    if hasattr(env, "action_buffer"):
        env.action_buffer.reset(env_ids)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(dt=env.step_dt)


def configure_deploy_env_cfg(env_cfg, use_implicit_motor: bool = False) -> None:
    """Configure the IsaacLab environment for deterministic single-env deployment.

    Args:
        env_cfg: UltraGameYaw environment config object to mutate in place.
        use_implicit_motor: Replace custom actuator configs with implicit motors.
    """
    env_cfg.noise.add_noise = False
    for event_name in (
        "push_robot",
        "randomize_rigid_body_inertia",
        "add_base_mass",
        "add_others_mass",
        "randomize_rigid_body_com",
        "randomize_joint_pos_bias",
        "randomize_joint_armature",
        "randomize_joint_friction",
        "randomize_actuator_friction",
        "randomize_motor_strength",
        "randomize_actuator_gains",
        "physics_material",
        "reset_base",
        "reset_robot_joints",
    ):
        _clear_event_if_present(env_cfg.domain_rand.events, event_name)

    if hasattr(env_cfg.domain_rand, "action_delay"):
        env_cfg.domain_rand.action_delay.enable = False

    env_cfg.scene.max_episode_length_s = 1000.0
    env_cfg.scene.seed = 42
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 2.5
    apply_deploy_init_state(env_cfg.scene.robot)
    if use_implicit_motor:
        actuator_names = use_implicit_motor_actuators(env_cfg.scene.robot)
        print(
            f"[sim2isaac] --use_implicit_motor enabled; replaced actuators with ImplicitActuatorCfg: "
            f"{', '.join(actuator_names)}",
            flush=True,
        )
    env_cfg.scene.terrain_generator = None
    env_cfg.scene.terrain_type = "plane"
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)
    env_cfg.scene.height_scanner.enable_height_scan = False
    env_cfg.scene.height_scanner.enable_feet_height_scan = False

    env_cfg.commands.debug_vis = True
    env_cfg.commands.heading_command = False
    env_cfg.commands.rel_standing_envs = 0.0
    # 关闭站/走/跑分桶采样：分桶 + 零范围会在 reset 时把 is_standing_env 锁成 True，
    # CustomVelocityCommand 每个 env.step 会据此把 vel_command_b 清零，导致手动注入的
    # 速度命令永远进不了观测（robot 不动）。与可用的 play.py 保持一致。
    if hasattr(env_cfg.commands, "cmd_bucket_probs"):
        env_cfg.commands.cmd_bucket_probs = None
    env_cfg.commands.use_curriculum = False
    env_cfg.commands.run_high_speed_sample_frac = None
    env_cfg.commands.resampling_time_range = (1000.0, 1000.0)
    env_cfg.commands.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.ranges.heading = (0.0, 0.0)


def vec3_to_numpy(value) -> np.ndarray:
    """Convert a tensor-like or sequence vec3 to a float64 numpy vector."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).reshape(3)


def get_viewer_origin_w(env, viewer_cfg) -> np.ndarray:
    """Read the world-space origin used by the viewport follow camera."""
    origin_type = getattr(viewer_cfg, "origin_type", "world")
    env_index = int(getattr(viewer_cfg, "env_index", 0))
    if origin_type == "world":
        return np.zeros(3, dtype=np.float64)
    if origin_type == "env":
        return vec3_to_numpy(env.scene.env_origins[env_index])
    if origin_type in ("asset_root", "asset_body"):
        asset_name = getattr(viewer_cfg, "asset_name", None)
        if asset_name is None:
            raise ValueError(f"Viewer origin_type={origin_type!r} requires asset_name.")
        asset = env.scene[asset_name]
        if origin_type == "asset_root":
            return vec3_to_numpy(asset.data.root_pos_w[env_index])
        body_name = getattr(viewer_cfg, "body_name", None)
        if body_name is None:
            raise ValueError("Viewer origin_type='asset_body' requires body_name.")
        body_id, _ = asset.find_bodies(body_name)
        if not body_id:
            raise ValueError(f"Viewer body {body_name!r} not found in asset {asset_name!r}.")
        return vec3_to_numpy(asset.data.body_pos_w[env_index, body_id[0]])
    raise ValueError(f"Unsupported viewer origin_type: {origin_type!r}")


def get_camera_target_w(env, viewer_cfg, target_height: float) -> np.ndarray:
    """Compute the followed camera target in world coordinates."""
    lookat = np.asarray(getattr(viewer_cfg, "lookat", (0.0, 0.0, 0.0)), dtype=np.float64)
    return get_viewer_origin_w(env, viewer_cfg) + lookat + np.array([0.0, 0.0, target_height], dtype=np.float64)


def normalize_camera_offset(offset: np.ndarray, fallback_offset: np.ndarray, distance: float) -> np.ndarray:
    """Keep the current orbit direction and optionally force a fixed camera distance."""
    offset = np.asarray(offset, dtype=np.float64).reshape(3)
    if np.linalg.norm(offset) < CAMERA_OFFSET_EPS:
        offset = np.asarray(fallback_offset, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(offset))
    if norm < CAMERA_OFFSET_EPS:
        offset = np.array([max(distance, DEFAULT_CAMERA_DISTANCE), 0.0, DEFAULT_CAMERA_DISTANCE * 0.4], dtype=np.float64)
        norm = float(np.linalg.norm(offset))
    return offset / norm * distance if distance > 0.0 else offset


def read_viewport_camera_offset(camera_state, fallback_offset: np.ndarray) -> np.ndarray:
    """Read current viewport orbit offset so mouse rotation is preserved while following."""
    try:
        eye = np.asarray(camera_state.position_world, dtype=np.float64).reshape(3)
        target = np.asarray(camera_state.target_world, dtype=np.float64).reshape(3)
    except Exception:
        return np.asarray(fallback_offset, dtype=np.float64).reshape(3)
    return eye - target


def setup_viewport_follow_camera(env, viewer_cfg, default_on: bool = True):
    """Follow the configured robot target while preserving viewport mouse orbit direction."""
    import carb.input
    import omni.appwindow
    import omni.kit.app
    from omni.kit.viewport.utility import get_active_viewport
    from omni.kit.viewport.utility.camera_state import ViewportCameraState

    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("No active IsaacSim viewport found for follow camera.")

    camera_state = ViewportCameraState(viewport=viewport)
    enabled = [default_on]
    warned = [False]
    distance = DEFAULT_CAMERA_DISTANCE
    target_height = DEFAULT_CAMERA_TARGET_HEIGHT
    fallback_offset = np.asarray(viewer_cfg.eye, dtype=np.float64) - np.asarray(viewer_cfg.lookat, dtype=np.float64)

    def _apply_camera(use_fallback_offset: bool = False) -> None:
        target = get_camera_target_w(env, viewer_cfg, target_height)
        offset = fallback_offset if use_fallback_offset else read_viewport_camera_offset(camera_state, fallback_offset)
        offset = normalize_camera_offset(offset, fallback_offset, distance)
        env.sim.set_camera_view(eye=tuple((target + offset).tolist()), target=tuple(target.tolist()))

    def _on_update(event) -> None:
        if not enabled[0]:
            return
        try:
            _apply_camera()
        except Exception as exc:
            if not warned[0]:
                print(f"[Camera] follow update skipped: {exc}", flush=True)
                warned[0] = True

    _apply_camera(use_fallback_offset=True)
    app_interface = omni.kit.app.get_app_interface()
    update_sub = app_interface.get_post_update_event_stream().create_subscription_to_pop(_on_update)

    appwindow = omni.appwindow.get_default_app_window()
    keyboard = appwindow.get_keyboard()
    input_iface = carb.input.acquire_input_interface()

    def _on_key(event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS and event.input.name == "F":
            enabled[0] = not enabled[0]
            print(f"[Camera] follow = {'ON' if enabled[0] else 'OFF'}", flush=True)
        return True

    key_sub = input_iface.subscribe_to_keyboard_events(keyboard, _on_key)
    print(
        f"[Camera] follow = {'ON' if enabled[0] else 'OFF'}  "
        f"(press F to toggle; distance={'current zoom' if distance <= 0.0 else f'{distance:g}m'})",
        flush=True,
    )
    return camera_state, update_sub, key_sub, enabled


def _assign_command_buffer(buffer, command: np.ndarray) -> None:
    """Write a 3D velocity command into a numpy or torch command buffer.

    Args:
        buffer: Command buffer with shape ``(num_envs, >=3)``.
        command: Body-frame command ``[vx, vy, yaw_rate]``.
    """
    if hasattr(buffer, "device"):
        import torch

        value = torch.as_tensor(command, dtype=buffer.dtype, device=buffer.device)
        buffer[:, 0] = value[0]
        buffer[:, 1] = value[1]
        buffer[:, 2] = value[2]
    else:
        buffer[:, 0] = command[0]
        buffer[:, 1] = command[1]
        buffer[:, 2] = command[2]


def apply_command_to_env(env, command: np.ndarray) -> None:
    """Pin all known command-generator buffers to the requested command.

    Args:
        env: IsaacLab environment with a command generator.
        command: Body-frame command ``[vx, vy, yaw_rate]``.
    """
    generator = env.command_generator
    for attr in ("command", "vel_command_b", "cmd_target_b"):
        if hasattr(generator, attr):
            _assign_command_buffer(getattr(generator, attr), command)
    # CustomVelocityCommand re-zeros vel_command_b every step for "standing"
    # envs inside env.step()->command_generator.compute(). The deploy config uses
    # zero command ranges, so the reset latches is_standing_env=True and the
    # injected command would be wiped before the policy observes it. Clear the
    # standing/heading latches so the manual command persists.
    for flag in ("is_standing_env", "is_heading_env"):
        if hasattr(generator, flag):
            getattr(generator, flag)[:] = False


def actor_obs_history_length(env) -> int:
    """Read actor observation history length from env config."""
    cfg = getattr(env, "cfg", None)
    robot_cfg = getattr(cfg, "robot", None)
    return max(1, int(getattr(robot_cfg, "actor_obs_history_length", 1)))


def refresh_commanded_observations(env, command: np.ndarray, fill_history: bool = False):
    """Apply command and rebuild actor observations with that command visible to the policy."""
    apply_command_to_env(env, command)
    repeat = actor_obs_history_length(env) if fill_history else 1
    obs = None
    extras = None
    for _ in range(repeat):
        apply_command_to_env(env, command)
        obs, extras = env.get_observations()
    return obs, extras


JOINT_TYPE_TN_PARAMS = {
    "hip_yaw": {"X1": 10.0, "X2": 20.9, "Y1": 150.0, "Y2": 250.0},
    "hip_roll": {"X1": 12.0, "X2": 18.8, "Y1": 600.0, "Y2": 657.0},
    "hip_pitch": {"X1": 17.0, "X2": 26.1, "Y1": 600.0, "Y2": 657.0},
    "knee_pitch": {"X1": 17.0, "X2": 26.1, "Y1": 600.0, "Y2": 657.0},
    "ankle_pitch": {"X1": 17.0, "X2": 26.1, "Y1": 350.0, "Y2": 400.0},
    "shoulder_pitch": {"X1": 13.0, "X2": 26.1, "Y1": 100.0, "Y2": 140.0},
    "waist_yaw": {"X1": 10.0, "X2": 20.9, "Y1": 150.0, "Y2": 250.0},
}


def joint_type_from_name(joint_name: str) -> str:
    """Map a robot joint name to the actuator type used for T-N limits.

    Args:
        joint_name: Robot joint name from IsaacLab.

    Returns:
        Actuator type key used by ``JOINT_TYPE_TN_PARAMS``.

    Raises:
        KeyError: If the joint name has no known actuator type mapping.
    """
    if joint_name.startswith("hip_yaw"):
        return "hip_yaw"
    if joint_name.startswith("hip_roll"):
        return "hip_roll"
    if joint_name.startswith("hip_pitch"):
        return "hip_pitch"
    if joint_name.startswith("knee_pitch"):
        return "knee_pitch"
    if joint_name.startswith("ankle_pitch"):
        return "ankle_pitch"
    if joint_name.startswith("shoulder_pitch"):
        return "shoulder_pitch"
    if joint_name.startswith("waist_yaw"):
        return "waist_yaw"
    raise KeyError(f"No T-N joint type mapping for {joint_name!r}")


def tn_curve_params_for_joint(joint_name: str) -> tuple[float, float, float, float]:
    """Return the T-N curve parameters for a joint name.

    Args:
        joint_name: Robot joint name from IsaacLab.

    Returns:
        Tuple ``(X1, X2, Y1, Y2)`` defining the T-N envelope.
    """
    params = JOINT_TYPE_TN_PARAMS[joint_type_from_name(joint_name)]
    return params["X1"], params["X2"], params["Y1"], params["Y2"]


def get_tn_plot_joint_names(joint_names: list[str]) -> list[str]:
    """Filter joint names down to the joints with known T-N parameters.

    Args:
        joint_names: Joint names reported by IsaacLab.

    Returns:
        Ordered subset of joints that can be plotted against T-N limits.
    """
    names = []
    for joint_name in joint_names:
        try:
            joint_type_from_name(joint_name)
        except KeyError:
            continue
        names.append(joint_name)
    return names


def fallback_joint_velocity_limit(joint_name: str) -> float | None:
    """Return fallback max |velocity| from the known T-N parameters."""
    try:
        return float(tn_curve_params_for_joint(joint_name)[1])
    except KeyError:
        return None


def fallback_joint_effort_limit(joint_name: str) -> float | None:
    """Return fallback max |torque| from the known T-N parameters."""
    try:
        return float(tn_curve_params_for_joint(joint_name)[3])
    except KeyError:
        return None


def limit_value_at(limits, idx: int) -> float | None:
    """Read one positive per-joint limit from an optional vector."""
    if limits is None:
        return None
    arr = np.asarray(limits, dtype=np.float64).reshape(-1)
    if idx >= arr.shape[0]:
        return None
    value = float(abs(arr[idx]))
    if not np.isfinite(value) or value <= 0.0:
        return None
    return value


def tn_torque_limits_for_velocity(joint_name: str, joint_vel: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute the T-N torque upper/lower envelope for a joint velocity trace."""
    try:
        x1, x2, y1, y2 = tn_curve_params_for_joint(joint_name)
    except KeyError:
        return None
    av = np.abs(joint_vel)
    lim = lambda y: np.where(av < x1, y, np.clip(-y / (x2 - x1) * (av - x1) + y, 0.0, None))
    upper = np.where(joint_vel > 0, lim(y1), lim(y2))
    lower = np.where(joint_vel > 0, -lim(y2), -lim(y1))
    return upper, lower


def select_effort_tensor(robot_data):
    """Select the available torque/effort tensor from IsaacLab robot data.

    Args:
        robot_data: IsaacLab articulation data object.

    Returns:
        The first available torque/effort tensor.

    Raises:
        AttributeError: If no supported torque/effort tensor exists.
    """
    for attr in ("applied_torque", "computed_torque", "joint_effort"):
        if hasattr(robot_data, attr):
            return getattr(robot_data, attr)
    raise AttributeError("Robot data has no applied_torque, computed_torque, or joint_effort field")


def tensor_to_numpy_row(value):
    """Copy the first environment row from a tensor-like value to numpy.

    Args:
        value: Torch tensor or numpy-like array with env dimension first.

    Returns:
        A detached numpy copy of row zero.
    """
    if hasattr(value, "detach"):
        return value[0].detach().cpu().numpy().copy()
    return np.asarray(value[0], dtype=np.float64).copy()


def quat_apply_inverse_wxyz(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Apply inverse quaternion rotation for wxyz quaternion and 3D vector."""
    w, x, y, z = [float(quat_wxyz[i]) for i in range(4)]
    qv = np.array([x, y, z], dtype=np.float64)
    v = np.asarray(vec, dtype=np.float64)
    return v * (2.0 * w * w - 1.0) - np.cross(qv, v) * w * 2.0 + qv * np.dot(qv, v) * 2.0


def yaw_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    """Extract yaw-only quaternion from wxyz root quaternion."""
    w, x, y, z = [float(quat_wxyz[i]) for i in range(4)]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = 0.5 * yaw
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def root_yaw_velocity_sample(robot_data) -> np.ndarray:
    """Return env0 [yaw-frame vx, yaw-frame vy, yaw angular velocity]."""
    for attr in ("root_quat_w", "root_lin_vel_w", "root_ang_vel_w"):
        if not hasattr(robot_data, attr):
            raise AttributeError(f"Robot data has no {attr}; cannot record yaw-frame base velocity")
    yaw_q = yaw_quat_wxyz(tensor_to_numpy_row(robot_data.root_quat_w))
    lin_yaw = quat_apply_inverse_wxyz(yaw_q, tensor_to_numpy_row(robot_data.root_lin_vel_w))
    ang_yaw = quat_apply_inverse_wxyz(yaw_q, tensor_to_numpy_row(robot_data.root_ang_vel_w))
    return np.array([lin_yaw[0], lin_yaw[1], ang_yaw[2]], dtype=np.float64)


def tensor_to_numpy_first_env(value):
    """Copy the first environment entry from an optional tensor-like value.

    Args:
        value: Optional torch tensor or numpy-like array with env dimension first.

    Returns:
        A detached numpy copy of entry zero, or ``None`` when ``value`` is
        ``None``.
    """
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value[0].detach().cpu().numpy().copy()
    return np.asarray(value[0], dtype=np.float64).copy()


def tensor_to_numpy_optional_limit_row(value):
    """Copy optional per-joint limits from a tensor-like value."""
    if value is None:
        return None
    if hasattr(value, "detach"):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value, dtype=np.float64)
    if arr.ndim >= 2:
        arr = arr[0]
    return np.asarray(arr, dtype=np.float64).copy()


def _subplot_shape(num_items: int) -> tuple[int, int]:
    """Choose a compact subplot grid for a set of traces.

    Args:
        num_items: Number of plots to display.

    Returns:
        Tuple ``(rows, cols)`` for matplotlib subplots.
    """
    if num_items <= 1:
        return max(1, num_items), 1
    return (num_items + 1) // 2, 2


def configure_matplotlib_backend(show: bool) -> None:
    """Use a non-GUI matplotlib backend when plots only need to be saved."""
    if not show:
        import matplotlib

        matplotlib.use("Agg", force=True)


def prepare_plot_output_dir(plot_dir: str | None) -> str | None:
    """Create and normalize the plot output directory."""
    if plot_dir is None or str(plot_dir).strip() == "":
        return None
    out_dir = os.path.abspath(os.path.expanduser(plot_dir))
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_plot_figure(fig, save_dir: str | None, filename: str) -> str | None:
    """Save a matplotlib figure when a save directory is configured."""
    if save_dir is None:
        return None
    out_path = os.path.join(save_dir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[sim2isaac] Saved plot: {out_path}", flush=True)
    return out_path


def install_sigint_exit_handler(stop_state: dict) -> object:
    """Install a Ctrl+C handler that lets the sim loop exit through cleanup once."""
    previous = signal.getsignal(signal.SIGINT)

    def _handle_sigint(_signum, _frame):
        stop_state["count"] = int(stop_state.get("count", 0)) + 1
        stop_state["requested"] = True
        if stop_state["count"] == 1:
            print("[sim2isaac] Ctrl+C received; exiting loop and saving plots. Press Ctrl+C again to force.", flush=True)
            return
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_sigint)
    return previous


class IsaacJointPlotRecorder:
    """Collect IsaacLab base/joint samples and render debug plots on exit.

    Attributes:
        enable: Whether to record plot samples.
        time: Recorded sample timestamps.
        joint_pos: Recorded joint position samples.
        joint_vel: Recorded joint velocity samples.
        joint_tau: Recorded joint torque/effort samples.
        base_yaw_vel: Recorded yaw-frame vx/vy and yaw angular velocity samples.
        joint_pos_limits: Position limits captured from robot data when present.
        joint_vel_limits: Velocity limits captured from robot data when present.
        joint_effort_limits: Effort limits captured from robot data when present.
    """

    def __init__(self, enable: bool):
        """Create an empty plot recorder.

        Args:
            enable: Enable joint/base data collection for plots.
        """
        self.enable = enable
        self.time = []
        self.joint_pos = []
        self.joint_vel = []
        self.joint_tau = []
        self.base_yaw_vel = []
        self.joint_pos_limits = None
        self.joint_vel_limits = None
        self.joint_effort_limits = None

    def sample_count(self) -> int:
        """Return the number of recorded policy-step samples."""
        return len(self.time)

    def record(self, env) -> None:
        """Capture one policy-step sample from the IsaacLab environment.

        Args:
            env: IsaacLab environment after a policy step.
        """
        if not self.enable:
            return

        robot_data = env.robot.data
        self.time.append(float(env.episode_length_buf[0].item()) * float(env.step_dt))
        self.joint_pos.append(tensor_to_numpy_row(robot_data.joint_pos))
        self.joint_vel.append(tensor_to_numpy_row(robot_data.joint_vel))
        self.joint_tau.append(tensor_to_numpy_row(select_effort_tensor(robot_data)))
        self.base_yaw_vel.append(root_yaw_velocity_sample(robot_data))
        if self.joint_pos_limits is None:
            self.joint_pos_limits = tensor_to_numpy_first_env(getattr(robot_data, "joint_pos_limits", None))
        if self.joint_vel_limits is None:
            self.joint_vel_limits = tensor_to_numpy_optional_limit_row(getattr(robot_data, "joint_vel_limits", None))
        if self.joint_effort_limits is None:
            self.joint_effort_limits = tensor_to_numpy_optional_limit_row(getattr(robot_data, "joint_effort_limits", None))

    def plot_tn(self, joint_names: list[str], save_dir: str | None = None, run_tag: str = "") -> str | None:
        """Plot joint velocity/torque samples against T-N envelopes.

        Args:
            joint_names: Joint names aligned with the recorded robot data.
            save_dir: Optional directory for saved plot PNGs.
            run_tag: Timestamp tag used in saved plot filenames.
        """
        if not self.enable or not self.joint_vel or not self.joint_tau:
            return
        import matplotlib.pyplot as plt

        vel = np.asarray(self.joint_vel)
        tau = np.asarray(self.joint_tau)
        names = get_tn_plot_joint_names(joint_names)
        if not names:
            print("[WARNING] No known actuator joints found for T-N plot.")
            return

        rows, cols = _subplot_shape(len(names))
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 2.6 * rows))
        axes_flat = np.atleast_1d(axes).ravel()
        for ax, name in zip(axes_flat, names):
            idx = joint_names.index(name)
            if idx >= vel.shape[1] or idx >= tau.shape[1]:
                ax.axis("off")
                continue
            x1, x2, y1, y2 = tn_curve_params_for_joint(name)
            ax.scatter(vel[:, idx], tau[:, idx], s=2, alpha=0.4)
            vv = np.linspace(-x2, x2, 200)
            av = np.abs(vv)
            lim = lambda y: np.where(av < x1, y, np.clip(-y / (x2 - x1) * (av - x1) + y, 0.0, None))
            upper = np.where(vv > 0, lim(y1), lim(y2))
            lower = np.where(vv > 0, -lim(y2), -lim(y1))
            ax.plot(vv, upper, "r-", lw=1)
            ax.plot(vv, lower, "r-", lw=1)
            ax.set_title(name)
            ax.set_xlabel("velocity (rad/s)")
            ax.set_ylabel("torque (N*m)")
            ax.grid(True, alpha=0.3)
        for ax in axes_flat[len(names) :]:
            ax.axis("off")
        fig.suptitle("Motor T-N performance", fontsize=12)
        fig.tight_layout()
        return save_plot_figure(fig, save_dir, f"{run_tag}_tn.png")

    def plot_base_yaw_velocity(self, save_dir: str | None = None, run_tag: str = "") -> str | None:
        """Plot recorded yaw-frame vx/vy and yaw angular velocity."""
        if not self.enable or not self.time or not self.base_yaw_vel:
            return
        import matplotlib.pyplot as plt

        t = np.asarray(self.time, dtype=np.float64)
        v = np.asarray(self.base_yaw_vel, dtype=np.float64)
        fig, ax = plt.subplots(1, 1, figsize=(9, 3.2), sharex=True)
        labels = ("vx_yaw", "vy_yaw", "yaw_rate")
        colors = ("tab:blue", "tab:green", "tab:red")
        for idx, (label, color) in enumerate(zip(labels, colors)):
            ax.plot(t, v[:, idx], label=label, color=color, linewidth=1.0)
        ax.set_title("Base yaw-frame velocity")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("m/s, rad/s")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        return save_plot_figure(fig, save_dir, f"{run_tag}_base_yaw_velocity.png")

    def plot_joint_positions(self, joint_names: list[str], save_dir: str | None = None, run_tag: str = "") -> str | None:
        """Plot recorded joint positions with limits when available.

        Args:
            joint_names: Joint names aligned with the recorded robot data.
            save_dir: Optional directory for saved plot PNGs.
            run_tag: Timestamp tag used in saved plot filenames.
        """
        if not self.enable or not self.joint_pos:
            return
        if not joint_names:
            print("[WARNING] No joint names available; skip joint position plot.", flush=True)
            return
        import matplotlib.pyplot as plt

        t = np.asarray(self.time, dtype=np.float64)
        q = np.asarray(self.joint_pos, dtype=np.float64)
        rows, cols = _subplot_shape(len(joint_names))
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 2.1 * rows), sharex=True)
        axes_flat = np.atleast_1d(axes).ravel()
        for idx, (ax, name) in enumerate(zip(axes_flat, joint_names)):
            ax.plot(t, q[:, idx], lw=1.0)
            if self.joint_pos_limits is not None and idx < self.joint_pos_limits.shape[0]:
                ax.axhline(self.joint_pos_limits[idx, 0], color="tab:red", linestyle="--", linewidth=0.9)
                ax.axhline(self.joint_pos_limits[idx, 1], color="tab:red", linestyle="--", linewidth=0.9)
            ax.set_title(name)
            ax.set_ylabel("q (rad)")
            ax.grid(True, alpha=0.3)
        for ax in axes_flat[len(joint_names) :]:
            ax.axis("off")
        for ax in axes_flat[: len(joint_names)]:
            ax.set_xlabel("time (s)")
        fig.suptitle("Joint positions with limits", fontsize=12)
        fig.tight_layout()
        return save_plot_figure(fig, save_dir, f"{run_tag}_joint_positions.png")

    def plot_joint_velocity_torque(self, joint_names: list[str], save_dir: str | None = None, run_tag: str = "") -> str | None:
        """Plot recorded joint velocity and torque traces.

        Args:
            joint_names: Joint names aligned with the recorded robot data.
            save_dir: Optional directory for saved plot PNGs.
            run_tag: Timestamp tag used in saved plot filenames.
        """
        if not self.enable or not self.joint_vel or not self.joint_tau:
            return
        if not joint_names:
            print("[WARNING] No joint names available; skip joint velocity/torque plot.", flush=True)
            return
        import matplotlib.pyplot as plt

        t = np.asarray(self.time, dtype=np.float64)
        vel = np.asarray(self.joint_vel, dtype=np.float64)
        tau = np.asarray(self.joint_tau, dtype=np.float64)
        n = len(joint_names)
        fig, axes = plt.subplots(n, 2, figsize=(12, max(2.0 * n, 4.0)), sharex=True)
        axes = np.atleast_2d(axes)
        for idx, name in enumerate(joint_names):
            ax_vel, ax_tau = axes[idx, 0], axes[idx, 1]
            ax_vel.plot(t, vel[:, idx], color="tab:blue", linewidth=0.9)
            ax_tau.plot(t, tau[:, idx], color="tab:red", linewidth=0.9)
            vel_limit = limit_value_at(self.joint_vel_limits, idx) or fallback_joint_velocity_limit(name)
            tau_limit = limit_value_at(self.joint_effort_limits, idx) or fallback_joint_effort_limit(name)
            tau_dynamic_limits = tn_torque_limits_for_velocity(name, vel[:, idx])
            if vel_limit is not None:
                ax_vel.axhline(vel_limit, color="tab:red", linestyle="--", linewidth=0.8, label="limit")
                ax_vel.axhline(-vel_limit, color="tab:red", linestyle="--", linewidth=0.8)
            if tau_dynamic_limits is not None:
                tau_upper, tau_lower = tau_dynamic_limits
                ax_tau.plot(t, tau_upper, color="tab:red", linestyle="--", linewidth=0.8, label="T-N limit")
                ax_tau.plot(t, tau_lower, color="tab:red", linestyle="--", linewidth=0.8)
            elif tau_limit is not None:
                ax_tau.axhline(tau_limit, color="tab:red", linestyle="--", linewidth=0.8, label="limit")
                ax_tau.axhline(-tau_limit, color="tab:red", linestyle="--", linewidth=0.8)
            ax_vel.set_ylabel(name)
            ax_vel.grid(True, alpha=0.3)
            ax_tau.grid(True, alpha=0.3)
            ax_vel.set_title("velocity (rad/s)" if idx == 0 else "")
            ax_tau.set_title("torque (N*m)" if idx == 0 else "")
            if idx == 0:
                ax_vel.legend(loc="best", fontsize=8)
                ax_tau.legend(loc="best", fontsize=8)
        axes[-1, 0].set_xlabel("time (s)")
        axes[-1, 1].set_xlabel("time (s)")
        fig.suptitle("Joint velocity and torque", fontsize=12)
        fig.tight_layout()
        return save_plot_figure(fig, save_dir, f"{run_tag}_joint_velocity_torque.png")

    def plot_all(self, joint_names: list[str], plot_dir: str | None = None, show: bool = True) -> list[str]:
        """Render all enabled base and joint plots.

        Args:
            joint_names: Joint names aligned with the recorded robot data.
            plot_dir: Optional directory for saved plot PNGs.
            show: Whether to open matplotlib windows after all figures are saved.
        """
        if not self.enable:
            print("[WARNING] Exit plots disabled; run with --plot_joints to save plot images.", flush=True)
            return []
        if not self.time:
            print("[WARNING] --plot_joints enabled but no samples were recorded; skip plots.", flush=True)
            return []
        configure_matplotlib_backend(show)
        save_dir = prepare_plot_output_dir(plot_dir)
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        saved_paths = [
            self.plot_base_yaw_velocity(save_dir, run_tag),
            self.plot_tn(joint_names, save_dir, run_tag),
            self.plot_joint_positions(joint_names, save_dir, run_tag),
            self.plot_joint_velocity_torque(joint_names, save_dir, run_tag),
        ]
        saved_paths = [path for path in saved_paths if path]

        import matplotlib.pyplot as plt

        if show:
            plt.show()
        else:
            plt.close("all")
        return saved_paths


def compute_feet_contact_mask(env) -> np.ndarray | None:
    """Return env0 per-foot binary contact mask (>1N) from the contact sensor.

    Args:
        env: IsaacLab environment exposing ``contact_sensor`` and ``feet_cfg``.

    Returns:
        Float array of shape ``(num_feet,)`` with 1.0 on contact, or ``None``
        when the contact sensor / feet config is unavailable.
    """
    import torch

    sensor = getattr(env, "contact_sensor", None)
    feet_cfg = getattr(env, "feet_cfg", None)
    if sensor is None or feet_cfg is None or getattr(feet_cfg, "body_ids", None) is None:
        return None
    data = sensor.data
    forces_hist = getattr(data, "net_forces_w_history", None)
    if forces_hist is not None:
        mask = (torch.max(torch.norm(forces_hist[:, :, feet_cfg.body_ids], dim=-1), dim=1)[0] > 1.0).float()
    else:
        forces = data.net_forces_w
        mask = (torch.norm(forces[:, feet_cfg.body_ids], dim=-1) > 1.0).float()
    return mask[0].detach().cpu().numpy().copy()


class MotionPklRecorder:
    """Collect env0 joint/body world-frame trajectory and save a cleaned-style pkl.

    The saved fields mirror ``play.py`` 's cleaned npz (``joint_pos``, ``joint_vel``,
    ``body_pos_w``, ``body_quat_w``, ``body_lin_vel_w``, ``body_ang_vel_w``,
    ``contact_mask``) plus ``fps`` and ``joint_names`` / ``body_names`` metadata.

    Attributes:
        enable: Whether to record trajectory frames.
        requested_fps: User-requested output FPS.
        stride: Number of policy steps between recorded frames.
        output_fps: Achieved output FPS after downsampling.
    """

    FRAME_KEYS = (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "contact_mask",
    )

    def __init__(self, enable: bool, fps: int, step_dt: float):
        """Create a trajectory recorder and compute the policy->output downsample stride.

        Args:
            enable: Enable per-step trajectory collection.
            fps: Target output FPS stored in the pkl.
            step_dt: Policy step time in seconds.
        """
        self.enable = enable
        self.requested_fps = int(fps)
        if enable and self.requested_fps <= 0:
            raise ValueError(f"--record_fps must be positive, got {fps!r}")
        policy_fps = (1.0 / step_dt) if step_dt > 0.0 else float(self.requested_fps)
        self.stride = max(1, int(round(policy_fps / float(self.requested_fps)))) if enable else 1
        self.output_fps = policy_fps / self.stride
        self.step_index = 0
        self.frames = {key: [] for key in self.FRAME_KEYS}
        self.joint_names = None
        self.body_names = None

    def sample_count(self) -> int:
        """Return the number of recorded frames."""
        return len(self.frames["joint_pos"])

    def record(self, env) -> None:
        """Capture one downsampled trajectory frame from env0 after a policy step.

        Args:
            env: IsaacLab environment after a policy step.
        """
        if not self.enable:
            return
        take_frame = (self.step_index % self.stride == 0)
        self.step_index += 1
        if not take_frame:
            return
        data = env.robot.data
        self.frames["joint_pos"].append(tensor_to_numpy_row(data.joint_pos))
        self.frames["joint_vel"].append(tensor_to_numpy_row(data.joint_vel))
        self.frames["body_pos_w"].append(tensor_to_numpy_row(data.body_pos_w))
        self.frames["body_quat_w"].append(tensor_to_numpy_row(data.body_quat_w))
        self.frames["body_lin_vel_w"].append(tensor_to_numpy_row(data.body_lin_vel_w))
        self.frames["body_ang_vel_w"].append(tensor_to_numpy_row(data.body_ang_vel_w))
        mask = compute_feet_contact_mask(env)
        if mask is not None:
            self.frames["contact_mask"].append(mask)
        if self.joint_names is None:
            self.joint_names = list(getattr(data, "joint_names", []) or [])
        if self.body_names is None:
            self.body_names = list(getattr(data, "body_names", []) or [])

    def save(self, output_path: str) -> str | None:
        """Stack recorded frames and pickle them to ``output_path``.

        Args:
            output_path: Destination ``.pkl`` path.

        Returns:
            The absolute saved path, or ``None`` when nothing was recorded.
        """
        if not self.enable:
            return None
        if not self.frames["joint_pos"]:
            print("[sim2isaac] --record_pkl enabled but no frames were recorded; skip saving.", flush=True)
            return None
        import pickle

        payload = {"fps": int(round(self.output_fps))}
        for key in self.FRAME_KEYS:
            if self.frames[key]:
                payload[key] = np.stack(self.frames[key], axis=0)
        if self.joint_names:
            payload["joint_names"] = self.joint_names
        if self.body_names:
            payload["body_names"] = self.body_names
        out_path = os.path.abspath(os.path.expanduser(output_path))
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(
            f"[sim2isaac] Saved cleaned trajectory pkl: {out_path} "
            f"(frames={payload['joint_pos'].shape[0]}, fps={payload['fps']}, stride={self.stride})",
            flush=True,
        )
        return out_path


def get_robot_joint_names(env) -> list[str]:
    """Read controlled joint names from the IsaacLab environment.

    Args:
        env: IsaacLab environment containing ``env.robot``.

    Returns:
        Joint names aligned with robot data tensors.

    Raises:
        AttributeError: If no joint names or joint-position shape can be read.
    """
    robot = env.robot
    data = getattr(robot, "data", None)
    if data is not None and hasattr(data, "joint_names"):
        return list(data.joint_names)
    if hasattr(robot, "joint_names"):
        return list(robot.joint_names)
    joint_pos = getattr(data, "joint_pos", None) if data is not None else None
    if joint_pos is not None and hasattr(joint_pos, "shape") and len(joint_pos.shape) >= 2:
        return [f"joint_{idx}" for idx in range(int(joint_pos.shape[1]))]
    raise AttributeError("Unable to read robot joint names from env.robot")


def register_ultra_game_yaw_task(task_registry) -> None:
    """Register the ultra_GameYaw task with the local task registry.

    Args:
        task_registry: Legged Lab task registry instance.
    """
    print("[sim2isaac] Importing ultra_game_yaw cfg...", flush=True)
    from legged_lab.envs.ultra_game_yaw.motion_run_cfg import (
        UltraGameYawMotionRunAgentCfg,
        UltraGameYawMotionRunEnvCfg,
    )
    print("[sim2isaac] Importing ultra_game_yaw env class...", flush=True)
    from legged_lab.envs.ultra_game_yaw.ultra_motion_run_env import UltraGameYawMotionRunEnv

    print("[sim2isaac] Registering ultra_game_yaw task...", flush=True)
    task_registry.register(
        DEFAULT_TASK,
        UltraGameYawMotionRunEnv,
        UltraGameYawMotionRunEnvCfg(),
        UltraGameYawMotionRunAgentCfg(),
    )


def run_sim2isaac(simulation_app, args_cli) -> None:
    """Create the environment and run the policy control loop.

    Args:
        simulation_app: IsaacSim application object from ``AppLauncher``.
        args_cli: Parsed CLI namespace including IsaacLab launcher arguments.
    """
    print("[sim2isaac] Importing runtime modules...", flush=True)
    import torch
    print("[sim2isaac] Imported torch.", flush=True)

    from legged_lab.utils.task_registry import task_registry
    print("[sim2isaac] Imported task registry.", flush=True)
    if args_cli.task != DEFAULT_TASK:
        raise ValueError(f"sim2isaac_joy only supports task {DEFAULT_TASK!r}, got {args_cli.task!r}")
    register_ultra_game_yaw_task(task_registry)
    print(f"[sim2isaac] Registered task: {DEFAULT_TASK}", flush=True)

    print(f"[sim2isaac] Loading task cfg: {args_cli.task}", flush=True)
    env_cfg, _ = task_registry.get_cfgs(args_cli.task)
    configure_deploy_env_cfg(env_cfg, use_implicit_motor=bool(args_cli.use_implicit_motor))
    env_class = task_registry.get_task_class(args_cli.task)
    print("[sim2isaac] Creating IsaacLab env...", flush=True)
    env = env_class(env_cfg, args_cli.headless)
    apply_deploy_initial_sim_state(env)
    print(
        f"[sim2isaac] Applied deterministic initial state: "
        f"base_pos={tuple(round(v, 4) for v in DEPLOY_INIT_BASE_POS)}, "
        f"joints={len(DEPLOY_INIT_JOINT_POS)}",
        flush=True,
    )

    camera_follow_handles = None
    if not args_cli.headless:
        camera_follow_handles = setup_viewport_follow_camera(env, env_cfg.viewer, default_on=True)

    print(f"[sim2isaac] Loading TorchScript policy: {args_cli.policy}", flush=True)
    policy = torch.jit.load(args_cli.policy, map_location=env.device)
    policy.eval()

    gamepad = GamepadCommandSource()
    manual_command = bool(args_cli.use_vel_cmd)
    if args_cli.use_vel_cmd:
        print(f"[sim2isaac] Using predefined velocity commands: {format_vel_cmd_sequence()}", flush=True)
    else:
        if not gamepad.setup():
            print(
                "[sim2isaac] No gamepad and no manual velocity command; robot will keep zero command. "
                "Use --use_vel_cmd to move without a joystick.",
                flush=True,
            )
    recorder = IsaacJointPlotRecorder(enable=args_cli.plot_joints)
    if args_cli.plot_joints:
        print(f"[sim2isaac] Exit plots enabled; PNGs will be saved under: {os.path.abspath(os.path.expanduser(args_cli.plot_dir))}", flush=True)
    else:
        print("[sim2isaac] Exit plots disabled; add --plot_joints if you want PNGs on Ctrl+C.", flush=True)
    step_dt = float(env.step_dt)

    pkl_recorder = MotionPklRecorder(enable=args_cli.record_pkl, fps=args_cli.record_fps, step_dt=step_dt)
    if args_cli.record_pkl:
        print(
            f"[sim2isaac] Trajectory pkl recording enabled: requested_fps={args_cli.record_fps}, "
            f"policy_fps={1.0 / step_dt:.1f}, stride={pkl_recorder.stride}, output_fps={pkl_recorder.output_fps:.1f}.",
            flush=True,
        )

    command = predefined_velocity_command(0.0) if manual_command else np.zeros(3, dtype=np.float64)
    print("[sim2isaac] Reading initial observations...", flush=True)
    obs, _ = refresh_commanded_observations(env, command, fill_history=manual_command)
    expected_actor_dim = getattr(env, "actor_obs_dim", None)
    if expected_actor_dim is not None and int(obs.shape[1]) != int(expected_actor_dim):
        raise RuntimeError(
            f"Policy observation shape check failed: obs dim={obs.shape[1]}, "
            f"env.actor_obs_dim={expected_actor_dim}"
        )

    print("[sim2isaac] Entering policy loop.", flush=True)

    step_count = 0
    debug_every = max(1, int(round(0.5 / step_dt)))  # ~每 0.5s 打印一次
    stop_state = {"requested": False, "count": 0}
    previous_sigint_handler = install_sigint_exit_handler(stop_state)

    try:
        while simulation_app.is_running() and not stop_state["requested"]:
            elapsed_time_s = step_count * step_dt
            if args_cli.duration > 0.0 and elapsed_time_s >= args_cli.duration:
                break

            if manual_command:
                command = predefined_velocity_command(elapsed_time_s)
            else:
                command = gamepad.read_command_target()
            apply_command_to_env(env, command)

            with torch.inference_mode():
                actions = policy(obs)
                if isinstance(actions, tuple):
                    actions = actions[0]
                obs, _, reset_buf, _ = env.step(actions)
            apply_command_to_env(env, command)
            if bool(reset_buf.any().item()):
                apply_deploy_initial_sim_state(env)
                obs, _ = refresh_commanded_observations(env, command, fill_history=manual_command)
            recorder.record(env)
            pkl_recorder.record(env)

            step_count += 1
            if step_count == 1 or step_count % debug_every == 0:
                applied = env.command_generator.command[0].detach().cpu().numpy()
                base_yaw_v = root_yaw_velocity_sample(env.robot.data)
                print(
                    f"[sim2isaac] t={elapsed_time_s:5.1f}s "
                    f"cmd={command.round(2)} applied={applied.round(2)} "
                    f"base_yaw_vel={base_yaw_v.round(2)} ep_len={int(env.episode_length_buf[0].item())}",
                    flush=True,
                )
    except KeyboardInterrupt:
        stop_state["requested"] = True
        print("[sim2isaac] Ctrl+C received; saving plots before exit.", flush=True)
    finally:
        try:
            joint_names = get_robot_joint_names(env)
        except Exception as exc:
            joint_names = []
            print(f"[WARNING] Failed to read joint names for exit plots: {exc}", flush=True)
        try:
            gamepad.close()
        except Exception as exc:
            print(f"[WARNING] Failed to close gamepad cleanly: {exc}", flush=True)
        try:
            show_plots = not stop_state["requested"] and not args_cli.no_show_plots and not args_cli.headless
            print(f"[sim2isaac] Exit plot samples: {recorder.sample_count()}", flush=True)
            saved_paths = recorder.plot_all(joint_names, plot_dir=args_cli.plot_dir, show=show_plots)
            if saved_paths:
                print(f"[sim2isaac] Exit plots saved: {len(saved_paths)} file(s).", flush=True)
        except Exception as exc:
            print(f"[WARNING] Failed to create exit plots: {exc}", flush=True)
        try:
            if args_cli.record_pkl:
                output_file = args_cli.record_pkl_file
                if not output_file:
                    output_file = os.path.join(
                        DEFAULT_RECORD_DIR, datetime.now().strftime("%Y%m%d_%H%M%S") + "_record.pkl"
                    )
                elif not output_file.endswith(".pkl"):
                    output_file += ".pkl"
                print(f"[sim2isaac] Trajectory pkl frames: {pkl_recorder.sample_count()}", flush=True)
                pkl_recorder.save(output_file)
        except Exception as exc:
            print(f"[WARNING] Failed to save trajectory pkl: {exc}", flush=True)
        finally:
            signal.signal(signal.SIGINT, previous_sigint_handler)
            env.close()


def main() -> None:
    """Launch IsaacSim and run the sim2isaac joystick entry point."""
    args_cli = parse_args()
    if not os.path.isfile(args_cli.policy):
        print(f"[ERROR] Policy file not found: {args_cli.policy}")
        raise SystemExit(1)

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    try:
        print("[sim2isaac] Simulation app launched.", flush=True)
        run_sim2isaac(simulation_app, args_cli)
    finally:
        print("[sim2isaac] Closing simulation app.", flush=True)
        simulation_app.close()


if __name__ == "__main__":
    main()
