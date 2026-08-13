"""LCM publisher (optional dependency).

This module publishes robot/mujoco state over LCM for external tools
(e.g. lcm-spy, custom debug GUIs). LCM is a sim2real-time helper and is
NOT required for training or for running play.py inside Isaac Sim.

If `lcm` and/or `isaaclab_lcm` are not installed, all publish_* functions
silently degrade to no-ops so callers don't need to guard each call site.

To enable LCM publishing:
    sudo apt install liblcm-dev      # or: conda install -c conda-forge lcm
    pip install lcm
    pip install -e packages/isaaclab_lcm   # 本地包，不在 PyPI
"""

import time
import warnings

import numpy as np
import torch

try:
    import lcm  # type: ignore

    if not hasattr(lcm, "LCM"):
        lcm_origin = getattr(lcm, "__file__", None) or getattr(lcm, "__path__", "unknown")
        raise AttributeError(
            f"module 'lcm' has no LCM attribute; imported from {lcm_origin}"
        )
    from isaaclab.utils.math import quat_apply_inverse, yaw_quat
    from isaaclab_lcm import robot_state_t  # type: ignore

    _lc = lcm.LCM()
    _LCM_AVAILABLE = True
except (ImportError, AttributeError) as _err:
    warnings.warn(
        f"[lcm_publisher] LCM disabled ({_err}). "
        f"publish_robot_state / publish_mujoco_state will be no-ops. "
        f"Install lcm + isaaclab_lcm to enable broadcasting.",
        stacklevel=2,
    )
    _lc = None
    robot_state_t = None  # type: ignore
    _LCM_AVAILABLE = False


def _quat_apply_wxyz(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate 3D vector by quaternion (w, x, y, z)."""
    w, x, y, z = [float(quat_wxyz[i]) for i in range(4)]
    qv = np.array([x, y, z], dtype=np.float64)
    v = np.asarray(vec, dtype=np.float64)
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(qv, v) * w * 2.0
    c = qv * np.dot(qv, v) * 2.0
    return a + b + c


def _quat_apply_inverse_wxyz(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Inverse rotation (w, x, y, z) applied to vector."""
    w, x, y, z = [float(quat_wxyz[i]) for i in range(4)]
    qv = np.array([x, y, z], dtype=np.float64)
    v = np.asarray(vec, dtype=np.float64)
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(qv, v) * w * 2.0
    c = qv * np.dot(qv, v) * 2.0
    return a - b + c


def _yaw_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    """Yaw-only quaternion (w,x,y,z), same idea as isaaclab.utils.math.yaw_quat."""
    w, x, y, z = [float(quat_wxyz[i]) for i in range(4)]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = 0.5 * yaw
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def mujoco_root_vel_yaw_frame(
    qpos: np.ndarray,
    qvel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Free-joint qvel (world frame) -> yaw frame (matches Isaac play / track_lin_vel_* rewards).

    MuJoCo free joint: qvel[0:3] linear vel, qvel[3:6] angular vel, both in world frame.
    """
    quat = np.asarray(qpos[3:7], dtype=np.float64)
    lin_world = np.asarray(qvel[0:3], dtype=np.float64)
    ang_world = np.asarray(qvel[3:6], dtype=np.float64)
    yaw_q = _yaw_quat_wxyz(quat)
    lin_yaw = _quat_apply_inverse_wxyz(yaw_q, lin_world)
    ang_yaw = _quat_apply_inverse_wxyz(yaw_q, ang_world)
    return lin_yaw, ang_yaw


def _quat_wxyz_to_rpy(quat_wxyz) -> list[float]:
    """Quaternion (w,x,y,z) -> roll, pitch, yaw (XYZ, rad)."""
    w, x, y, z = float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return [roll, pitch, yaw]


def _root_vel_yaw_frame(robot) -> tuple[list[float], list[float]]:
    """Linear/angular velocity in yaw frame (same convention as track_lin_vel_* rewards)."""
    root_quat = robot.data.root_quat_w[0:1]
    yaw_q = yaw_quat(root_quat)
    lin_vel_yaw = quat_apply_inverse(yaw_q, robot.data.root_lin_vel_w[0:1])[0]
    ang_vel_yaw = quat_apply_inverse(yaw_q, robot.data.root_ang_vel_w[0:1])[0]
    return lin_vel_yaw.cpu().numpy().tolist(), ang_vel_yaw.cpu().numpy().tolist()


def _sim_timestamp_ns(sim_time_s: float | None) -> int:
    """Encode time in ns for LCM; prefer simulation time over wall clock."""
    if sim_time_s is not None:
        return int(float(sim_time_s) * 1e9)
    return int(time.time() * 1e9)


def publish_robot_state(
    robot,
    command=None,
    est_vel=None,
    contact_sensor=None,
    feet_body_ids=None,
    feet_sensor_ids=None,
    sim_time_s: float | None = None,
):
    """Publish robot state on ISAACLAB_ROBOT_STATE.

    Args:
        robot: Isaac Lab articulation (env.robot).
        command: Optional (num_envs, 3) tensor — lin_vel_x, lin_vel_y, ang_vel_z.
        est_vel: Optional estimated linear velocity (3,) for base_lin_vel_est.
        contact_sensor: Optional ContactSensor for foot contact forces.
        feet_body_ids: Optional robot body indices for ankle_pitch links (foot velocity).
        feet_sensor_ids: Optional contact sensor body indices for feet (contact force).
        sim_time_s: Simulation time [s] (e.g. episode_length * step_dt). Defaults to wall clock.
    """
    if not _LCM_AVAILABLE:
        return

    msg = robot_state_t()
    msg.timestamp = _sim_timestamp_ns(sim_time_s)

    msg.base_lin_vel, msg.base_ang_vel = _root_vel_yaw_frame(robot)

    if command is not None:
        cmd = command[0].detach().cpu().numpy().tolist()
        msg.cmd_vel = [float(cmd[0]), float(cmd[1]), float(cmd[2])]
    else:
        msg.cmd_vel = [0.0, 0.0, 0.0]

    msg.base_height = [float(robot.data.root_pose_w[0, 2].cpu().item())]
    msg.base_rpy = _quat_wxyz_to_rpy(robot.data.root_quat_w[0].cpu().numpy())

    joint_pos = robot.data.joint_pos[0].cpu().numpy()
    joint_vel = robot.data.joint_vel[0].cpu().numpy()
    joint_tau = robot.data.applied_torque[0].cpu().numpy()
    msg.num_joints = int(joint_pos.shape[0])
    msg.joint_names = [str(n) for n in robot.data.joint_names]
    msg.joint_pos = joint_pos.tolist()
    msg.joint_vel = joint_vel.tolist()
    msg.joint_torque = joint_tau.tolist()
    msg.joint_vel_limit = robot.data.joint_vel_limits[0].cpu().numpy().tolist()
    msg.joint_effort_limit = robot.data.joint_effort_limits[0].cpu().numpy().tolist()

    if est_vel is not None:
        est = np.asarray(est_vel, dtype=float).reshape(-1)
        msg.base_lin_vel_est = est[:3].tolist() if est.size >= 3 else [float(est[0]), 0.0, 0.0]
    else:
        msg.base_lin_vel_est = [0.0, 0.0, 0.0]

    msg.foot_lin_vel = [0.0] * 6
    msg.foot_contact_force = [0.0] * 6
    if feet_body_ids is not None and len(feet_body_ids) >= 2:
        for i, body_id in enumerate(feet_body_ids[:2]):
            vel = robot.data.body_lin_vel_w[0, body_id].cpu().numpy()
            msg.foot_lin_vel[i * 3 : i * 3 + 3] = vel.tolist()
    if contact_sensor is not None and feet_sensor_ids is not None and len(feet_sensor_ids) >= 2:
        for i, sensor_id in enumerate(feet_sensor_ids[:2]):
            force = contact_sensor.data.net_forces_w[0, sensor_id].cpu().numpy()
            msg.foot_contact_force[i * 3 : i * 3 + 3] = force.tolist()

    _lc.publish("ISAACLAB_ROBOT_STATE", msg.encode())


def publish_mujoco_state(
    mj_data,
    num_action,
    command=None,
    est_vel=None,
    joint_names: list[str] | None = None,
    joint_vel_limits: list[float] | None = None,
    joint_effort_limits: list[float] | None = None,
    sim_time_s: float | None = None,
):
    if not _LCM_AVAILABLE:
        return

    msg = robot_state_t()
    if sim_time_s is None:
        sim_time_s = float(mj_data.time)
    msg.timestamp = _sim_timestamp_ns(sim_time_s)

    lin_yaw, ang_yaw = mujoco_root_vel_yaw_frame(mj_data.qpos, mj_data.qvel)
    msg.base_lin_vel = lin_yaw.astype(float).tolist()
    msg.base_ang_vel = ang_yaw.astype(float).tolist()

    if command is not None:
        cmd = np.asarray(command, dtype=float).reshape(-1)
        msg.cmd_vel = [float(cmd[0]), float(cmd[1]), float(cmd[2])] if cmd.size >= 3 else [0.0, 0.0, 0.0]
    else:
        msg.cmd_vel = [0.0, 0.0, 0.0]

    msg.base_height = [float(mj_data.qpos[2])]
    msg.base_rpy = _quat_wxyz_to_rpy(mj_data.qpos[3:7])

    joint_pos = np.array(mj_data.sensordata[0:num_action], dtype=float)
    joint_vel = np.array(mj_data.sensordata[num_action : 2 * num_action], dtype=float)
    qfrc_act = np.array(mj_data.qfrc_actuator, dtype=float)
    torque_offset = max(0, qfrc_act.shape[0] - num_action)
    joint_tau = qfrc_act[torque_offset : torque_offset + num_action]

    n = int(num_action)
    msg.num_joints = n
    if joint_names is not None and len(joint_names) == n:
        msg.joint_names = [str(name) for name in joint_names]
    else:
        msg.joint_names = [f"joint_{i}" for i in range(n)]
    msg.joint_pos = joint_pos.tolist()
    msg.joint_vel = joint_vel.tolist()
    msg.joint_torque = joint_tau.tolist()
    msg.joint_vel_limit = (
        [float(v) for v in joint_vel_limits]
        if joint_vel_limits is not None and len(joint_vel_limits) == n
        else [26.1] * n
    )
    msg.joint_effort_limit = (
        [float(v) for v in joint_effort_limits]
        if joint_effort_limits is not None and len(joint_effort_limits) == n
        else [400.0] * n
    )

    if est_vel is not None:
        est = np.asarray(est_vel, dtype=float).reshape(-1)
        msg.base_lin_vel_est = est[:3].tolist() if est.size >= 3 else [float(est[0]), 0.0, 0.0]
    else:
        msg.base_lin_vel_est = [0.0, 0.0, 0.0]

    msg.foot_lin_vel = [0.0] * 6
    msg.foot_contact_force = [0.0] * 6

    _lc.publish("MUJOCO_ROBOT_STATE", msg.encode())


def get_him_estimated_lin_vel(runner, obs):
    """HIM encoder yaw-frame linear velocity estimate in m/s (for LCM / debug plots)."""
    policy = runner.alg.policy
    if getattr(policy, "estimator", None) is None:
        return None
    if runner.cfg.get("empirical_normalization", False):
        obs = runner.obs_normalizer(obs)
    with torch.no_grad():
        est_vel, _ = policy.estimator(obs)
    return est_vel[0, :3].detach().cpu().numpy().tolist()


def get_him_est_lin_vel_from_jit(policy, obs_history) -> list[float] | None:
    """TorchScript HIM policy: call ``estimate_lin_vel`` if exported (policy.pt from play)."""
    if not hasattr(policy, "estimate_lin_vel"):
        return None
    with torch.no_grad():
        obs = torch.as_tensor(obs_history, dtype=torch.float32)
        vel = policy.estimate_lin_vel(obs)
    return vel.reshape(-1)[:3].cpu().numpy().tolist()
