# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# This file is for tiangong2plus (20 DoF, 锁腰):
#   - loads legged_lab/assets/tiangong2plus/xml/tiangong2_plus.xml by default
#     (SDK 版 MJCF: 自带 21 motor 链序/ground plane/imu 传感器, 全部 geom
#      无碰撞; 运行时由 load_model_with_overrides 修正: meshdir 绝对化 +
#      注入 lock_waist weld + 打开脚掌 geom 碰撞; 观测直接从 qpos/qvel
#      读取, 不依赖 sensordata)
#   - default policy path -> Exported_policy/policy.pt
#   - actor obs layout matches tiangong2plus_motion_run_env.compute_current_observations
#     (75 dims per step x history 10 = 750 for TorchScript policy input, 50 Hz)
#   - 关节序 = DEPLOY_JOINT_ORDER (部署网络序, 与策略输入/输出一致; 注意与
#     play.py 导出的 isaac_joint_names.txt (USD 内部序) 不同, 本脚本不使用后者)
"""tiangong2plus (天工2.0plus, 20 DoF 锁腰) 多风格策略的 MuJoCo Sim2Sim 运行脚本。

实现参照 sim2sim_yrp03_v2.py：MJCF 注入的 motor actuator 全部改由 Python 显式 PD
驱动（恒在线），并按 tiangong2plus.py CustomDelayedPDActuator 的 T-N 曲线
（X1/X2/Y1/Y2, 取自 URDF effort/velocity limit）做力矩-速度包络限幅；复用
命令输入、HIM 历史观测、--record-data 分析日志、LCM 发布和退出绘图流程。

与 Isaac 训练侧的对齐点（tiangong2plus_motion_run_env / tiangong2plus.py）:
  - 单帧 obs 75 维 = 角速度(3) + 重力投影(3) + 指令(3) + 关节
    pos/vel/上帧 action(各 20, DEPLOY_JOINT_ORDER) + 步态相位占位(6, 恒 0),
    历史 10 帧展平 = 750; 策略 50 Hz (物理 400 Hz x decimation 8, 与训练
    Isaac 物理步长 0.0025 s 一致)。
  - waist 由 MJCF weld equality 锁死, 与锁腰 USD (PhysicsFixedJoint) 等效。
  - 初始姿态: hip_pitch -15° / knee +30° / ankle_pitch -15° / shoulder_pitch 0.5,
    pelvis 高 0.96 m。

运行示例：

Usage:
  # 1) run play.py once; it exports and copies policy.pt -> Exported_policy/policy.pt

  # 2) run sim2sim:
  python legged_lab/scripts/sim2sim/sim2sim_tiangong2plus.py
  python legged_lab/scripts/sim2sim/sim2sim_tiangong2plus.py --policy logs/tiangong2plus_run/<run>/exported/policy.pt
  python legged_lab/scripts/sim2sim/sim2sim_tiangong2plus.py --use_vel_cmd
  python legged_lab/scripts/sim2sim/sim2sim_tiangong2plus.py --use_vel_cmd --plot_joints --no_show_plots
  python legged_lab/scripts/sim2sim/sim2sim_tiangong2plus.py --log-torques joint_torques.csv
  python legged_lab/scripts/sim2sim/sim2sim_tiangong2plus.py --record-data
  # 无策略结构自检 (不加载 policy/viewer, PD 保持默认站姿 3 s):
  python legged_lab/scripts/sim2sim/sim2sim_tiangong2plus.py --self-test
  # 关节统计报告（固定 vx=5 m/s，热身 4s 后按 400 Hz 物理子步记录 2s，退出时出图/表/docx）:
  python legged_lab/scripts/sim2sim/sim2sim_tiangong2plus.py --stat-report --stat-vx 5 --stat-warmup-time 4 --stat-record-time 2

  # 3) LCM 曲线（另开终端，默认 --source mujoco）:
  python legged_lab/scripts/utils/lcm_subscriber.py
  python legged_lab/scripts/utils/lcm_subscriber.py --source isaaclab   # play.py

Controls:
  NumPad 8/2  -> vx +/- 0.5
  NumPad 4/6  -> vy -/+ 0.2
  NumPad 7/9  -> yaw +/- 0.2
  0           -> zero command
  BACKSPACE   -> reset simulation
  ESC         -> quit
"""

from __future__ import annotations

import argparse
import csv
from collections import deque
from datetime import datetime
import os
import sys
import time
import xml.etree.ElementTree as ET

import glfw
import mujoco
import mujoco_viewer
import numpy as np
import torch
from pynput import keyboard

try:
    import pygame

    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# --- MuJoCo 3.x 兼容 shim ---
# mujoco-python-viewer 0.1.4（PyPI 最新版，已停更）仍按 MuJoCo 2.x 给
# mjv_moveCamera 多传一个 scn 参数；3.x 已删除该参数，鼠标拖动/滚轮会 TypeError。
# 在创建 viewer 前包装掉多余参数即可。注意 mjv_movePerturb 在 3.x 仍保留 scn，
# 无需处理。本 shim 只影响本进程，不改动 conda 环境。
if int(mujoco.__version__.split(".")[0]) >= 3:
    _mjv_move_camera_orig = mujoco.mjv_moveCamera

    def _mjv_move_camera_compat(m, action, reldx, reldy, scn, cam):
        _mjv_move_camera_orig(m, action, reldx, reldy, cam)

    mujoco.mjv_moveCamera = _mjv_move_camera_compat

LEGGED_LAB_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
if LEGGED_LAB_ROOT_DIR not in sys.path:
    sys.path.insert(0, LEGGED_LAB_ROOT_DIR)

from legged_lab.scripts.lcm_publisher import (  # noqa: E402
    get_him_est_lin_vel_from_jit,
    mujoco_root_vel_yaw_frame,
    publish_mujoco_state,
)


DEFAULT_VEL_CMD_SEQUENCE = (
    (500.0, (5.0, 0.0, 0.0)),
)
DEFAULT_VEL_CMD_SEQUENCE_HELP = " -> ".join(
    f"({cmd[0]:g}, {cmd[1]:g}, {cmd[2]:g}) m/s for {duration:g}s"
    for duration, cmd in DEFAULT_VEL_CMD_SEQUENCE
)
DEFAULT_PLOT_DIR = os.path.join(LEGGED_LAB_ROOT_DIR, "logs", "sim2sim_tiangong2plus_plots")
DEFAULT_RECORD_DIR = os.path.join(LEGGED_LAB_ROOT_DIR, "logs", "sim2sim_tiangong2plus_records")
DEFAULT_STAT_REPORT_DIR = os.path.join(LEGGED_LAB_ROOT_DIR, "logs", "sim2sim_tiangong2plus_stat_report")
# 统计记录窗口内的防呆：基座低于该高度判定为摔倒，终止统计
STAT_FALL_BASE_HEIGHT = 0.5
MAX_CMD = np.array([15.0, 0.5, 0.5], dtype=np.float64)  # curriculum_max_lin_vel_x=15
MAX_CMD_VX_NEG = 0.5
CMD_DEADZONE = 0.05
GAIT_WINDOW_S = 5.0  # 步频/步幅滑动平均窗口（秒）

# --- DEPLOY_JOINT_ORDER (与 tiangong2plus_motion_run_env.py 完全一致) ---
# 策略输入/输出关节序 = 本表; MJCF motor 为链序 (经 deploy2ctrl 映射)。
DEPLOY_JOINT_ORDER = (
    "hip_roll_l_joint", "hip_roll_r_joint",
    "shoulder_pitch_l_joint", "shoulder_pitch_r_joint",
    "hip_pitch_l_joint", "hip_pitch_r_joint",
    "shoulder_roll_l_joint", "shoulder_roll_r_joint",
    "hip_yaw_l_joint", "hip_yaw_r_joint",
    "shoulder_yaw_l_joint", "shoulder_yaw_r_joint",
    "knee_pitch_l_joint", "knee_pitch_r_joint",
    "elbow_pitch_l_joint", "elbow_pitch_r_joint",
    "ankle_pitch_l_joint", "ankle_pitch_r_joint",
    "ankle_roll_l_joint", "ankle_roll_r_joint",
)
# 训练 env 中不受控、由 weld/fixed 锁死的腰
LOCKED_WAIST_JOINT = "waist_yaw_joint"


def parse_args() -> argparse.Namespace:
    default_policy = os.path.join(LEGGED_LAB_ROOT_DIR, "Exported_policy", "policy.pt")
    default_model = os.path.join(LEGGED_LAB_ROOT_DIR, "legged_lab/assets/tiangong2plus/xml/tiangong2_plus.xml")

    parser = argparse.ArgumentParser(description="Run sim2sim for the tiangong2plus (20 DoF lock-waist) multi-AMP policy in MuJoCo (explicit-PD motor version).")
    parser.add_argument("--policy", type=str, default=default_policy, help="Path to TorchScript-exported policy.pt (from play.py).")
    parser.add_argument("--model", type=str, default=default_model, help="Path to the tiangong2plus MJCF (actuators/sensors/floor injected at load time).")
    parser.add_argument("--duration", type=float, default=200.0, help="Sim seconds.")
    parser.add_argument("--log-torques", type=str, default=None, metavar="CSV", help="Save base/cmd vel, per-joint q/qd/tau and limits to CSV (MuJoCo actuator order). One row per physics step. Written on exit.")
    parser.add_argument(
        "--record-data",
        nargs="?",
        const="",
        default=None,
        metavar="CSV",
        help=(
            "Record one analysis row per 50 Hz policy step: style/cmd/yaw/error/HIM plus "
            "per-joint action, target, position, velocity and torque. Without a path, writes "
            f"a timestamped CSV under {DEFAULT_RECORD_DIR}."
        ),
    )
    parser.add_argument(
        "--plot_joints",
        action="store_true",
        help="Plot and save all exit diagnostics: base velocity, T-N scatter, joint positions, velocity and torque.",
    )
    parser.add_argument("--plot_dir", type=str, default=DEFAULT_PLOT_DIR, help="Directory to save plot PNGs when --plot_joints is enabled.")
    parser.add_argument("--no_show_plots", action="store_true", help="Save plot PNGs without opening matplotlib windows.")
    parser.add_argument("--use_vel_cmd", "--use-vel-cmd", action="store_true", help=f"Use predefined velocity commands instead of joystick/keyboard input: {DEFAULT_VEL_CMD_SEQUENCE_HELP}.")
    parser.add_argument(
        "--self-test",
        "--self_test",
        action="store_true",
        help="结构自检：不加载 policy/viewer，PD 保持默认站姿跑 3 s，校验模型注入/观测维度/稳定性后退出。",
    )
    parser.add_argument(
        "--stat-report", "--stat_report",
        action="store_true",
        help=(
            "关节统计报告模式：固定 --stat-vx 速度指令，先热身 --stat-warmup-time 秒，"
            "再按 400 Hz 物理子步记录 --stat-record-time 秒的 q/qd/tau，"
            "退出时生成对齐 template.docx 的统计表/图/report.md/report.docx。"
        ),
    )
    parser.add_argument("--stat-vx", type=float, default=5.0, help="统计工况的前向速度指令 [m/s]。")
    parser.add_argument("--stat-warmup-time", type=float, default=4.0, help="统计热身时长 [s]（仿真时间）。")
    parser.add_argument("--stat-record-time", type=float, default=2.0, help="统计记录时长 [s]（建议 >=2 个步态周期）。")
    parser.add_argument("--stat-output-dir", type=str, default=None, help=f"统计报告输出目录；默认 {DEFAULT_STAT_REPORT_DIR}/<时间戳>。")
    args, unknown = parser.parse_known_args()
    if unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    return args


# ---------------------------------------------------------------------------
# Joint order: policy IO 序 = DEPLOY_JOINT_ORDER; MJCF motor 链序经 deploy2ctrl 映射,
# 与 tiangong2plus_motion_run_env.DEPLOY_JOINT_ORDER 一致。waist_yaw 不在其中
# (锁腰), MJCF 里由 weld equality 约束 pelvis <-> waist_yaw_link。
# ---------------------------------------------------------------------------
MUJOCO_JOINT_NAMES = list(DEPLOY_JOINT_ORDER)

# Isaac velocity_limit_sim (rad/s) from assets/tiangong2plus/tiangong2plus.py
# (tiangong2_plus_urdf.urdf <limit> 原值) — MJCF 无 joint vel limits。
JOINT_VELOCITY_LIMIT_RAD_S: dict[str, float] = {
    "hip_yaw_l_joint": 16.755160819145562,
    "hip_yaw_r_joint": 16.755160819145562,
    "hip_roll_l_joint": 16.755160819145562,
    "hip_roll_r_joint": 16.755160819145562,
    "hip_pitch_l_joint": 15.707963267948966,
    "hip_pitch_r_joint": 15.707963267948966,
    "knee_pitch_l_joint": 15.707963267948966,
    "knee_pitch_r_joint": 15.707963267948966,
    "ankle_pitch_l_joint": 14.137166941154069,
    "ankle_pitch_r_joint": 14.137166941154069,
    "ankle_roll_l_joint": 14.137166941154069,
    "ankle_roll_r_joint": 14.137166941154069,
    "shoulder_pitch_l_joint": 14.137166941154069,
    "shoulder_pitch_r_joint": 14.137166941154069,
    "shoulder_roll_l_joint": 14.137166941154069,
    "shoulder_roll_r_joint": 14.137166941154069,
    "shoulder_yaw_l_joint": 14.137166941154069,
    "shoulder_yaw_r_joint": 14.137166941154069,
    "elbow_pitch_l_joint": 14.137166941154069,
    "elbow_pitch_r_joint": 14.137166941154069,
}

# PD gains from tiangong2plus.py (stiffness/damping of CustomDelayedPDActuatorCfg)。
JOINT_TYPE_PD_PARAMS: dict[str, dict[str, float]] = {
    "hip_yaw": {"kp": 500.0, "kd": 10.0},
    "hip_roll": {"kp": 700.0, "kd": 20.0},
    "hip_pitch": {"kp": 700.0, "kd": 10.0},
    "knee_pitch": {"kp": 700.0, "kd": 10.0},
    "ankle_pitch": {"kp": 100.0, "kd": 6.0},
    "ankle_roll": {"kp": 40.0, "kd": 2.0},
    "shoulder_pitch": {"kp": 50.0, "kd": 3.0},
    "shoulder_roll": {"kp": 50.0, "kd": 3.0},
    "shoulder_yaw": {"kp": 50.0, "kd": 3.0},
    "elbow_pitch": {"kp": 50.0, "kd": 3.0},
}

# T-N 曲线参数（X1/X2/Y1/Y2）from tiangong2plus.py：X2 = velocity_limit_sim，
# Y2 = effort_limit_sim（URDF 原值）。注意 hip_pitch/knee/ankle 的 X1>X2 与
# 训练侧 CustomDelayedPDActuator 完全一致（tunable baseline，暂未整定）。
JOINT_TYPE_TN_PARAMS: dict[str, dict[str, float]] = {
    "hip_yaw": {"X1": 10.0, "X2": 16.755160819145562, "Y1": 150.0, "Y2": 235.0},
    "hip_roll": {"X1": 12.0, "X2": 16.755160819145562, "Y1": 150.0, "Y2": 235.0},
    "hip_pitch": {"X1": 17.0, "X2": 15.707963267948966, "Y1": 600.0, "Y2": 330.0},
    "knee_pitch": {"X1": 17.0, "X2": 15.707963267948966, "Y1": 600.0, "Y2": 330.0},
    "ankle_pitch": {"X1": 17.0, "X2": 14.137166941154069, "Y1": 55.0, "Y2": 55.0},
    "ankle_roll": {"X1": 17.0, "X2": 14.137166941154069, "Y1": 55.0, "Y2": 55.0},
    "shoulder_pitch": {"X1": 13.0, "X2": 14.137166941154069, "Y1": 55.0, "Y2": 55.0},
    "shoulder_roll": {"X1": 13.0, "X2": 14.137166941154069, "Y1": 55.0, "Y2": 55.0},
    "shoulder_yaw": {"X1": 13.0, "X2": 14.137166941154069, "Y1": 55.0, "Y2": 55.0},
    "elbow_pitch": {"X1": 13.0, "X2": 14.137166941154069, "Y1": 55.0, "Y2": 55.0},
}

# 关节 armature（转子惯量）from tiangong2plus.py（沿用 yrp03_v2 基线）；
# MJCF 未带 armature，运行时由 _init_explicit_actuator 注入 loaded model。
JOINT_TYPE_ARMATURE: dict[str, float] = {
    "hip_yaw": 0.048865,
    "hip_roll": 0.40318,
    "hip_pitch": 0.40318,
    "knee_pitch": 0.40318,
    "ankle_pitch": 0.21448,
    "ankle_roll": 0.1,
    "shoulder_pitch": 0.041125,
    "shoulder_roll": 0.041125,
    "shoulder_yaw": 0.041125,
    "elbow_pitch": 0.041125,
}

JOINT_TO_TYPE: dict[str, str] = {
    "hip_yaw_l_joint": "hip_yaw",
    "hip_yaw_r_joint": "hip_yaw",
    "hip_roll_l_joint": "hip_roll",
    "hip_roll_r_joint": "hip_roll",
    "hip_pitch_l_joint": "hip_pitch",
    "hip_pitch_r_joint": "hip_pitch",
    "knee_pitch_l_joint": "knee_pitch",
    "knee_pitch_r_joint": "knee_pitch",
    "ankle_pitch_l_joint": "ankle_pitch",
    "ankle_pitch_r_joint": "ankle_pitch",
    "ankle_roll_l_joint": "ankle_roll",
    "ankle_roll_r_joint": "ankle_roll",
    "shoulder_pitch_l_joint": "shoulder_pitch",
    "shoulder_pitch_r_joint": "shoulder_pitch",
    "shoulder_roll_l_joint": "shoulder_roll",
    "shoulder_roll_r_joint": "shoulder_roll",
    "shoulder_yaw_l_joint": "shoulder_yaw",
    "shoulder_yaw_r_joint": "shoulder_yaw",
    "elbow_pitch_l_joint": "elbow_pitch",
    "elbow_pitch_r_joint": "elbow_pitch",
}


def build_actuator_params_from_joint_types() -> dict[str, dict[str, float]]:
    return {
        joint_name: {
            **JOINT_TYPE_PD_PARAMS[joint_type],
            **JOINT_TYPE_TN_PARAMS[joint_type],
            "armature": JOINT_TYPE_ARMATURE[joint_type],
            "velocity_limit": JOINT_VELOCITY_LIMIT_RAD_S[joint_name],
        }
        for joint_name, joint_type in JOINT_TO_TYPE.items()
    }


ACTUATOR_PARAMS: dict[str, dict[str, float]] = build_actuator_params_from_joint_types()


def get_tn_plot_joint_names(mujoco_joint_names: list[str]) -> list[str]:
    return [name for name in mujoco_joint_names if name in ACTUATOR_PARAMS]


def get_tn_curve_params(joint_name: str) -> tuple[float, float, float, float]:
    params = ACTUATOR_PARAMS[joint_name]
    return params["X1"], params["X2"], params["Y1"], params["Y2"]


def build_stat_report_params(joint_names: list[str]) -> dict[str, np.ndarray]:
    """按给定关节顺序组装 legged_lab.utils.stat_report 需要的每关节参数。"""
    keys = ("kp", "kd", "effort_limit", "velocity_limit", "y1", "y2", "x1", "x2")
    params = {k: np.zeros(len(joint_names)) for k in keys}
    for i, name in enumerate(joint_names):
        p = ACTUATOR_PARAMS[name]
        params["kp"][i], params["kd"][i] = p["kp"], p["kd"]
        params["effort_limit"][i] = p["Y2"]
        params["velocity_limit"][i] = p["velocity_limit"]
        params["y1"][i], params["y2"][i] = p["Y1"], p["Y2"]
        params["x1"][i], params["x2"][i] = p["X1"], p["X2"]
    return params


def load_model_with_overrides(model_path: str) -> mujoco.MjModel:
    """tiangong2_plus.xml 运行时修正（不改动磁盘文件）:
      1. compiler meshdir -> 绝对路径 (from_xml_string 无基准目录)
      2. 地板: 已有 plane geom (新版 "ground") 则复用, 否则注入
      3. 锁腰: 无 equality weld 则注入 pelvis<->waist_yaw_link weld,
         等效锁腰 USD 的 PhysicsFixedJoint (waist 电机 ctrl 恒 0)
      4. 碰撞: 新版 MJCF 全部 geom contype/conaffinity=0 (SDK 动力学
         模型, 无接触); 只打开 ankle_roll 脚掌 geom 的碰撞 (与地板,
         地板默认 ct1/ca1)。无 knee<->ankle 凸包互嵌问题。
      5. actuator: 新版 MJCF 自带 21 个 motor (含 waist_yaw, 链序,
         非 DEPLOY 序), 保持原顺序, 由 _init_explicit_actuator 建映射;
         若 actuator 为空 (旧裸 MJCF) 则按 DEPLOY_JOINT_ORDER 注入 20 个
    观测所需的角速度/重力投影/关节状态直接从 qpos/qvel 读取, 无需 sensor。
    """
    tree = ET.parse(model_path)
    root = tree.getroot()
    xml_dir = os.path.dirname(os.path.abspath(model_path))

    compiler = root.find("compiler")
    if compiler is None or "meshdir" not in compiler.attrib:
        raise RuntimeError(f"{model_path} 缺少 compiler meshdir, 不是预期的 tiangong2_plus MJCF")
    meshdir = compiler.attrib["meshdir"]
    if not os.path.isabs(meshdir):
        compiler.set("meshdir", os.path.normpath(os.path.join(xml_dir, meshdir)))

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError(f"{model_path} 缺少 worldbody")

    # --- 地板 ---
    plane_geoms = [g for g in worldbody.findall("geom") if g.get("type") == "plane"]
    if plane_geoms:
        print(f"[INFO] MJCF 已有地板 geom: {[g.get('name') for g in plane_geoms]}")
    else:
        ET.SubElement(
            worldbody, "geom",
            {"name": "floor", "type": "plane", "size": "100 100 0.1",
             "rgba": "0.9 0.9 0.9 1", "friction": "1.0 0.1 0.1", "condim": "3"},
        )
        print("[INFO] MJCF 无地板, 已注入 floor plane")

    # --- 锁腰 weld ---
    equality = root.find("equality")
    has_weld = equality is not None and any(
        w.get("body2") == "waist_yaw_link" for w in equality.findall("weld")
    )
    if has_weld:
        print("[INFO] MJCF 已有 waist weld")
    else:
        if equality is None:
            equality = ET.SubElement(root, "equality")
        ET.SubElement(
            equality, "weld",
            {"name": "lock_waist", "body1": "pelvis", "body2": "waist_yaw_link"},
        )
        print("[INFO] MJCF 无 waist weld, 已注入 lock_waist (等效锁腰 USD)")

    # --- 碰撞: 只打开脚掌 geom ---
    # 新版 MJCF 在 ankle_roll link 下自带 foot_sole_l/r 碰撞块 (box), 优先只
    # 开 sole; 没有 sole 的旧 MJCF 则打开该 body 下全部 geom (脚掌网格凸包)。
    n_feet = 0
    for body in worldbody.iter("body"):
        if body.get("name") in ("ankle_roll_l_link", "ankle_roll_r_link"):
            geoms = body.findall("geom")
            soles = [g for g in geoms if (g.get("name") or "").startswith("foot_sole")]
            targets = soles if soles else geoms
            for geom in targets:
                geom.set("contype", "1")
                geom.set("conaffinity", "1")
                n_feet += 1
    print(f"[INFO] 脚掌碰撞 geom 已开启: {n_feet} 个 (其余 geom 保持无碰撞)")

    # --- actuator: 新版自带 21 motor 则保持; 旧裸 MJCF 注入 20 ---
    actuator = root.find("actuator")
    if actuator is not None and len(actuator) > 0:
        print(f"[INFO] MJCF 自带 {len(actuator)} 个 actuator, 保持原顺序")
    else:
        if actuator is None:
            actuator = ET.SubElement(root, "actuator")
        for name in DEPLOY_JOINT_ORDER:
            y2 = ACTUATOR_PARAMS[name]["Y2"]
            ET.SubElement(
                actuator, "motor",
                {"name": name, "joint": name, "ctrlrange": f"{-y2} {y2}", "ctrllimited": "true"},
            )
        print(f"[INFO] MJCF 无 actuator, 已注入 {len(DEPLOY_JOINT_ORDER)} 个 motor (DEPLOY_JOINT_ORDER)")

    xml_string = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml_string)
    return model


def predefined_velocity_command(elapsed_time_s: float) -> np.ndarray:
    elapsed_time_s = max(0.0, elapsed_time_s)
    segment_start_s = 0.0
    for duration_s, command in DEFAULT_VEL_CMD_SEQUENCE:
        segment_end_s = segment_start_s + duration_s
        if elapsed_time_s < segment_end_s:
            return np.asarray(command, dtype=np.float64)
        segment_start_s = segment_end_s
    return np.zeros(3, dtype=np.float64)


# Style hysteresis (aligned with motion_run_cfg.py StyleFromCommandCfg)
STAND_ENTER = 0.05
STAND_EXIT = 0.08
RUN_ENTER = 1.0
RUN_EXIT = 0.85

# obs_scales.commands from motion_run_cfg.py NormalizationCfg
COMMAND_OBS_SCALE = np.array([0.2, 1.0, 1.0], dtype=np.float64)

GAMEPAD_PROFILES = {
    "xbox": (0, 1, 3),
    "betop": (0, 1, 2),
    "ps": (0, 1, 2),
}


def _detect_gamepad_profile(name: str) -> str:
    name_lower = name.lower()
    if any(k in name_lower for k in ("betop", "北通", "bt", "gamepad")):
        return "betop"
    if any(k in name_lower for k in ("xbox", "xinput", "045e")):
        return "xbox"
    if any(k in name_lower for k in ("sony", "dualshock", "dualsense", "ps4", "ps5", "054c")):
        return "ps"
    return "xbox"


def _apply_deadzone(val: float, threshold: float) -> float:
    if abs(val) < threshold:
        return 0.0
    sign = 1.0 if val >= 0.0 else -1.0
    return sign * (abs(val) - threshold) / (1.0 - threshold)


def map_gamepad_axes_to_command(left_x: float, left_y: float, right_x: float) -> np.ndarray:
    vx_axis = _apply_deadzone(left_y, CMD_DEADZONE)
    vy_axis = _apply_deadzone(left_x, CMD_DEADZONE)
    yaw_axis = _apply_deadzone(right_x, CMD_DEADZONE)
    vx = vx_axis * MAX_CMD[0] if vx_axis >= 0.0 else vx_axis * MAX_CMD_VX_NEG
    return np.array([vx, vy_axis * MAX_CMD[1], yaw_axis * MAX_CMD[2]], dtype=np.float64)


def format_vel_cmd_sequence(sequence=DEFAULT_VEL_CMD_SEQUENCE) -> str:
    return " -> ".join(
        f"({cmd[0]:g}, {cmd[1]:g}, {cmd[2]:g}) m/s for {duration:g}s"
        for duration, cmd in sequence
    )


def resolve_record_data_path(path: str | None) -> str | None:
    if path is None:
        return None
    if path.strip():
        resolved = os.path.abspath(os.path.expanduser(path))
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        resolved = os.path.join(DEFAULT_RECORD_DIR, f"tiangong2plus_analysis_{timestamp}.csv")
    if os.path.splitext(resolved)[1].lower() != ".csv":
        raise ValueError(f"--record-data output must use a .csv extension: {resolved}")
    return resolved


def _joint_pos_subplot_shape(num_joints: int) -> tuple[int, int]:
    if num_joints <= 1:
        return max(1, num_joints), 1
    return (num_joints + 1) // 2, 2


def configure_matplotlib_backend(show: bool) -> None:
    if not show:
        import matplotlib

        matplotlib.use("Agg", force=True)


def prepare_plot_output_dir(plot_dir: str | None) -> str | None:
    if plot_dir is None or str(plot_dir).strip() == "":
        return None
    out_dir = os.path.abspath(os.path.expanduser(plot_dir))
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_plot_figure(fig, save_dir: str | None, filename: str) -> str | None:
    if save_dir is None:
        return None
    out_path = os.path.join(save_dir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[sim2sim] Saved plot: {out_path}", flush=True)
    return out_path


def tn_torque_limits_for_velocity(joint_name: str, joint_vel: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if joint_name not in ACTUATOR_PARAMS:
        return None
    x1, x2, y1, y2 = get_tn_curve_params(joint_name)
    vel = np.asarray(joint_vel, dtype=np.float64)
    abs_vel = np.abs(vel)

    def _lim(y: float) -> np.ndarray:
        return np.where(abs_vel < x1, y, np.clip(-y / (x2 - x1) * (abs_vel - x1) + y, 0.0, None))

    upper = np.where(vel > 0.0, _lim(y1), _lim(y2))
    lower = np.where(vel > 0.0, -_lim(y2), -_lim(y1))
    return upper, lower


def set_mujoco_integrator(model, integrator: str) -> None:
    integrators = {
        "euler": mujoco.mjtIntegrator.mjINT_EULER,
        "rk4": mujoco.mjtIntegrator.mjINT_RK4,
        "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
        "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
    }
    key = integrator.lower()
    if key not in integrators:
        valid = ", ".join(sorted(integrators))
        raise ValueError(f"Unknown MuJoCo integrator {integrator!r}. Valid values: {valid}")
    model.opt.integrator = integrators[key]


class SimToSimCfg:
    """Configuration. MUST match tiangong2plus/motion_run_cfg.py and its environment."""

    class sim:
        sim_duration = 100.0
        num_action = 20
        num_styles = 3
        # 3+3+3 + 20+20+20 + 6 gait pad (style 由 |cmd_x| 内部推断, 不进 obs;
        # 与 play/export actor 一致: 75 = 9 + 20*3 + 6)
        num_obs_per_step = 75
        actor_obs_history_length = 10
        # MuJoCo 物理子步与训练 Isaac 一致 (0.0025 s, MJCF <option> 同为
        # implicitfast); 0.0025*8 = 0.02 s 策略周期 (50 Hz) 与部署推理一致。
        dt = 0.0025
        integrator = "implicitfast"  # "euler" "rk4" "implicit" implicitfast
        decimation = 8  # MuJoCo 400 Hz, policy 50 Hz
        clip_observations = 100.0
        clip_actions = 100.0
        action_scale = 0.25

    class robot:
        # tiangong2plus.py InitialStateCfg.pos[2]: 站立 pelvis 高 0.96
        base_height_init: float = 0.96

    class normalization:
        clip_observations = 100.0

        class obs_scales:
            lin_vel = 0.2
            ang_vel = 0.5
            projected_gravity = 1.0
            joint_pos = 1.0
            joint_vel = 0.05
            actions = 1.0
            height_scan = 1.0

    class debug:
        print_velocity = True
        print_every_n_policy_steps = 20


class MujocoRunner:
    """MuJoCo sim2sim driver for the tiangong2plus (20 DoF lock-waist) multi-AMP policy."""

    def __init__(
        self,
        cfg: SimToSimCfg,
        policy_path: str | None,
        model_path: str,
        torque_log_path: str | None = None,
        record_data_path: str | None = None,
        plot_joints: bool = False,
        plot_dir: str | None = None,
        show_plots: bool = True,
        use_vel_cmd: bool = False,
        stat_report: bool = False,
        stat_vx: float = 5.0,
        stat_warmup_time: float = 4.0,
        stat_record_time: float = 2.0,
        stat_output_dir: str | None = None,
        headless: bool = False,
    ):
        self.cfg = cfg
        self.headless = headless
        self.plot_joints = plot_joints
        self.plot_dir = plot_dir
        self.show_plots = show_plots
        self.use_vel_cmd = use_vel_cmd
        # --- 关节统计报告模式（400 Hz 物理子步采样） ---
        self.stat_report = stat_report
        self.stat_vx = float(stat_vx)
        self.stat_warmup_time = float(stat_warmup_time)
        self.stat_record_time = float(stat_record_time)
        self.stat_output_dir = stat_output_dir
        self.policy_path = policy_path
        self._stat_t0: float | None = None
        self._stat_done = False
        self._stat_log: dict[str, list] = {"t": [], "q": [], "qd": [], "tau": [], "vx": []}
        self._plot_time: list[float] = []
        self._plot_base_yaw_vel: list[np.ndarray] = []
        self._plot_joint_pos: list[np.ndarray] = []
        self._plot_joint_vel: list[np.ndarray] = []
        self._plot_joint_tau: list[np.ndarray] = []

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"MJCF 不存在: {model_path}")
        self.model = load_model_with_overrides(model_path)
        self.model.opt.timestep = self.cfg.sim.dt
        set_mujoco_integrator(self.model, self.cfg.sim.integrator)
        print(
            f"[INFO] MuJoCo option override: timestep={self.cfg.sim.dt}, "
            f"integrator={self.cfg.sim.integrator}"
        )

        self.policy = None
        self._him_est_vel = False
        if policy_path is not None:
            if not os.path.isfile(policy_path):
                raise FileNotFoundError(f"Policy 不存在: {policy_path}")
            self.policy = torch.jit.load(policy_path)
            self._him_est_vel = hasattr(self.policy, "estimate_lin_vel")
            if self._him_est_vel:
                print("[INFO] HIM policy: LCM will publish base_lin_vel_est via estimate_lin_vel().")
            else:
                print("[WARN] policy.pt has no estimate_lin_vel(); re-run play.py to refresh Exported_policy.")

        self.data = mujoco.MjData(self.model)
        if headless:
            self.viewer = None
        else:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
            self.viewer._render_every_frame = False
        self.listener = None
        self.joystick = None
        self.gamepad_profile = "xbox"
        self.exit_flag = False
        self.torque_log_path = torque_log_path
        self._torque_log: list[list[float]] = []
        self.record_data_path = record_data_path
        self._analysis_log: list[list[float | int | str]] = []
        self._analysis_step = 0
        self._episode_id = -1

        self.init_variables()
        self._init_explicit_actuator()
        if self.torque_log_path is not None or self.plot_joints:
            self._init_joint_log_limits()
        if self.torque_log_path is not None:
            print(
                f"[INFO] Joint state logging enabled -> {self.torque_log_path} "
                f"(yaw-frame base/cmd vel + q, qd, tau + limits; MuJoCo actuator order)"
            )
        if self.record_data_path is not None:
            print(
                f"[INFO] 50 Hz analysis recording enabled -> {self.record_data_path} "
                "(style/cmd/yaw/error/HIM + per-joint action/q_des/q/qd/tau)"
            )
        if self.plot_joints:
            print(
                f"[INFO] Exit plots enabled; PNGs will be saved under: "
                f"{os.path.abspath(os.path.expanduser(self.plot_dir)) if self.plot_dir else '(disabled)'}"
            )
        self.obs_scales = self.cfg.normalization.obs_scales
        self.clip_obs = self.cfg.normalization.clip_observations
        print("[INFO] Observation noise disabled")
        self.reset_simulation()
        if self.viewer is not None:
            # One-time default orbit; do not overwrite every frame so mouse can orbit/zoom.
            self._apply_initial_viewer_camera()

    # ------------------------------------------------------------------
    # 状态读取: MJCF 无 sensor, 直接从 qpos/qvel 经关节地址取数
    # ------------------------------------------------------------------
    def _joint_q(self) -> np.ndarray:
        return np.array(self.data.qpos[self._exp_qpos_adr], dtype=np.float64)

    def _joint_qd(self) -> np.ndarray:
        return np.array(self.data.qvel[self._exp_dof_adr], dtype=np.float64)

    def _base_quat_wxyz(self) -> np.ndarray:
        return np.array(self.data.qpos[3:7], dtype=np.float64)  # (w, x, y, z)

    def _base_ang_vel_body(self) -> np.ndarray:
        """pelvis 质心角速度转到机体系 (与 Isaac root_ang_vel_b 同口径)。"""
        quat_xyzw = self._base_quat_wxyz()[[1, 2, 3, 0]]
        return self.quat_rotate_inverse(quat_xyzw, np.array(self.data.qvel[3:6], dtype=np.float64))

    def init_variables(self) -> None:
        self.dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.dof_pos = np.zeros(self.cfg.sim.num_action)
        self.dof_vel = np.zeros(self.cfg.sim.num_action)
        self.action = np.zeros(self.cfg.sim.num_action)

        # Default DOF pos (DEPLOY_JOINT_ORDER). Match tiangong2plus.py InitialStateCfg.joint_pos:
        # hip_pitch -15° / knee +30° / ankle_pitch -15° / shoulder_pitch 0.5, 其余 0。
        d2r = np.deg2rad
        self.default_dof_pos = np.array(
            [
                0.0,  # hip_roll_l_joint
                0.0,  # hip_roll_r_joint
                0.5,  # shoulder_pitch_l_joint
                0.5,  # shoulder_pitch_r_joint
                d2r(-15.0),  # hip_pitch_l_joint
                d2r(-15.0),  # hip_pitch_r_joint
                0.0,  # shoulder_roll_l_joint
                0.0,  # shoulder_roll_r_joint
                0.0,  # hip_yaw_l_joint
                0.0,  # hip_yaw_r_joint
                0.0,  # shoulder_yaw_l_joint
                0.0,  # shoulder_yaw_r_joint
                d2r(30.0),  # knee_pitch_l_joint
                d2r(30.0),  # knee_pitch_r_joint
                0.0,  # elbow_pitch_l_joint
                0.0,  # elbow_pitch_r_joint
                d2r(-15.0),  # ankle_pitch_l_joint
                d2r(-15.0),  # ankle_pitch_r_joint
                0.0,  # ankle_roll_l_joint
                0.0,  # ankle_roll_r_joint
            ]
        )

        self.mujoco_joint_names = MUJOCO_JOINT_NAMES
        # 策略输入/输出序 = DEPLOY_JOINT_ORDER (MJCF motor 链序经 deploy2ctrl 映射)
        self.mujoco_to_isaac_idx = list(range(len(self.mujoco_joint_names)))
        self.isaac_to_mujoco_idx = list(range(len(self.mujoco_joint_names)))
        self._joint_vel_limits_lcm = [
            JOINT_VELOCITY_LIMIT_RAD_S[name] for name in self.mujoco_joint_names
        ]
        # LCM 限幅线使用 Isaac T-N 曲线的 Y2（effort_limit_sim），与 Python 限幅一致。
        self._joint_effort_limits_lcm = [
            ACTUATOR_PARAMS[name]["Y2"] for name in self.mujoco_joint_names
        ]
        self._init_foot_geom_ids()
        self._print_joint_mapping()

        self.episode_length_buf = 0

        self.command_vel = np.array([0.0, 0.0, 0.0])
        self.command_target_vel = np.array([0.0, 0.0, 0.0])
        self.keyboard_command_target_vel = np.array([0.0, 0.0, 0.0])
        self.obs_history = np.zeros(
            (self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length,),
            dtype=np.float32,
        )
        self._obs_history_initialized = False
        self.num_styles = self.cfg.sim.num_styles
        self.style_ids = 0
        self.style_onehot = np.zeros(self.num_styles, dtype=np.float32)
        self.style_onehot[self.style_ids] = 1.0

        # Gait tracking（与训练 env 同口径，但取最近 GAIT_WINDOW_S 秒的滑动平均）：
        # cadence = 落地次数/run 时长，stride = 2*净前移/落地次数。
        self._gait_window_len = max(1, int(round(GAIT_WINDOW_S / self.dt)))
        self._gait_window = deque(maxlen=self._gait_window_len)
        self._gait_step_count = 0
        self._gait_forward_dist = 0.0
        self._gait_run_steps = 0
        self._gait_contact_prev = False

    def _init_foot_geom_ids(self) -> None:
        """脚踝 body (ankle_roll_l/r_link) 上的全部 geom 作为着地检测单元
        (tiangong2plus MJCF 无独立 toe geom, 脚掌网格在 ankle_roll link 上);
        同时缓存地板 plane geom id。"""
        foot_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ("ankle_roll_l_link", "ankle_roll_r_link")
        ]
        foot_body_ids = [bid for bid in foot_body_ids if bid >= 0]
        if not foot_body_ids:
            raise RuntimeError("MJCF 里没找到 ankle_roll_l/r_link body，着地检测不可用")
        self._gait_foot_geom_ids = [
            gid for gid in range(self.model.ngeom) if int(self.model.geom_bodyid[gid]) in foot_body_ids
        ]
        plane_ids = np.where(self.model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)[0]
        if len(plane_ids) == 0:
            raise RuntimeError("MJCF 里没有 plane geom (地板)")
        self._floor_geom_id = int(plane_ids[0])
        print(
            f"[INFO] 着地检测 geoms: "
            f"{[mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or f'geom{g}' for g in self._gait_foot_geom_ids]}, "
            f"地板 geom: {mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, self._floor_geom_id) or self._floor_geom_id}"
        )

    def _print_joint_mapping(self) -> None:
        print("=" * 72)
        print("Joint-order sanity check (DEPLOY_JOINT_ORDER; MJCF ctrl 序经 deploy2ctrl 映射)")
        print("=" * 72)
        for idx, name in enumerate(self.mujoco_joint_names):
            print(f"{idx:>6} {name}")

    _VEL_HEADER_EVERY_N_LINES = 20
    _vel_print_line_count = 0

    def _foot_contact_any(self) -> bool:
        """任一脚踝 geom 与地板接触（左脚或右脚着地）。"""
        foot_ids = self._gait_foot_geom_ids
        floor_id = self._floor_geom_id
        for contact in self.data.contact[: self.data.ncon]:
            if contact.geom1 in foot_ids and contact.geom2 == floor_id:
                return True
            if contact.geom2 in foot_ids and contact.geom1 == floor_id:
                return True
        return False

    def _update_gait_metrics(self) -> None:
        """每个 policy step（50 Hz）更新一次：维护最近 GAIT_WINDOW_S 秒的滑动窗口。

        与训练 env 的 Gait/cadence、Gait/stride_length 同口径，但改为最近 5s 平均：
          cadence = 窗口内落地次数 / 窗口内 run 时长，stride = 2 * 窗口内净前移 / 窗口内落地次数。
        只在 run style（style_id==2）累计，避免 stand/walk 污染。
        """
        contact_any = self._foot_contact_any()
        is_run = self.style_ids == 2
        footfall = False
        forward_inc = 0.0
        if is_run:
            if contact_any and not self._gait_contact_prev:
                footfall = True
            lin_yaw, _ = mujoco_root_vel_yaw_frame(self.data.qpos, self.data.qvel)
            forward_inc = max(float(lin_yaw[0]), 0.0) * self.dt

        # 窗口满时弹出最旧一条，从滚动累计里减掉其贡献
        if len(self._gait_window) == self._gait_window_len:
            old_run, old_footfall, old_inc = self._gait_window[0]
            if old_run:
                self._gait_run_steps -= 1
                if old_footfall:
                    self._gait_step_count -= 1
                self._gait_forward_dist -= old_inc
        self._gait_window.append((is_run, footfall, forward_inc))
        if is_run:
            self._gait_run_steps += 1
            if footfall:
                self._gait_step_count += 1
            self._gait_forward_dist += forward_inc

        self._gait_contact_prev = contact_any

    def _gait_summary(self) -> tuple[float, float]:
        """最近 GAIT_WINDOW_S 秒平均 (cadence [steps/s], stride [m])；步数不足时返回 NaN。"""
        run_time = self._gait_run_steps * self.dt
        cadence = (self._gait_step_count / run_time) if run_time > 0.0 else float("nan")
        stride = (
            (2.0 * self._gait_forward_dist / self._gait_step_count)
            if self._gait_step_count >= 1
            else float("nan")
        )
        return cadence, stride

    def _maybe_print_velocity_status(self) -> None:
        if not getattr(self.cfg.debug, "print_velocity", False):
            return
        every_n = max(1, getattr(self.cfg.debug, "print_every_n_policy_steps", 20))
        if (self.episode_length_buf % every_n) != 0:
            return

        lin_yaw, ang_yaw = mujoco_root_vel_yaw_frame(self.data.qpos, self.data.qvel)
        cmd = self.command_vel
        style_name = {0: "stand", 1: "walk", 2: "run"}.get(self.style_ids, f"s{self.style_ids}")

        err_x = cmd[0] - lin_yaw[0]
        err_y = cmd[1] - lin_yaw[1]
        err_w = cmd[2] - ang_yaw[2]

        cadence, stride = self._gait_summary()
        cad_str = f"{cadence:5.2f}" if np.isfinite(cadence) else "   --"
        stride_str = f"{stride:5.2f}" if np.isfinite(stride) else "   --"

        if self._vel_print_line_count % self._VEL_HEADER_EVERY_N_LINES == 0:
            print(
                "[vel]    t  | style | "
                "cmd[ vx    vy    ωz ] | yaw[ vx    vy    ωz ] | "
                "err[ vx    vy    ωz ] | cad[steps/s] | stride[m]"
            )
        print(
            f"[vel] {self.data.time:5.2f} | "
            f"{style_name:>5s} | "
            f"cmd[{cmd[0]:+5.2f} {cmd[1]:+5.2f} {cmd[2]:+5.2f}] | "
            f"yaw[{lin_yaw[0]:+5.2f} {lin_yaw[1]:+5.2f} {ang_yaw[2]:+5.2f}] | "
            f"err[{err_x:+5.2f} {err_y:+5.2f} {err_w:+5.2f}] | "
            f"cad {cad_str} | stride {stride_str}"
        )
        self._vel_print_line_count += 1

    def _update_style_from_command(self) -> None:
        """Style 滞回（与 tiangong2plus_motion_run_env._update_style_from_command 一致，仅用于日志）。"""
        cmd = self.command_vel
        speed_x = float(np.abs(cmd[0]))
        active_enter = bool(np.any(np.abs(cmd) >= STAND_ENTER))
        active_exit = bool(np.any(np.abs(cmd) >= STAND_EXIT))
        is_stand = bool(np.all(np.abs(cmd) < STAND_ENTER))

        prev = int(self.style_ids)
        new = prev
        if prev == 0:
            if speed_x >= RUN_ENTER:
                new = 2
            elif active_exit:
                new = 1
        elif prev == 2:
            if speed_x < RUN_EXIT:
                new = 1
            if not active_enter:
                new = 0
        else:
            if speed_x >= RUN_ENTER:
                new = 2
            if not active_enter:
                new = 0
        if is_stand:
            new = 0

        self.style_ids = new
        self.style_onehot = np.zeros(self.num_styles, dtype=np.float32)
        self.style_onehot[self.style_ids] = 1.0

    def get_obs(self) -> np.ndarray:
        """单帧 75 维 + 10 帧历史 = 750，与 tiangong2plus_motion_run_env 完全一致。"""
        self.dof_pos = self._joint_q()
        self.dof_vel = self._joint_qd()

        self._update_style_from_command()

        ang_vel = self._base_ang_vel_body() * self.obs_scales.ang_vel
        projected_gravity = (
            self.quat_rotate_inverse(
                self._base_quat_wxyz()[[1, 2, 3, 0]].astype(np.double),
                np.array([0, 0, -1]),
            )
            * self.obs_scales.projected_gravity
        )
        command = self.command_vel * COMMAND_OBS_SCALE
        delta_pos = self.dof_pos - self.default_dof_pos
        joint_pos = delta_pos[self.isaac_to_mujoco_idx] * self.obs_scales.joint_pos
        joint_vel = self.dof_vel[self.isaac_to_mujoco_idx] * self.obs_scales.joint_vel
        action = (
            np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions)
            * self.obs_scales.actions
        )
        gait_pad = np.zeros(self.cfg.sim.num_obs_per_step - 9 - 3 * self.cfg.sim.num_action)

        obs = np.concatenate(
            [
                ang_vel,
                projected_gravity,
                command,
                joint_pos,
                joint_vel,
                action,
                gait_pad,
            ],
            axis=0,
        ).astype(np.float32)
        assert obs.shape == (self.cfg.sim.num_obs_per_step,), f"单帧 obs 维度错误: {obs.shape}"
        if not self._obs_history_initialized:
            # Isaac Lab CircularBuffer.append() fills every history slot with the
            # first post-reset observation; zero-padding here creates an unseen
            # HIM/GRU startup state and can destabilize the policy immediately.
            self.obs_history[:] = np.tile(obs, self.cfg.sim.actor_obs_history_length)
            self._obs_history_initialized = True
        else:
            self.obs_history = np.roll(self.obs_history, shift=-self.cfg.sim.num_obs_per_step)
            self.obs_history[-self.cfg.sim.num_obs_per_step :] = obs

        return np.clip(self.obs_history, -self.clip_obs, self.clip_obs)

    def position_control(self) -> np.ndarray:
        """目标关节角 = action*0.25 + default (DEPLOY 序)。"""
        actions_scaled = self.action * self.cfg.sim.action_scale
        return actions_scaled[self.mujoco_to_isaac_idx] + self.default_dof_pos

    def _init_explicit_actuator(self) -> None:
        """按关节名建立 DEPLOY<->MJCF actuator 映射并缓存显式 PD/T-N 参数。

        MJCF motor 顺序 (链序, 含 waist) 与 DEPLOY_JOINT_ORDER 不同, 这里以
        关节名为准：状态读取走 jnt_qposadr/jnt_dofadr (DEPLOY 序), 力矩输出
        scatter 到 MJCF ctrl (waist 恒 0)。只改加载后的 ``MjModel``。
        """
        m = self.model
        act_by_joint: dict[str, int] = {}
        for a in range(m.nu):
            jid = int(m.actuator_trnid[a, 0])
            if jid < 0:
                raise RuntimeError(f"Actuator #{a} is not attached to a joint.")
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
            act_by_joint[name] = a

        n = len(DEPLOY_JOINT_ORDER)
        missing = [name for name in DEPLOY_JOINT_ORDER if name not in act_by_joint]
        if missing:
            raise RuntimeError(f"MJCF 缺少 DEPLOY 关节的 motor: {missing}")
        if m.nu not in (n, n + 1):
            raise RuntimeError(f"tiangong2plus MJCF 应有 {n} (旧裸 MJCF) 或 {n + 1} (含 waist) 个 motor, 实际 {m.nu}")
        self._deploy2ctrl = np.array([act_by_joint[name] for name in DEPLOY_JOINT_ORDER], dtype=int)
        self._waist_act_idx = act_by_joint.get(LOCKED_WAIST_JOINT, -1)

        self._exp_qpos_adr = np.zeros(n, dtype=int)
        self._exp_dof_adr = np.zeros(n, dtype=int)
        self._exp_kp = np.zeros(n, dtype=np.float64)
        self._exp_kd = np.zeros(n, dtype=np.float64)
        self._exp_x1 = np.zeros(n, dtype=np.float64)
        self._exp_x2 = np.zeros(n, dtype=np.float64)
        self._exp_y1 = np.zeros(n, dtype=np.float64)
        self._exp_y2 = np.zeros(n, dtype=np.float64)

        for i, name in enumerate(DEPLOY_JOINT_ORDER):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"MJCF has no joint named {name!r}")
            self._exp_qpos_adr[i] = m.jnt_qposadr[jid]
            self._exp_dof_adr[i] = m.jnt_dofadr[jid]
            p = ACTUATOR_PARAMS[name]
            self._exp_kp[i] = p["kp"]
            self._exp_kd[i] = p["kd"]
            self._exp_x1[i] = p["X1"]
            self._exp_x2[i] = p["X2"]
            self._exp_y1[i] = p["Y1"]
            self._exp_y2[i] = p["Y2"]
            # MJCF 关节无 damping；防御性清零，显式 PD 的 -kd*qd 是唯一阻尼来源。
            m.dof_damping[self._exp_dof_adr[i]] = 0.0
            # MJCF 未带 armature；注入 Isaac armature（转子惯量）以对齐训练动力学。
            m.dof_armature[self._exp_dof_adr[i]] = p["armature"]

            # motor 转直接力矩 (ctrl 即 Nm), ctrlrange = ±Y2 (Isaac
            # effort_limit_sim)，Python T-N 限幅是唯一动力学约束。
            a = int(self._deploy2ctrl[i])
            m.actuator_gaintype[a] = mujoco.mjtGain.mjGAIN_FIXED
            m.actuator_biastype[a] = mujoco.mjtBias.mjBIAS_NONE
            m.actuator_gainprm[a, :] = 0.0
            m.actuator_gainprm[a, 0] = 1.0
            m.actuator_biasprm[a, :] = 0.0
            m.actuator_ctrllimited[a] = 1
            m.actuator_ctrlrange[a, :] = (-self._exp_y2[i], self._exp_y2[i])
            m.jnt_actfrcrange[jid, :] = (-self._exp_y2[i], self._exp_y2[i])

        # waist motor (若存在): 同样转直接力矩, 但 ctrl 恒 0 (锁腰由 weld 保证)
        if self._waist_act_idx >= 0:
            a = self._waist_act_idx
            m.actuator_gaintype[a] = mujoco.mjtGain.mjGAIN_FIXED
            m.actuator_biastype[a] = mujoco.mjtBias.mjBIAS_NONE
            m.actuator_gainprm[a, :] = 0.0
            m.actuator_gainprm[a, 0] = 1.0
            m.actuator_biasprm[a, :] = 0.0

        # 锁腰 weld 的存在性校验
        eq_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_EQUALITY, i) for i in range(m.neq)]
        if "lock_waist" not in eq_names:
            raise RuntimeError("MJCF 缺少 lock_waist weld equality, 腰未锁死")
        print(
            f"[INFO] tiangong2plus explicit actuator enabled: nu={m.nu} "
            f"(waist motor idx={self._waist_act_idx}, ctrl 恒 0); motor torque = "
            "kp*(q_des-q)-kd*qd, clipped by the Isaac T-N curve; Isaac armature injected."
        )

    def torque_control(self) -> np.ndarray:
        """Explicit PD torque with T-N clip, 输出为完整 MJCF ctrl (nu 维)。

        Reads live qpos/qvel every physics substep (not the policy-step cache) so the
        PD law is evaluated against current state, matching MuJoCo's position servo.
        """
        q_des = self.position_control()
        q = self.data.qpos[self._exp_qpos_adr]
        qd = self.data.qvel[self._exp_dof_adr]
        tau = self._exp_kp * (q_des - q) - self._exp_kd * qd
        tau = self._clip_effort_tn(tau, qd)
        ctrl = np.zeros(self.model.nu, dtype=np.float64)
        ctrl[self._deploy2ctrl] = tau
        return ctrl

    def _clip_effort_tn(self, tau: np.ndarray, qd: np.ndarray) -> np.ndarray:
        """Torque-speed envelope (mirror of CustomDelayedPDActuator._clip_effort)。"""
        same_direction = (qd * tau) > 0
        max_effort = np.where(same_direction, self._exp_y1, self._exp_y2)
        # ramp from max_effort down to 0 between X1 and X2 once |qd| >= X1.
        k = -max_effort / (self._exp_x2 - self._exp_x1)
        ramp = np.clip(k * (np.abs(qd) - self._exp_x1) + max_effort, 0.0, None)
        max_effort = np.where(np.abs(qd) < self._exp_x1, max_effort, ramp)
        return np.clip(tau, -max_effort, max_effort)

    # ------------------------------------------------------------------
    # --log-torques / --stat-report / --record-data 记录
    # ------------------------------------------------------------------
    def _init_joint_log_limits(self) -> None:
        """Precompute per-joint limit columns (MuJoCo pos/tau; Isaac-matched qd)"""
        n = len(self.mujoco_joint_names)
        self._log_q_min = np.zeros(n, dtype=np.float64)
        self._log_q_max = np.zeros(n, dtype=np.float64)
        self._log_qd_min = np.zeros(n, dtype=np.float64)
        self._log_qd_max = np.zeros(n, dtype=np.float64)
        self._log_tau_min = np.zeros(n, dtype=np.float64)
        self._log_tau_max = np.zeros(n, dtype=np.float64)

        for i, name in enumerate(self.mujoco_joint_names):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"MJCF has no joint named {name!r}")
            self._log_q_min[i] = float(self.model.jnt_range[jid, 0])
            self._log_q_max[i] = float(self.model.jnt_range[jid, 1])
            # jnt_actfrcrange 已在 _init_explicit_actuator 设为 ±effort_limit_sim。
            self._log_tau_min[i] = float(self.model.jnt_actfrcrange[jid, 0])
            self._log_tau_max[i] = float(self.model.jnt_actfrcrange[jid, 1])
            vel_lim = JOINT_VELOCITY_LIMIT_RAD_S.get(name)
            if vel_lim is None:
                raise RuntimeError(f"No velocity_limit_sim entry for joint {name!r}")
            self._log_qd_min[i] = -vel_lim
            self._log_qd_max[i] = vel_lim

    @staticmethod
    def _joint_log_csv_header(joint_names: list[str]) -> list[str]:
        header = [
            "time",
            "base_vx",
            "base_vy",
            "base_vz",
            "base_wx",
            "base_wy",
            "base_wz",
            "base_z",
            "cmd_vx",
            "cmd_vy",
            "cmd_wz",
        ]
        for name in joint_names:
            header.extend(
                [
                    f"q_{name}",
                    f"qd_{name}",
                    f"tau_{name}",
                    f"q_min_{name}",
                    f"q_max_{name}",
                    f"qd_min_{name}",
                    f"qd_max_{name}",
                    f"tau_min_{name}",
                    f"tau_max_{name}",
                ]
            )
        return header

    def _record_joint_state(self) -> None:
        """Append one row: base/cmd vel + q/qd/tau and limit columns (MJCF actuator order)"""
        n = self.cfg.sim.num_action
        q = self._joint_q()
        qd = self._joint_qd()
        tau = np.array(self.data.actuator_force, dtype=np.float64)
        lin_yaw, ang_yaw = mujoco_root_vel_yaw_frame(self.data.qpos, self.data.qvel)
        cmd = np.asarray(self.command_vel, dtype=np.float64)

        row: list[float] = [
            self.data.time,
            float(lin_yaw[0]),
            float(lin_yaw[1]),
            float(lin_yaw[2]),
            float(ang_yaw[0]),
            float(ang_yaw[1]),
            float(ang_yaw[2]),
            float(self.data.qpos[2]),
            float(cmd[0]),
            float(cmd[1]),
            float(cmd[2]),
        ]
        for i in range(n):
            row.extend(
                [
                    q[i],
                    qd[i],
                    tau[i],
                    self._log_q_min[i],
                    self._log_q_max[i],
                    self._log_qd_min[i],
                    self._log_qd_max[i],
                    self._log_tau_min[i],
                    self._log_tau_max[i],
                ]
            )
        self._torque_log.append(row)

    def _write_torque_log(self) -> None:
        if self.torque_log_path is None or not self._torque_log:
            return
        header = self._joint_log_csv_header(self.mujoco_joint_names)
        out_dir = os.path.dirname(os.path.abspath(self.torque_log_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(self.torque_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(self._torque_log)
        print(
            f"[INFO] Wrote {len(self._torque_log)} joint-state samples to {self.torque_log_path}"
        )

    def _record_stat_sample(self) -> None:
        """统计模式逐物理子步记录 q/qd/tau（400 Hz；与 --log-torques 同一采样位置）。

        qpos/qvel 与 actuator_force(tau) 同属本子步积分前阶段，时刻自洽。
        """
        t = float(self.data.time)
        if t < self.stat_warmup_time:
            return
        if self._stat_t0 is None:
            self._stat_t0 = t
            print(f"[INFO] 统计记录开始 @ t={t:.2f}s")
        elif t - self._stat_t0 >= self.stat_record_time:
            self._stat_done = True
            print(f"[INFO] 统计记录结束，共 {len(self._stat_log['t'])} 个样本")
            return
        if float(self.data.qpos[2]) < STAT_FALL_BASE_HEIGHT:
            raise RuntimeError(
                f"机器人摔倒（base_z={self.data.qpos[2]:.2f} m @ t={t:.2f} s），终止统计。"
            )
        n = self.cfg.sim.num_action
        lin_yaw, _ = mujoco_root_vel_yaw_frame(self.data.qpos, self.data.qvel)
        self._stat_log["t"].append(t - self._stat_t0)
        self._stat_log["q"].append(self._joint_q().copy())
        self._stat_log["qd"].append(self._joint_qd().copy())
        self._stat_log["tau"].append(np.asarray(self.data.actuator_force, dtype=np.float64).copy())
        self._stat_log["vx"].append(float(lin_yaw[0]))

    def _write_stat_report(self) -> None:
        """统计窗口数据 -> legged_lab.utils.stat_report.generate_report（图/表/md/docx）。"""
        if not self.stat_report:
            return
        n_rec = len(self._stat_log["t"])
        if n_rec < 10:
            print(f"[WARN] 统计样本过少（{n_rec}），跳过报告生成。")
            return
        from legged_lab.utils.stat_report import generate_report

        n = self.cfg.sim.num_action
        t = np.asarray(self._stat_log["t"])
        q, qd, tau = (np.asarray(self._stat_log[k]) for k in ("q", "qd", "tau"))
        base_vx = np.asarray(self._stat_log["vx"])
        stance = [
            f"{self.mujoco_joint_names[i]} {np.rad2deg(self.default_dof_pos[i]):.1f}°"
            for i in range(n)
            if abs(self.default_dof_pos[i]) > 1e-3
        ]
        policy_dt = self.cfg.sim.dt * self.cfg.sim.decimation
        model_rows = [
            ("关节自由度", str(n), "受控关节数（锁腰 20 DoF）"),
            ("模型总质量", f"{float(np.sum(self.model.body_mass)):.2f} kg", "MJCF body_mass 求和"),
            ("初始基座高度", f"{self.cfg.robot.base_height_init:.2f} m", "MJCF pelvis 初始高度"),
            ("默认站姿", "；".join(stance), "非零默认角关节"),
            ("动作维度", str(n), "策略输出维度"),
            ("单步观测维度", str(self.cfg.sim.num_obs_per_step), f"历史 {self.cfg.sim.actor_obs_history_length} 步展平 = {self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length}"),
            ("策略频率", f"{1.0 / policy_dt:.0f} Hz", f"物理 {1.0 / self.cfg.sim.dt:.0f} Hz × decimation {self.cfg.sim.decimation}"),
        ]
        out_dir = self.stat_output_dir or os.path.join(
            DEFAULT_STAT_REPORT_DIR, datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        info = {
            "source": "MuJoCo Sim2Sim（tiangong2plus 20DoF 锁腰，显式 PD + T-N 限幅）",
            "policy": self.policy_path or "(self-test)",
            "cmd_vx": self.stat_vx,
            "policy_hz": 1.0 / self.cfg.sim.dt,  # 采样率 = 物理子步频率（400 Hz），非策略频率
            "warmup_desc": f"热身 {self.stat_warmup_time:.2f} s",
            "model_rows": model_rows,
        }
        generate_report(
            out_dir, t, q, qd, tau, base_vx,
            list(self.mujoco_joint_names), build_stat_report_params(list(self.mujoco_joint_names)), info,
        )

    @staticmethod
    def _analysis_csv_header(joint_names: list[str]) -> list[str]:
        header = [
            "time",
            "policy_step",
            "episode_id",
            "episode_step",
            "style_id",
            "style_name",
            "cmd_vx",
            "cmd_vy",
            "cmd_wz",
            "yaw_vx",
            "yaw_vy",
            "yaw_wz",
            "err_vx",
            "err_vy",
            "err_wz",
            "cadence",
            "stride_length",
            "him_vx",
            "him_vy",
            "him_vz",
            "base_x",
            "base_y",
            "base_z",
            "base_qw",
            "base_qx",
            "base_qy",
            "base_qz",
            "projected_gravity_x",
            "projected_gravity_y",
            "projected_gravity_z",
        ]
        for name in joint_names:
            header.extend([f"action_{name}", f"q_des_{name}", f"q_{name}", f"qd_{name}", f"tau_{name}"])
        return header

    def _record_analysis_data(self, est_vel: list[float] | None) -> None:
        n = self.cfg.sim.num_action
        q = self._joint_q()
        qd = self._joint_qd()
        tau = np.asarray(self.data.actuator_force, dtype=np.float64)
        q_des = self.position_control()
        lin_yaw, ang_yaw = mujoco_root_vel_yaw_frame(self.data.qpos, self.data.qvel)
        cmd = np.asarray(self.command_vel, dtype=np.float64)
        measured = np.array([lin_yaw[0], lin_yaw[1], ang_yaw[2]], dtype=np.float64)
        error = cmd - measured
        him = np.full(3, np.nan, dtype=np.float64) if est_vel is None else np.asarray(est_vel, dtype=np.float64)
        if him.shape != (3,):
            raise RuntimeError(f"HIM estimated velocity must have shape (3,), got {him.shape}")
        cadence, stride = self._gait_summary()
        quat_wxyz = np.asarray(self.data.qpos[3:7], dtype=np.float64)
        quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
        projected_gravity = self.quat_rotate_inverse(quat_xyzw, np.array([0.0, 0.0, -1.0]))
        style_name = {0: "stand", 1: "walk", 2: "run"}.get(int(self.style_ids), f"style_{self.style_ids}")
        row: list[float | int | str] = [
            float(self.data.time),
            self._analysis_step,
            self._episode_id,
            int(self.episode_length_buf),
            int(self.style_ids),
            style_name,
            *cmd.tolist(),
            *measured.tolist(),
            *error.tolist(),
            cadence,
            stride,
            *him.tolist(),
            *np.asarray(self.data.qpos[:3], dtype=np.float64).tolist(),
            *quat_wxyz.tolist(),
            *projected_gravity.tolist(),
        ]
        for i in range(n):
            row.extend([float(self.action[i]), float(q_des[i]), float(q[i]), float(qd[i]), float(tau[i])])
        self._analysis_log.append(row)
        self._analysis_step += 1

    def _write_analysis_log(self) -> None:
        if self.record_data_path is None or not self._analysis_log:
            return
        out_dir = os.path.dirname(os.path.abspath(self.record_data_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(self.record_data_path, "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(self._analysis_csv_header(self.mujoco_joint_names))
            writer.writerows(self._analysis_log)
        print(f"[INFO] Wrote {len(self._analysis_log)} policy-step analysis samples to {self.record_data_path}")

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def run(self) -> None:
        self.setup_keyboard_listener()
        self.listener.start()
        if self.stat_report:
            print(
                f"[INFO] 统计报告模式: vx={self.stat_vx:.2f} m/s, 热身 {self.stat_warmup_time:.2f}s, "
                f"记录 {self.stat_record_time:.2f}s @ {1.0 / self.cfg.sim.dt:.0f} Hz（物理子步），记录结束后自动退出。"
            )
        elif self.use_vel_cmd:
            print(f"[INFO] Using predefined velocity commands: {format_vel_cmd_sequence()}")
        else:
            self.setup_gamepad()

        try:
            while self.data.time < self.cfg.sim.sim_duration:
                if self.stat_report:
                    self.command_target_vel[:] = np.array([self.stat_vx, 0.0, 0.0])
                elif self.use_vel_cmd:
                    self.command_target_vel[:] = predefined_velocity_command(float(self.data.time))
                else:
                    self.update_command_from_gamepad()
                self.command_vel[:] = self.command_target_vel
                self.obs_history = self.get_obs()
                if self.policy is not None:
                    obs_t = torch.tensor(self.obs_history, dtype=torch.float32).unsqueeze(0)
                    policy_out = self.policy(obs_t).detach().numpy()
                    self.action[:] = policy_out[0, : self.cfg.sim.num_action]
                    self.action = np.clip(
                        self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions
                    )
                # self-test / 无 policy 时 action 恒 0, PD 保持默认站姿

                for _sim_update in range(self.cfg.sim.decimation):
                    step_start_time = time.time()

                    self.data.ctrl = self.torque_control()
                    mujoco.mj_step(self.model, self.data)
                    if self.torque_log_path is not None:
                        self._record_joint_state()
                    if self.plot_joints:
                        self._record_plot_sample()

                    if _sim_update == 0:
                        if self.stat_report and not self._stat_done:
                            self._record_stat_sample()
                        if self.stat_report and self._stat_done:
                            return

                    if self.viewer is not None:
                        self._update_viewer_camera_follow()
                        self.viewer.render()

                        reset_pressed = False
                        if hasattr(self.viewer, "is_key_pressed"):
                            reset_pressed = self.viewer.is_key_pressed(glfw.KEY_BACKSPACE)
                        else:
                            window = self._get_glfw_window()
                            if window is not None:
                                reset_pressed = glfw.get_key(window, glfw.KEY_BACKSPACE) == glfw.PRESS

                        if reset_pressed:
                            self.reset_simulation()

                    if self.should_exit():
                        return

                    elapsed = time.time() - step_start_time
                    sleep_time = self.cfg.sim.dt - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                self.episode_length_buf += 1
                self._update_gait_metrics()
                est_vel = (
                    get_him_est_lin_vel_from_jit(self.policy, self.obs_history)
                    if self._him_est_vel
                    else None
                )
                if self.record_data_path is not None:
                    self._record_analysis_data(est_vel)
                publish_mujoco_state(
                    self.data,
                    self.cfg.sim.num_action,
                    command=self.command_vel,
                    est_vel=est_vel,
                    joint_names=self.mujoco_joint_names,
                    joint_vel_limits=self._joint_vel_limits_lcm,
                    joint_effort_limits=self._joint_effort_limits_lcm,
                    sim_time_s=float(self.data.time),
                )
                self._maybe_print_velocity_status()
        finally:
            self.cleanup()

    @staticmethod
    def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """q is (x, y, z, w)."""
        q_w = q[-1]
        q_vec = q[:3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * q_w * 2.0
        c = q_vec * np.dot(q_vec, v) * 2.0
        return a - b + c

    def _apply_initial_viewer_camera(self) -> None:
        """Set a reasonable first frame only; mujoco_viewer mouse orbit is left alone after this."""
        self.viewer.cam.lookat[:] = self.data.qpos[:3]
        self.viewer.cam.distance = 5.0
        self.viewer.cam.azimuth = 0.0
        self.viewer.cam.elevation = -20.0

    def _update_viewer_camera_follow(self) -> None:
        target = self.data.qpos[:3].copy()
        target[2] = 0.8
        self.viewer.cam.lookat[:] = 0.92 * self.viewer.cam.lookat + 0.08 * target

    def reset_simulation(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = np.array([0.0, 0.0, self.cfg.robot.base_height_init])
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])  # identity quat (w, x, y, z)
        # 关节默认值按各自 qpos 地址写入 (freejoint 占 qpos[0:7], 不能连续切片)
        self.data.qpos[self._exp_qpos_adr] = self.default_dof_pos
        mujoco.mj_forward(self.model, self.data)

        self._episode_id += 1
        self.episode_length_buf = 0
        self.action = np.zeros(self.cfg.sim.num_action)
        self.command_vel = np.array([0.0, 0.0, 0.0])
        self.command_target_vel = np.array([0.0, 0.0, 0.0])
        self.keyboard_command_target_vel = np.array([0.0, 0.0, 0.0])
        self.style_ids = 0
        self.style_onehot = np.zeros(self.num_styles, dtype=np.float32)
        self.style_onehot[0] = 1.0
        self.obs_history = np.zeros(
            (self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length,),
            dtype=np.float32,
        )
        self._obs_history_initialized = False
        self._gait_window.clear()
        self._gait_step_count = 0
        self._gait_forward_dist = 0.0
        self._gait_run_steps = 0
        self._gait_contact_prev = False
        print("[INFO] Simulation reset")

    def _get_glfw_window(self):
        if self.viewer is None:
            return None
        if hasattr(self.viewer, "_window"):
            return self.viewer._window
        if hasattr(self.viewer, "window"):
            return self.viewer.window
        return None

    def should_exit(self) -> bool:
        window = self._get_glfw_window()
        if window is None:
            return False
        if glfw.window_should_close(window):
            return True
        return glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS

    def cleanup(self) -> None:
        self._write_torque_log()
        self._write_analysis_log()
        self._write_stat_report()
        self.exit_flag = True
        if self.listener is not None:
            self.listener.stop()
        if PYGAME_AVAILABLE and self.joystick is not None:
            pygame.quit()
        if self.viewer is not None:
            self.viewer.close()
        self._plot_all()

    # ------------------------------------------------------------------
    # 退出诊断绘图
    # ------------------------------------------------------------------
    def _record_plot_sample(self) -> None:
        n = self.cfg.sim.num_action
        q = self._joint_q()
        qd = self._joint_qd()
        tau = np.array(self.data.actuator_force, dtype=np.float64)
        lin_yaw, ang_yaw = mujoco_root_vel_yaw_frame(self.data.qpos, self.data.qvel)
        self._plot_time.append(float(self.data.time))
        self._plot_base_yaw_vel.append(np.array([lin_yaw[0], lin_yaw[1], ang_yaw[2]], dtype=np.float64))
        self._plot_joint_pos.append(q)
        self._plot_joint_vel.append(qd)
        self._plot_joint_tau.append(tau)

    def _plot_base_yaw_velocity(self, save_dir: str | None = None, run_tag: str = "") -> str | None:
        if not self.plot_joints or not self._plot_time or not self._plot_base_yaw_vel:
            return None
        import matplotlib.pyplot as plt

        t = np.asarray(self._plot_time, dtype=np.float64)
        v = np.asarray(self._plot_base_yaw_vel, dtype=np.float64)
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

    def _plot_tn_curves(self, save_dir: str | None = None, run_tag: str = "") -> str | None:
        """Scatter (joint_vel, torque) + T-N envelope for all actuator joints."""
        if not self.plot_joints or not self._plot_joint_vel or not self._plot_joint_tau:
            return None
        import matplotlib.pyplot as plt

        vel = np.asarray(self._plot_joint_vel, dtype=np.float64)  # (T, num_action), MuJoCo order
        tau = np.asarray(self._plot_joint_tau, dtype=np.float64)
        names = get_tn_plot_joint_names(self.mujoco_joint_names)
        if not names:
            print("[WARNING] No known actuator joints found for T-N plot.", flush=True)
            return None
        rows, cols = _joint_pos_subplot_shape(len(names))
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 2.6 * rows))
        axes_flat = np.atleast_1d(axes).ravel()
        for ax, name in zip(axes_flat, names):
            mj_idx = self.mujoco_joint_names.index(name)
            x1, x2, y1, y2 = get_tn_curve_params(name)
            ax.scatter(vel[:, mj_idx], tau[:, mj_idx], s=2, alpha=0.4)
            # Envelope (mirrors CustomDelayedPDActuator._clip_effort): same-direction
            # uses Y1, opposite-direction uses Y2; ramps to 0 between X1 and X2.
            vv = np.linspace(-x2, x2, 200)
            av = np.abs(vv)
            lim = lambda y: np.where(av < x1, y, np.clip(-y / (x2 - x1) * (av - x1) + y, 0.0, None))
            upper = np.where(vv > 0, lim(y1), lim(y2))
            lower = np.where(vv > 0, -lim(y2), -lim(y1))
            ax.plot(vv, upper, "r-", lw=1)
            ax.plot(vv, lower, "r-", lw=1)
            ax.set_title(name)
            ax.set_xlabel("velocity (rad/s)")
            ax.set_ylabel("torque (N·m)")
            ax.grid(True, alpha=0.3)
        for ax in axes_flat[len(names) :]:
            ax.axis("off")
        fig.suptitle("Motor T-N performance", fontsize=12)
        fig.tight_layout()
        return save_plot_figure(fig, save_dir, f"{run_tag}_tn.png")

    def _plot_joint_positions(self, save_dir: str | None = None, run_tag: str = "") -> str | None:
        """Plot joint position traces with MuJoCo joint position limits."""
        if not self.plot_joints or not self._plot_joint_pos:
            return None
        import matplotlib.pyplot as plt

        t = np.asarray(self._plot_time, dtype=np.float64)
        q = np.asarray(self._plot_joint_pos, dtype=np.float64)
        n = len(self.mujoco_joint_names)
        rows, cols = _joint_pos_subplot_shape(n)
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 2.1 * rows), sharex=True)
        axes_flat = np.atleast_1d(axes).ravel()

        for i, name in enumerate(self.mujoco_joint_names):
            ax = axes_flat[i]
            ax.plot(t, q[:, i], color="tab:blue", linewidth=0.9, label="q")
            ax.axhline(self._log_q_min[i], color="tab:red", linestyle="--", linewidth=0.9, label="limit")
            ax.axhline(self._log_q_max[i], color="tab:red", linestyle="--", linewidth=0.9)
            ax.set_title(name, fontsize=9)
            ax.set_ylabel("rad")
            ax.grid(True, alpha=0.3)

        for ax in axes_flat[n:]:
            ax.set_visible(False)
        for ax in axes_flat[-cols:]:
            if ax.get_visible():
                ax.set_xlabel("time (s)")

        handles, labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.suptitle("Joint positions with limits", fontsize=12)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
        return save_plot_figure(fig, save_dir, f"{run_tag}_joint_positions.png")

    def _plot_joint_velocity_torque(self, save_dir: str | None = None, run_tag: str = "") -> str | None:
        if not self.plot_joints or not self._plot_joint_vel or not self._plot_joint_tau:
            return None
        import matplotlib.pyplot as plt

        t = np.asarray(self._plot_time, dtype=np.float64)
        vel = np.asarray(self._plot_joint_vel, dtype=np.float64)
        tau = np.asarray(self._plot_joint_tau, dtype=np.float64)
        n = len(self.mujoco_joint_names)
        fig, axes = plt.subplots(n, 2, figsize=(12, max(2.0 * n, 4.0)), sharex=True)
        axes = np.atleast_2d(axes)
        for idx, name in enumerate(self.mujoco_joint_names):
            ax_vel, ax_tau = axes[idx, 0], axes[idx, 1]
            ax_vel.plot(t, vel[:, idx], color="tab:blue", linewidth=0.9)
            ax_tau.plot(t, tau[:, idx], color="tab:red", linewidth=0.9)
            ax_vel.axhline(self._log_qd_max[idx], color="tab:red", linestyle="--", linewidth=0.8, label="limit")
            ax_vel.axhline(self._log_qd_min[idx], color="tab:red", linestyle="--", linewidth=0.8)
            tau_dynamic_limits = tn_torque_limits_for_velocity(name, vel[:, idx])
            if tau_dynamic_limits is not None:
                tau_upper, tau_lower = tau_dynamic_limits
                ax_tau.plot(t, tau_upper, color="tab:red", linestyle="--", linewidth=0.8, label="T-N limit")
                ax_tau.plot(t, tau_lower, color="tab:red", linestyle="--", linewidth=0.8)
            else:
                ax_tau.axhline(self._log_tau_max[idx], color="tab:red", linestyle="--", linewidth=0.8, label="limit")
                ax_tau.axhline(self._log_tau_min[idx], color="tab:red", linestyle="--", linewidth=0.8)
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

    def _plot_total_joint_power(self, save_dir: str | None = None, run_tag: str = "") -> str | None:
        """所有关节功率绝对值之和随时间变化曲线。功率 = torque * velocity。"""
        if not self.plot_joints or not self._plot_joint_vel or not self._plot_joint_tau:
            return None
        import matplotlib.pyplot as plt

        t = np.asarray(self._plot_time, dtype=np.float64)
        vel = np.asarray(self._plot_joint_vel, dtype=np.float64)
        tau = np.asarray(self._plot_joint_tau, dtype=np.float64)
        # 每时刻所有关节 |tau_i * qd_i| 求和
        total_power = np.sum(np.abs(tau * vel), axis=1)
        fig, ax = plt.subplots(1, 1, figsize=(9, 3.6), sharex=True)
        ax.plot(t, total_power, color="tab:orange", linewidth=1.0)
        ax.set_title("Total absolute joint power")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("power (W)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return save_plot_figure(fig, save_dir, f"{run_tag}_total_joint_power.png")

    def _plot_all(self) -> list[str]:
        if not self.plot_joints:
            return []
        if not self._plot_time:
            print("[WARNING] --plot_joints enabled but no samples were recorded; skip plots.", flush=True)
            return []
        configure_matplotlib_backend(self.show_plots)
        save_dir = prepare_plot_output_dir(self.plot_dir)
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        saved_paths = [
            self._plot_base_yaw_velocity(save_dir, run_tag),
            self._plot_tn_curves(save_dir, run_tag),
            self._plot_joint_positions(save_dir, run_tag),
            self._plot_joint_velocity_torque(save_dir, run_tag),
            self._plot_total_joint_power(save_dir, run_tag),
        ]
        saved_paths = [path for path in saved_paths if path]

        import matplotlib.pyplot as plt

        if self.show_plots:
            plt.show()
        else:
            plt.close("all")
        if saved_paths:
            print(f"[sim2sim] Exit plots saved: {len(saved_paths)} file(s).", flush=True)
        return saved_paths

    # ------------------------------------------------------------------
    # 键盘 / 手柄
    # ------------------------------------------------------------------
    def adjust_command_vel(self, idx: int, increment: float) -> None:
        self.keyboard_command_target_vel[idx] += increment
        lower = np.array([-MAX_CMD_VX_NEG, -MAX_CMD[1], -MAX_CMD[2]], dtype=np.float64)
        upper = MAX_CMD.copy()
        self.keyboard_command_target_vel = np.clip(self.keyboard_command_target_vel, lower, upper)
        self.command_target_vel[:] = self.keyboard_command_target_vel

    def setup_gamepad(self) -> None:
        if not PYGAME_AVAILABLE:
            print("[WARNING] pygame not installed. Install with: pip install pygame")
            return

        if "SDL_JOYSTICK_DEVICE" not in os.environ:
            import glob

            js_devices = sorted(glob.glob("/dev/input/js*"))
            if js_devices:
                os.environ["SDL_JOYSTICK_DEVICE"] = js_devices[-1]

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            print("[WARNING] 未检测到手柄，使用键盘控制")
            return

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.gamepad_profile = _detect_gamepad_profile(self.joystick.get_name())
        print(f"[INFO] 检测到手柄: {self.joystick.get_name()} -> 使用 '{self.gamepad_profile}' 轴映射")
        print(
            f"[INFO] 轴映射: LEFT_X={GAMEPAD_PROFILES[self.gamepad_profile][0]}, "
            f"LEFT_Y={GAMEPAD_PROFILES[self.gamepad_profile][1]}, "
            f"RIGHT_X={GAMEPAD_PROFILES[self.gamepad_profile][2]}"
        )

    def read_gamepad_input(self) -> np.ndarray:
        if self.joystick is None:
            return np.zeros(3, dtype=np.float64)

        pygame.event.pump()
        axis_left_x, axis_left_y, axis_right_x = GAMEPAD_PROFILES.get(
            self.gamepad_profile, GAMEPAD_PROFILES["xbox"]
        )
        left_x = self.joystick.get_axis(axis_left_x)
        left_y = -self.joystick.get_axis(axis_left_y)
        right_x = self.joystick.get_axis(axis_right_x) if self.joystick.get_numaxes() > axis_right_x else 0.0
        return map_gamepad_axes_to_command(left_x=left_x, left_y=left_y, right_x=right_x)

    def update_command_from_gamepad(self) -> None:
        if self.joystick is None:
            self.command_target_vel[:] = self.keyboard_command_target_vel
            return

        joystick_command = self.read_gamepad_input()
        if np.any(joystick_command != 0.0):
            self.command_target_vel[:] = joystick_command
        else:
            self.command_target_vel[:] = self.keyboard_command_target_vel

    def setup_keyboard_listener(self) -> None:
        def on_press(key):
            try:
                if key.char == "8":
                    self.adjust_command_vel(0, 0.5)
                elif key.char == "2":
                    self.adjust_command_vel(0, -0.5)
                elif key.char == "4":
                    self.adjust_command_vel(1, -0.2)
                elif key.char == "6":
                    self.adjust_command_vel(1, 0.2)
                elif key.char == "7":
                    self.adjust_command_vel(2, 0.2)
                elif key.char == "9":
                    self.adjust_command_vel(2, -0.2)
                elif key.char == "0":
                    self.keyboard_command_target_vel[:] = 0.0
                    self.command_target_vel[:] = 0.0
            except AttributeError:
                pass

        self.listener = keyboard.Listener(on_press=on_press)


def run_self_test(cfg: SimToSimCfg, model_path: str, duration_s: float = 1.0) -> None:
    """结构自检：无 policy/viewer，PD 保持默认站姿，校验注入/观测/稳定性。

    注意: 该机器人 71.5 kg + 踝 kp=100/限 55 Nm (训练侧基线, 沿用 yrp03_v2,
    未整定), 零动作下静态平衡裕度不足, 约 2 s 后会缓慢前倾倒地——与 Isaac
    训练侧行为一致 (策略主动伺服才能站稳)。因此自检只验证 1 s 内的注入正确性
    /观测维度/数值稳定性, 不要求持续站立。
    """
    runner = MujocoRunner(
        cfg=cfg, policy_path=None, model_path=model_path, headless=True,
    )
    runner.cfg.sim.sim_duration = duration_s
    assert runner.model.nu in (cfg.sim.num_action, cfg.sim.num_action + 1), f"nu={runner.model.nu}"
    obs = runner.get_obs()
    assert obs.shape == (
        cfg.sim.num_obs_per_step * cfg.sim.actor_obs_history_length,
    ), f"obs_history shape={obs.shape}"
    assert np.isfinite(obs).all(), "obs_history 含非有限值"

    # 信息: 默认站姿下踝 pitch 静息重力矩 (qfrc_bias), 解释零动作为何站不久
    import mujoco
    waist_adr = mujoco.mj_name2id(runner.model, mujoco.mjtObj.mjOBJ_JOINT, LOCKED_WAIST_JOINT)
    waist_qpos = runner.model.jnt_qposadr[waist_adr]
    ankle_l = mujoco.mj_name2id(runner.model, mujoco.mjtObj.mjOBJ_JOINT, "ankle_pitch_l_joint")
    ankle_dof = runner.model.jnt_dofadr[ankle_l]
    print(f"[SELF-TEST] 静息踝 pitch 重力矩 qfrc_bias={float(runner.data.qfrc_bias[ankle_dof]):.1f} Nm "
          f"(kp=100 -> 需 {abs(float(runner.data.qfrc_bias[ankle_dof])) / 100:.2f} rad 静差)")

    n_steps = int(duration_s / cfg.sim.dt)
    for _ in range(n_steps):
        runner.data.ctrl = runner.torque_control()
        mujoco.mj_step(runner.model, runner.data)
    base_z = float(runner.data.qpos[2])
    quat_w = float(runner.data.qpos[3])
    max_abs_tau = float(np.max(np.abs(runner.data.actuator_force)))
    print(f"[SELF-TEST] steps={n_steps} base_z={base_z:.3f} quat_w={quat_w:.3f} max|tau|={max_abs_tau:.1f}")
    assert base_z > 0.85, f"base_z={base_z:.3f} 过低, 注入/PD 有误"
    assert abs(quat_w) > 0.95, f"quat_w={quat_w:.3f}, 姿态倾覆"
    assert max_abs_tau <= 330.0 + 1e-6, f"max|tau|={max_abs_tau:.1f} 超出 effort limit"
    waist_angle = float(runner.data.qpos[waist_qpos])
    assert abs(waist_angle) < 1e-3, f"腰未锁死: waist_yaw={waist_angle}"
    print("[SELF-TEST] PASS (1 s 注入/观测/稳定性 OK; 持续站立需策略, 见上说明)")


if __name__ == "__main__":
    args = parse_args()
    try:
        record_data_path = resolve_record_data_path(args.record_data)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(2)

    if not os.path.isfile(args.model):
        print(f"[ERROR] MJCF file not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Loaded model:  {args.model}")

    sim_cfg = SimToSimCfg()
    sim_cfg.sim.sim_duration = args.duration

    if args.self_test:
        run_self_test(sim_cfg, args.model)
        sys.exit(0)

    if not os.path.isfile(args.policy):
        print(f"[ERROR] Policy file not found: {args.policy}", file=sys.stderr)
        print(
            "        Run play.py first (auto-copies exported/policy.pt to Exported_policy/) or pass --policy.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"[INFO] Loaded policy: {args.policy}")

    runner = MujocoRunner(
        cfg=sim_cfg,
        policy_path=args.policy,
        model_path=args.model,
        torque_log_path=args.log_torques,
        record_data_path=record_data_path,
        plot_joints=args.plot_joints,
        plot_dir=args.plot_dir,
        show_plots=not args.no_show_plots,
        use_vel_cmd=args.use_vel_cmd,
        stat_report=args.stat_report,
        stat_vx=args.stat_vx,
        stat_warmup_time=args.stat_warmup_time,
        stat_record_time=args.stat_record_time,
        stat_output_dir=args.stat_output_dir,
    )
    runner.run()
