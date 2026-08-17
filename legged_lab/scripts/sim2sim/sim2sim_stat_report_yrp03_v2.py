"""yrp03-v2 MuJoCo Sim2Sim 奔跑数据统计与报告生成脚本（headless，无 viewer/LCM）。

复用 legged_lab/scripts/sim2sim/sim2sim_yrp03_v2.py 的配置与执行器参数（显式 PD +
T-N 曲线限幅、Isaac 关节顺序映射、观测构造），在固定前向速度指令下奔跑：
先按 --warmup_time 秒热身达到稳定奔跑，再按 --record_time 秒以 100 Hz 策略步记录
各关节 q/qd/tau，最后调用 legged_lab/utils/stat_report.py 生成
docs/template_analysis.md 所需的统计表、每类关节 4 张图与 report.md。

运行示例：
    # 先运行过一次 play.py（导出 Exported_policy/policy.pt 与 isaac_joint_names.txt）
    python legged_lab/scripts/sim2sim/sim2sim_stat_report_yrp03_v2.py \
        --lin_vel_x 10.0 --warmup_time 4.0 --record_time 2.0
    python legged_lab/scripts/sim2sim/sim2sim_stat_report_yrp03_v2.py \
        --policy logs/yrp03_v2_run/<run>/exported/policy.pt --output_dir logs/my_report

输出目录默认为 logs/sim2sim_stat_report/<时间戳>/，内含 report.md、joint_log.csv、
joint_params.csv 与 figs/*.png。
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import mujoco
import numpy as np
import torch

LEGGED_LAB_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
if LEGGED_LAB_ROOT_DIR not in sys.path:
    sys.path.insert(0, LEGGED_LAB_ROOT_DIR)

from legged_lab.scripts.sim2sim.sim2sim_yrp03_v2 import (
    ACTUATOR_PARAMS,
    COMMAND_OBS_SCALE,
    ISAAC_JOINT_NAMES_DEFAULT,
    MUJOCO_JOINT_NAMES,
    MujocoRunner,
    SimToSimCfg,
    build_idx_mappings,
    load_joint_names_file,
    set_mujoco_integrator,
)
from legged_lab.utils.stat_report import PARAM_KEYS, generate_report

DEFAULT_POLICY = os.path.join(LEGGED_LAB_ROOT_DIR, "Exported_policy", "policy.pt")
DEFAULT_MODEL = os.path.join(LEGGED_LAB_ROOT_DIR, "legged_lab/assets/yrp03_v2/xml/yrp03-v2-zb.xml")
DEFAULT_JOINT_NAMES_FILE = os.path.join(LEGGED_LAB_ROOT_DIR, "Exported_policy", "isaac_joint_names.txt")
DEFAULT_OUTPUT_ROOT = os.path.join(LEGGED_LAB_ROOT_DIR, "logs", "sim2sim_stat_report")

# 记录窗口内的防呆：基座低于该高度判定为摔倒，终止统计
FALL_BASE_HEIGHT = 0.5


# --- Args ---

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MuJoCo sim2sim joint statistics report (see module docstring).")
    parser.add_argument("--policy", type=str, default=DEFAULT_POLICY, help="TorchScript policy.pt（play.py 导出）。")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="yrp03-v2 floating-base MJCF。")
    parser.add_argument("--isaac-joint-names-file", type=str, default=DEFAULT_JOINT_NAMES_FILE, help="Isaac 关节顺序文件。")
    parser.add_argument("--lin_vel_x", type=float, default=10.0, help="固定前向速度指令 [m/s]。")
    parser.add_argument("--warmup_time", type=float, default=4.0, help="热身时长 [s]（仿真时间），达到稳定奔跑后再记录。")
    parser.add_argument("--record_time", type=float, default=2.0, help="记录时长 [s]（建议 >=2 个步态周期）。")
    parser.add_argument("--output_dir", type=str, default=None, help=f"输出目录；默认 {DEFAULT_OUTPUT_ROOT}/<时间戳>。")
    return parser.parse_args()


# --- MuJoCo 显式 PD 仿真（对齐 sim2sim_yrp03_v2.MujocoRunner，去掉 viewer/LCM/键盘） ---

def init_sim(cfg: SimToSimCfg, model_path: str) -> dict:
    """加载模型、注入显式 PD/T-N 参数与 armature、复位到默认站姿；返回仿真状态字典。"""
    model = mujoco.MjModel.from_xml_path(model_path)
    model.opt.timestep = cfg.sim.dt
    set_mujoco_integrator(model, cfg.sim.integrator)
    data = mujoco.MjData(model)
    n = cfg.sim.num_action
    if model.nu != n:
        raise RuntimeError(f"MJCF actuator 数 {model.nu} != num_action {n}")

    st = {
        "model": model, "data": data, "action_scale": cfg.sim.action_scale,
        "qpos_adr": np.zeros(n, dtype=int), "dof_adr": np.zeros(n, dtype=int),
        "kp": np.zeros(n), "kd": np.zeros(n),
        "x1": np.zeros(n), "x2": np.zeros(n), "y1": np.zeros(n), "y2": np.zeros(n),
    }
    for i in range(n):
        jid = int(model.actuator_trnid[i, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name != MUJOCO_JOINT_NAMES[i]:
            raise RuntimeError(f"actuator {i} 驱动关节 {name!r} != 期望 {MUJOCO_JOINT_NAMES[i]!r}")
        p = ACTUATOR_PARAMS[name]
        st["qpos_adr"][i] = model.jnt_qposadr[jid]
        st["dof_adr"][i] = model.jnt_dofadr[jid]
        for k in ("kp", "kd", "x1", "x2", "y1", "y2"):
            st[k][i] = p[k.upper() if k.startswith(("x", "y")) else k]
        model.dof_damping[st["dof_adr"][i]] = 0.0  # 被动阻尼已由显式 PD 的 -kd*qd 覆盖
        model.dof_armature[st["dof_adr"][i]] = p["armature"]
        model.actuator_gaintype[i] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_NONE
        model.actuator_gainprm[i, :] = 0.0
        model.actuator_gainprm[i, 0] = 1.0
        model.actuator_biasprm[i, :] = 0.0
        model.actuator_ctrllimited[i] = 1
        model.actuator_ctrlrange[i, :] = (-p["Y2"], p["Y2"])
        model.jnt_actfrcrange[jid, :] = (-p["Y2"], p["Y2"])

    # --- 默认站姿（MuJoCo 顺序，与 MujocoRunner.init_variables 一致） ---
    st["default_dof_pos"] = np.array(
        [0.0, 0.0, np.deg2rad(-15.0), np.deg2rad(30.0), np.deg2rad(-15.0), 0.0] * 2  # 左右腿
        + [0.5, 0.5, 0.0]  # 双肩 + 腰
    )
    reset_sim(st, cfg)
    return st


def reset_sim(st: dict, cfg: SimToSimCfg) -> None:
    data = st["data"]
    mujoco.mj_resetData(st["model"], data)
    data.qpos[0:3] = np.array([0.0, 0.0, cfg.robot.base_height_init])
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qpos[st["qpos_adr"]] = st["default_dof_pos"]
    mujoco.mj_forward(st["model"], data)
    st["action"] = np.zeros(cfg.sim.num_action)
    st["obs_history"] = None  # 首个观测 tile 填满整个历史（对齐 Isaac CircularBuffer）


def torque_control(st: dict) -> np.ndarray:
    """显式 PD + T-N 包络限幅（镜像 CustomDelayedPDActuator._clip_effort）。"""
    data = st["data"]
    # 策略输出为 Isaac 顺序，需重排为 MuJoCo 执行器顺序（对齐 MujocoRunner.position_control）
    q_des = st["action"][st["mj2isaac"]] * st["action_scale"] + st["default_dof_pos"]
    q = data.qpos[st["qpos_adr"]]
    qd = data.qvel[st["dof_adr"]]
    tau = st["kp"] * (q_des - q) - st["kd"] * qd
    same_dir = (qd * tau) > 0
    max_effort = np.where(same_dir, st["y1"], st["y2"])
    ramp = np.clip(-max_effort / (st["x2"] - st["x1"]) * (np.abs(qd) - st["x1"]) + max_effort, 0.0, None)
    max_effort = np.where(np.abs(qd) < st["x1"], max_effort, ramp)
    return np.clip(tau, -max_effort, max_effort)


def compute_obs(st: dict, cfg: SimToSimCfg, isaac2mj: list[int], command: np.ndarray) -> np.ndarray:
    """单步观测 + 10 步历史（对齐 MujocoRunner.get_obs，无 style/gait 维度）。"""
    data, n = st["data"], cfg.sim.num_action
    dof_pos = data.sensordata[0:n]
    dof_vel = data.sensordata[n : 2 * n]
    ang_vel = data.sensor("angular-velocity").data.astype(np.double) * cfg.normalization.obs_scales.ang_vel
    gravity = (
        MujocoRunner.quat_rotate_inverse(
            data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.double), np.array([0, 0, -1])
        )
        * cfg.normalization.obs_scales.projected_gravity
    )
    obs = np.concatenate(
        [
            ang_vel,
            gravity,
            command * COMMAND_OBS_SCALE,
            (dof_pos - st["default_dof_pos"])[isaac2mj] * cfg.normalization.obs_scales.joint_pos,
            dof_vel[isaac2mj] * cfg.normalization.obs_scales.joint_vel,
            np.clip(st["action"], -cfg.sim.clip_actions, cfg.sim.clip_actions),
        ]
    ).astype(np.float32)
    per_step = cfg.sim.num_obs_per_step
    if st["obs_history"] is None:
        st["obs_history"] = np.tile(obs, cfg.sim.actor_obs_history_length)
    else:
        st["obs_history"] = np.roll(st["obs_history"], shift=-per_step)
        st["obs_history"][-per_step:] = obs
    return np.clip(st["obs_history"], -cfg.normalization.clip_observations, cfg.normalization.clip_observations)


def base_vx_yaw(data) -> float:
    """yaw 系质心前向速度（qpos 四元数 wxyz）。"""
    w, x, y, z = data.qpos[3:7]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return float(np.cos(yaw) * data.qvel[0] + np.sin(yaw) * data.qvel[1])


def build_report_params() -> dict[str, np.ndarray]:
    """按 MUJOCO_JOINT_NAMES 顺序组装 stat_report 需要的每关节参数。"""
    n = len(MUJOCO_JOINT_NAMES)
    params = {k: np.zeros(n) for k in PARAM_KEYS}
    for i, name in enumerate(MUJOCO_JOINT_NAMES):
        p = ACTUATOR_PARAMS[name]
        params["kp"][i], params["kd"][i] = p["kp"], p["kd"]
        params["effort_limit"][i] = p["Y2"]
        params["velocity_limit"][i] = p["velocity_limit"]
        params["y1"][i], params["y2"][i] = p["Y1"], p["Y2"]
        params["x1"][i], params["x2"][i] = p["X1"], p["X2"]
    return params


# --- Main ---

def main() -> None:
    args = parse_args()
    if not os.path.isfile(args.policy):
        raise SystemExit(f"[ERROR] 策略文件不存在: {args.policy}（先运行 play.py 导出，或用 --policy 指定）")
    if not os.path.isfile(args.model):
        raise SystemExit(f"[ERROR] MJCF 文件不存在: {args.model}")
    if args.warmup_time < 0 or args.record_time <= 0:
        raise SystemExit("[ERROR] --warmup_time 需 >=0，--record_time 需 >0")

    isaac_joint_names = ISAAC_JOINT_NAMES_DEFAULT
    if os.path.isfile(args.isaac_joint_names_file):
        isaac_joint_names = load_joint_names_file(args.isaac_joint_names_file)
        print(f"[INFO] Loaded Isaac joint order ({len(isaac_joint_names)} joints) from {args.isaac_joint_names_file}")
    else:
        print(f"[WARN] {args.isaac_joint_names_file} 不存在，使用内置猜测顺序 ISAAC_JOINT_NAMES_DEFAULT。")
    if sorted(isaac_joint_names) != sorted(MUJOCO_JOINT_NAMES):
        raise SystemExit("[ERROR] Isaac 关节名与 MJCF 关节名不一致。")
    mj2isaac, isaac2mj = build_idx_mappings(MUJOCO_JOINT_NAMES, isaac_joint_names)

    cfg = SimToSimCfg()
    st = init_sim(cfg, args.model)
    st["mj2isaac"] = mj2isaac
    policy = torch.jit.load(args.policy)
    print(f"[INFO] Loaded policy: {args.policy}\n[INFO] Loaded model:  {args.model}")

    policy_dt = cfg.sim.dt * cfg.sim.decimation  # 0.01 s，100 Hz
    warmup_steps = int(round(args.warmup_time / policy_dt))
    record_steps = int(round(args.record_time / policy_dt))
    command = np.array([args.lin_vel_x, 0.0, 0.0], dtype=np.float64)
    n = cfg.sim.num_action

    def run_policy_step():
        obs = torch.tensor(compute_obs(st, cfg, isaac2mj, command), dtype=torch.float32).unsqueeze(0)
        action = policy(obs).detach().numpy()[0, :n]
        st["action"] = np.clip(action, -cfg.sim.clip_actions, cfg.sim.clip_actions)
        for _ in range(cfg.sim.decimation):
            st["data"].ctrl = torque_control(st)
            mujoco.mj_step(st["model"], st["data"])
        if st["data"].qpos[2] < FALL_BASE_HEIGHT:
            raise RuntimeError(f"机器人摔倒（base_z={st['data'].qpos[2]:.2f} m @ t={st['data'].time:.2f} s），终止统计。")

    print(f"[INFO] Warmup {args.warmup_time:.2f}s ({warmup_steps} 步) @ vx={args.lin_vel_x:.2f} m/s ...")
    for _ in range(warmup_steps):
        run_policy_step()

    print(f"[INFO] Recording {args.record_time:.2f}s ({record_steps} 步 @ {1.0 / policy_dt:.0f} Hz) ...")
    rec_t, rec_q, rec_qd, rec_tau, rec_vx = [], [], [], [], []
    t0 = float(st["data"].time)
    for _ in range(record_steps):
        run_policy_step()
        rec_t.append(float(st["data"].time) - t0)
        rec_q.append(np.asarray(st["data"].sensordata[0:n], dtype=np.float64).copy())
        rec_qd.append(np.asarray(st["data"].sensordata[n : 2 * n], dtype=np.float64).copy())
        rec_tau.append(np.asarray(st["data"].actuator_force, dtype=np.float64).copy())
        rec_vx.append(base_vx_yaw(st["data"]))

    t = np.asarray(rec_t)
    q, qd, tau = np.asarray(rec_q), np.asarray(rec_qd), np.asarray(rec_tau)
    base_vx = np.asarray(rec_vx)

    # --- 报告模型信息 ---
    total_mass = float(np.sum(st["model"].body_mass))
    stance = [
        f"{MUJOCO_JOINT_NAMES[i]} {np.rad2deg(st['default_dof_pos'][i]):.1f}°"
        for i in range(n)
        if abs(st["default_dof_pos"][i]) > 1e-3
    ]
    model_rows = [
        ("关节自由度", str(n), "受控关节数"),
        ("模型总质量", f"{total_mass:.2f} kg", "MJCF body_mass 求和"),
        ("初始基座高度", f"{cfg.robot.base_height_init:.2f} m", "MJCF pelvis 初始高度"),
        ("默认站姿", "；".join(stance), "非零默认角关节"),
        ("动作维度", str(n), "策略输出维度"),
        ("单步观测维度", str(cfg.sim.num_obs_per_step), f"历史 {cfg.sim.actor_obs_history_length} 步"),
        ("策略频率", f"{1.0 / policy_dt:.0f} Hz", f"物理 {1.0 / cfg.sim.dt:.0f} Hz × decimation {cfg.sim.decimation}"),
    ]
    out_dir = args.output_dir or os.path.join(DEFAULT_OUTPUT_ROOT, datetime.now().strftime("%Y%m%d_%H%M%S"))
    info = {
        "source": "MuJoCo Sim2Sim（yrp03-v2，显式 PD + T-N 限幅）",
        "policy": args.policy,
        "cmd_vx": args.lin_vel_x,
        "policy_hz": 1.0 / policy_dt,
        "warmup_desc": f"热身 {args.warmup_time:.2f} s（{warmup_steps} 步）",
        "model_rows": model_rows,
    }
    generate_report(out_dir, t, q, qd, tau, base_vx, MUJOCO_JOINT_NAMES, build_report_params(), info)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[INFO] Interrupted.")
    except Exception as exc:
        print(f"[ERROR] sim2sim_stat_report failed: {exc}")
        raise
