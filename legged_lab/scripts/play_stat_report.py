"""Isaac Sim play 奔跑数据统计与报告生成脚本。

参照 legged_lab/scripts/play.py 的流程加载策略并钉死速度指令，在固定工况下奔跑，
按策略步（decimation 后）记录各关节的位置 q、转速 qd、输出转矩 tau，统计并生成
docs/template_analysis.md 所需的全部仿真侧信息：

1. 仿真参数表：机器人模型参数（自由度、总质量、初始基座高度、默认站姿、观测/动作维度）
   与各类关节的 Kp/Kd/转矩限值/速度限值（含执行器 T-N 曲线 X1/X2/Y1 参数）。
2. 统计表：每关节的峰值/RMS 转速、峰值/RMS 转矩、功率（正/负/绝对峰值与 RMS，P=tau*qd）、
   转速-转矩包络最大利用率与包络外样本数。
3. 图片：每类关节 4 张（转速-时间、转矩-时间、功率-时间、转速-转矩包络），左右同类关节
   同图且同纵轴范围；包络由执行器 T-N 曲线参数按四象限对称扩展生成。

注意：模板中“理论转速/理论转矩/理论功率”列来自设计指标，仿真内无法获得，报告中留空待人工填写。

运行示例（headless 推荐）：
    python legged_lab/scripts/play_stat_report.py --task yrp03_v2_run --headless \
        --lin_vel_x 10.0 --warmup_steps 200 --record_steps 100
输出目录默认为 <checkpoint_run_dir>/play_report/，内含 report.md、joint_log.csv、params CSV 与 figs/*.png。
"""

import argparse
import os

import numpy as np
import torch
from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip


# --- Args ---

def parse_args():
    parser = argparse.ArgumentParser(description="Play + joint statistics report (see module docstring).")
    parser.add_argument("--task", type=str, default="yrp03_v2_run", help="Name of the task.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (report uses env 0).")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
    parser.add_argument("--lin_vel_x", type=float, default=10.0, help="Fixed forward velocity command [m/s].")
    parser.add_argument("--warmup_steps", type=int, default=200, help="Policy steps to stabilize before recording.")
    parser.add_argument("--record_steps", type=int, default=100, help="Policy steps to record (>=2 gait cycles).")
    parser.add_argument("--output_dir", type=str, default=None, help="Report output dir; default <run_dir>/play_report.")
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()
    return args


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from isaaclab_tasks.utils import get_checkpoint_path

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg
from rsl_rl.runners import AmpOnPolicyRunner, MultiAmpOnPolicyRunner, OnPolicyRunner

# --- Constants & Config ---

# 关节分组（同类关节一张图）；顺序决定报告表格与图片编号顺序
GROUP_ORDER = [
    "hip_roll", "hip_yaw", "hip_pitch", "knee_pitch",
    "ankle_pitch", "ankle_roll", "shoulder_pitch", "waist_yaw",
]
GROUP_LABEL_CN = {
    "hip_roll": "髋关节 Roll", "hip_yaw": "髋关节 Yaw", "hip_pitch": "髋关节 Pitch",
    "knee_pitch": "膝关节 Pitch", "ankle_pitch": "踝关节 Pitch", "ankle_roll": "踝关节 Roll",
    "shoulder_pitch": "肩关节 Pitch", "waist_yaw": "腰关节 Yaw",
}
GROUP_LABEL_EN = {
    "hip_roll": "Hip Roll", "hip_yaw": "Hip Yaw", "hip_pitch": "Hip Pitch",
    "knee_pitch": "Knee Pitch", "ankle_pitch": "Ankle Pitch", "ankle_roll": "Ankle Roll",
    "shoulder_pitch": "Shoulder Pitch", "waist_yaw": "Waist Yaw",
}
SIDE_LABEL_CN = {"l": "左", "r": "右", "m": ""}
SIDE_LABEL_EN = {"l": "L ", "r": "R ", "m": ""}
CJK_FONT_CANDIDATES = [
    "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei",
    "WenQuanYi Micro Hei", "SimHei", "Microsoft YaHei", "PingFang SC",
]
PARAM_KEYS = ["kp", "kd", "effort_limit", "velocity_limit", "y1", "y2", "x1", "x2"]


# --- Utilities ---

def setup_cjk_font() -> bool:
    """尝试注册中文字体；返回是否可用（不可用时图内用英文标签）。"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in CJK_FONT_CANDIDATES:
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            print(f"[INFO] 使用中文字体: {name}")
            return True
    print("[WARN] 未找到中文字体，图内标签使用英文。")
    return False


def parse_joint_name(joint_name: str) -> tuple[str, str]:
    """'hip_roll_l_joint' -> ('hip_roll', 'l')；'waist_yaw_joint' -> ('waist_yaw', 'm')。"""
    base = joint_name[:-len("_joint")] if joint_name.endswith("_joint") else joint_name
    if base.endswith("_l"):
        return base[:-2], "l"
    if base.endswith("_r"):
        return base[:-2], "r"
    return base, "m"


def build_joint_index(joint_names: list[str]) -> tuple[list[dict], list[str]]:
    """按 GROUP_ORDER 组织关节；返回 (每组条目列表, 实际出现的组名列表)。

    条目: {"name", "group", "side", "idx"}；组内按 l/r/m 排序。
    """
    entries = []
    for idx, name in enumerate(joint_names):
        group, side = parse_joint_name(name)
        entries.append({"name": name, "group": group, "side": side, "idx": idx})
    extra = sorted({e["group"] for e in entries} - set(GROUP_ORDER))
    groups = [g for g in GROUP_ORDER if any(e["group"] == g for e in entries)] + extra
    side_rank = {"l": 0, "r": 1, "m": 2}
    by_group = [
        sorted([e for e in entries if e["group"] == g], key=lambda e: side_rank[e["side"]])
        for g in groups
    ]
    return by_group, groups


def extract_joint_params(env, joint_names: list[str]) -> dict[str, np.ndarray]:
    """从 articulation data 与执行器实例提取每关节参数（kp/kd/限值/T-N 曲线）。"""
    robot, n = env.robot, len(joint_names)
    p = {k: np.full(n, np.nan) for k in PARAM_KEYS}
    eff = getattr(robot.data, "joint_effort_limits", None)
    if eff is not None:
        p["effort_limit"][:] = np.abs(eff[0].detach().cpu().numpy())
    vel = getattr(robot.data, "joint_vel_limits", None)
    if vel is not None:
        p["velocity_limit"][:] = np.abs(vel[0].detach().cpu().numpy())

    def _put(key, val, gidx):
        if val is None:
            return
        arr = val.detach().cpu().numpy() if torch.is_tensor(val) else np.asarray(val)
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 2:
            arr = arr[0]
        if arr.shape[0] == gidx.shape[0]:
            p[key][gidx] = arr

    for act in getattr(robot, "actuators", {}).values():
        idx = getattr(act, "joint_indices", None)
        if idx is None:
            gidx = np.arange(n)
        elif isinstance(idx, slice):
            gidx = np.arange(n)[idx]
        else:
            gidx = idx.detach().cpu().numpy().astype(int) if torch.is_tensor(idx) else np.asarray(idx, dtype=int)
        _put("kp", getattr(act, "stiffness", None), gidx)
        _put("kd", getattr(act, "damping", None), gidx)
        _put("y1", getattr(act, "_effort_y1", None), gidx)
        _put("y2", getattr(act, "_effort_y2", None), gidx)
        _put("x1", getattr(act, "_velocity_x1", None), gidx)
        _put("x2", getattr(act, "_velocity_x2", None), gidx)

    # --- 兜底：缺失参数用限值补齐 ---
    p["y1"] = np.where(np.isnan(p["y1"]), p["effort_limit"], p["y1"])
    p["y2"] = np.where(np.isnan(p["y2"]), p["effort_limit"], p["y2"])
    p["x2"] = np.where(np.isnan(p["x2"]), p["velocity_limit"], p["x2"])
    p["x1"] = np.where(np.isnan(p["x1"]), p["x2"], p["x1"])
    p["velocity_limit"] = np.where(np.isnan(p["velocity_limit"]), p["x2"], p["velocity_limit"])
    if np.isnan(p["effort_limit"]).any() or np.isnan(p["x2"]).any():
        bad = [joint_names[i] for i in range(n) if np.isnan(p["effort_limit"][i]) or np.isnan(p["x2"][i])]
        raise RuntimeError(f"关节限值缺失，无法构建包络: {bad}")
    return p


def envelope_cap(abs_w: np.ndarray, y: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """T-N 曲线能力上限（对称扩展）：|w|<=X1 时为 Y，之后线性降到 X2 处为 0。"""
    slope_safe = np.maximum(x2 - x1, 1e-9)
    return np.where(abs_w <= x1, y, np.clip(y * (x2 - abs_w) / slope_safe, 0.0, None))


def compute_stats(t, qd, tau, p, by_group, groups):
    """逐关节统计峰值/RMS/功率/包络利用率；返回 {joint_idx: dict} 与组级结论。"""
    stats = {}
    y_env = np.nanmax(np.stack([p["y1"], p["y2"], p["effort_limit"]]), axis=0)
    cap = envelope_cap(np.abs(qd), y_env[None, :], p["x1"][None, :], p["x2"][None, :])
    power = tau * qd
    for gi, gname in enumerate(groups):
        for e in by_group[gi]:
            i = e["idx"]
            c = cap[:, i]
            util = np.where(c > 1e-9, np.abs(tau[:, i]) / np.maximum(c, 1e-9), np.inf)
            stats[i] = {
                "peak_qd": float(np.max(np.abs(qd[:, i]))), "rms_qd": float(np.sqrt(np.mean(qd[:, i] ** 2))),
                "peak_tau": float(np.max(np.abs(tau[:, i]))), "rms_tau": float(np.sqrt(np.mean(tau[:, i] ** 2))),
                "p_pos": float(np.max(power[:, i])), "p_neg": float(np.min(power[:, i])),
                "p_abs": float(np.max(np.abs(power[:, i]))), "p_rms": float(np.sqrt(np.mean(power[:, i] ** 2))),
                "util": float(np.max(util)), "n_out": int(np.sum(util > 1.0 + 1e-6)),
            }
    return stats


def label_of(e, use_cn: bool) -> str:
    if use_cn:
        return SIDE_LABEL_CN[e["side"]] + GROUP_LABEL_CN.get(e["group"], e["group"])
    return SIDE_LABEL_EN[e["side"]] + GROUP_LABEL_EN.get(e["group"], e["group"])


# --- Plotting ---

def plot_time_series(t, series_list, labels, ylabel, title, path, same_ylim=True):
    """单组关节时序图：左右同轴同纵轴范围。"""
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for k, (s, lb) in enumerate(zip(series_list, labels)):
        ax.plot(t, s, linewidth=0.9, label=lb, color=colors[k % len(colors)])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    data = np.concatenate([np.atleast_1d(s) for s in series_list])
    lo, hi = float(np.min(data)), float(np.max(data))
    margin = max((hi - lo) * 0.1, 1e-3)
    if same_ylim:
        ax.set_ylim(lo - margin, hi + margin)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_envelope(qd, tau, p, entries, labels, title, path):
    """转速-转矩散点 + 四象限对称扩展包络。"""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    colors = ["tab:blue", "tab:cyan", "tab:purple"]
    for k, e in enumerate(entries):
        i = e["idx"]
        ax.scatter(qd[:, i], tau[:, i], s=4, alpha=0.5, color=colors[k % len(colors)], label=labels[k])
    # 包络取组内最外边界
    y_env = max(float(np.nanmax(p["y1"][e["idx"]])) for e in entries)
    y_env = max(y_env, float(np.nanmax([max(p["y2"][e["idx"]], p["effort_limit"][e["idx"]]) for e in entries])))
    x1 = max(float(p["x1"][e["idx"]]) for e in entries)
    x2 = max(float(p["x2"][e["idx"]]) for e in entries)
    bx = np.array([-x2, -x1, 0.0, x1, x2])
    by = envelope_cap(np.abs(bx), np.full_like(bx, y_env), np.full_like(bx, x1), np.full_like(bx, x2))
    ax.plot(bx, by, color="tab:orange", linestyle="--", linewidth=1.2, label="envelope")
    ax.plot(bx, -by, color="tab:orange", linestyle="--", linewidth=1.2)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Speed [rad/s]")
    ax.set_ylabel("Torque [N·m]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --- Report ---

def fmt(v, nd=3):
    return f"{v:.{nd}f}"


def write_report(path, info, by_group, groups, stats, p, use_cn, fig_files):
    """生成 markdown 报告（结构对齐 template.docx 3.3 节）。"""
    lines = []
    lines.append("# 机器人奔跑初训练与验证 — 仿真数据统计报告\n")
    lines.append(f"- 任务: `{info['task']}`；checkpoint: `{info['checkpoint']}`")
    lines.append(f"- 前向速度指令: {info['cmd_vx']:.2f} m/s；实际质心前向速度均值: {info['actual_vx']:.2f} m/s")
    lines.append(
        f"- 统计窗口: {info['n_samples']} 个策略步采样点（{info['policy_hz']:.0f} Hz，"
        f"约 {info['duration']:.2f} s）；热身 {info['warmup_steps']} 步\n"
    )

    lines.append("## 1. 机器人模型与状态空间\n")
    lines.append("| 参数项 | 设置值 | 说明 |")
    lines.append("|---|---|---|")
    for row in info["model_rows"]:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## 2. 关节控制与输出限值\n")
    lines.append("| 关节类型 | Kp | Kd | 转矩限值（N·m） | 速度限值（rad/s） | X1（拐点 rad/s） | Y1（同向峰值 N·m） |")
    lines.append("|---|---|---|---|---|---|---|")
    for gi, gname in enumerate(groups):
        e0 = by_group[gi][0]
        i = e0["idx"]
        lines.append(
            f"| {label_of({'side': 'm', 'group': gname}, use_cn)} | {fmt(p['kp'][i], 1)} | {fmt(p['kd'][i], 1)} "
            f"| {fmt(p['effort_limit'][i], 1)} | {fmt(p['velocity_limit'][i], 1)} "
            f"| {fmt(p['x1'][i], 1)} | {fmt(p['y1'][i], 1)} |"
        )
    lines.append("")

    def _stat_table(title, cols, key_fmt):
        lines.append(f"## {title}\n")
        lines.append("| 关节 | " + " | ".join(cols) + " |")
        lines.append("|---|---|" + "---|" * (len(cols) - 1))
        for gi, gname in enumerate(groups):
            for e in by_group[gi]:
                s = stats[e["idx"]]
                lines.append(f"| {label_of(e, use_cn)} | " + " | ".join(key_fmt(s, k) for k in cols) + " |")
        lines.append("")

    key_map = {
        "峰值转速 (rad/s)": "peak_qd", "RMS 转速 (rad/s)": "rms_qd",
        "峰值转矩 (N·m)": "peak_tau", "RMS 转矩 (N·m)": "rms_tau",
        "最大正功率 (W)": "p_pos", "最大负功率 (W)": "p_neg",
        "绝对峰值功率 (W)": "p_abs", "RMS 功率 (W)": "p_rms",
        "最大包络利用率": "util", "包络外样本数": "n_out",
    }
    for title, cols in [
        ("3. 转速统计", ["峰值转速 (rad/s)", "RMS 转速 (rad/s)"]),
        ("4. 转矩统计", ["峰值转矩 (N·m)", "RMS 转矩 (N·m)"]),
        ("5. 功率统计（P=τ×q̇）", ["最大正功率 (W)", "最大负功率 (W)", "绝对峰值功率 (W)", "RMS 功率 (W)"]),
        ("6. 转速-转矩包络", ["最大包络利用率", "包络外样本数"]),
    ]:
        _stat_table(title, cols, lambda s, c, _m=key_map: (
            f"{s[_m[c]] * 100:.1f}%" if _m[c] == "util" else (str(s[_m[c]]) if _m[c] == "n_out" else fmt(s[_m[c]]))
        ))

    # --- 主要结果结论 ---
    all_idx = [e["idx"] for gi in range(len(groups)) for e in by_group[gi]]
    def _best(key, fn=max):
        i = fn(all_idx, key=lambda j: stats[j][key])
        e = next(e for gi in range(len(groups)) for e in by_group[gi] if e["idx"] == i)
        return label_of(e, use_cn), stats[i][key]
    lines.append("## 7. 主要结果\n")
    n_out_total = sum(stats[i]["n_out"] for i in all_idx)
    lines.append(f"- 绝对峰值转速最高: {_best('peak_qd')[0]} = {_best('peak_qd')[1]:.3f} rad/s")
    lines.append(f"- 绝对峰值转矩最高: {_best('peak_tau')[0]} = {_best('peak_tau')[1]:.3f} N·m")
    lines.append(f"- 绝对峰值功率最高: {_best('p_abs')[0]} = {_best('p_abs')[1]:.1f} W")
    lines.append(f"- 最大包络利用率: {_best('util')[0]} = {_best('util')[1] * 100:.1f}%")
    lines.append(
        f"- 包络外样本: {n_out_total} / {info['n_samples'] * len(all_idx)} 个工作点"
        "（包络按执行器 T-N 曲线四象限对称扩展；最终选型仍需结合额定连续转矩、热容量、减速比、效率和母线电压）\n"
    )

    lines.append("## 8. 图片索引\n")
    for f in fig_files:
        lines.append(f"- {f}")
    lines.append("")
    lines.append("> 注：模板中“理论转速/理论转矩/理论功率”列来自设计指标，需人工补充。")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_raw_csv(path, t, q, qd, tau, joint_names, base_vx):
    """保存原始时间序列（便于复用 plot_sim2sim_ultra_yaw_csv.py 等工具二次绘图）。"""
    header = ["time", "base_vx"] + [f"{k}_{n}" for n in joint_names for k in ("q", "qd", "tau")]
    data = np.column_stack([t, base_vx, q, qd, tau])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        np.savetxt(f, data, delimiter=",", fmt="%.6f")


# --- Main ---

def main():
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)

    # --- 与 play.py 一致的确定性 play 配置：关噪声/随机化、平地、固定指令 ---
    env_cfg.noise.add_noise = False
    for evt in (
        "push_robot", "randomize_rigid_body_inertia", "add_base_mass", "add_others_mass",
        "randomize_rigid_body_com", "randomize_joint_armature", "randomize_joint_friction",
        "randomize_actuator_friction", "randomize_motor_strength", "randomize_actuator_gains",
        "physics_material",
    ):
        if hasattr(env_cfg.domain_rand.events, evt):
            setattr(env_cfg.domain_rand.events, evt, None)
    env_cfg.scene.max_episode_length_s = 1000.0
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = 2.5
    env_cfg.commands.debug_vis = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.cmd_bucket_probs = None
    env_cfg.commands.use_curriculum = False
    env_cfg.commands.run_high_speed_sample_frac = None
    env_cfg.commands.resampling_time_range = (0.1, 0.1)
    env_cfg.commands.ranges.lin_vel_x = (args_cli.lin_vel_x, args_cli.lin_vel_x)
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.ranges.heading = (0.0, 0.0)
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)
    env_cfg.scene.height_scanner.enable_height_scan = False
    env_cfg.scene.height_scanner.enable_feet_height_scan = False
    env_cfg.scene.terrain_generator = None
    env_cfg.scene.terrain_type = "plane"
    env_cfg.sim.physx.gpu_max_rigid_patch_count = 2 * 2**15

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    env_class = task_registry.get_task_class(args_cli.task)
    env = env_class(env_cfg, args_cli.headless)

    log_root_path = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    print(f"[INFO] Loading checkpoint: {resume_path}")

    runner = eval(agent_cfg.runner_class_name)(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.device)

    joint_names = [str(n) for n in env.robot.data.joint_names]
    by_group, groups = build_joint_index(joint_names)
    params = extract_joint_params(env, joint_names)
    step_dt = float(env.step_dt)
    n_joints = len(joint_names)
    print(f"[INFO] joints={n_joints}, groups={groups}, step_dt={step_dt * 1000:.1f} ms")

    out_dir = os.path.abspath(args_cli.output_dir) if args_cli.output_dir else os.path.join(log_dir, "play_report")
    fig_dir = os.path.join(out_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)

    obs, _ = env.get_observations()

    def _pin_command():
        cg = env.command_generator
        cg.vel_command_b[:, 0] = args_cli.lin_vel_x
        cg.vel_command_b[:, 1] = 0.0
        cg.vel_command_b[:, 2] = 0.0
        if hasattr(cg, "cmd_target_b"):
            cg.cmd_target_b[:, 0] = args_cli.lin_vel_x
            cg.cmd_target_b[:, 1] = 0.0
            cg.cmd_target_b[:, 2] = 0.0

    # --- 热身：达到稳定奔跑 ---
    print(f"[INFO] Warmup {args_cli.warmup_steps} steps @ vx={args_cli.lin_vel_x:.2f} m/s ...")
    for _ in range(args_cli.warmup_steps):
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
        _pin_command()

    # --- 记录 ---
    print(f"[INFO] Recording {args_cli.record_steps} steps ...")
    rec_t, rec_q, rec_qd, rec_tau, rec_vx = [], [], [], [], []
    truncated = False
    prev_ep_len = int(env.episode_length_buf[0].item())
    for _ in range(args_cli.record_steps):
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
        _pin_command()
        ep_len = int(env.episode_length_buf[0].item())
        if ep_len < prev_ep_len:
            print(f"[WARN] 记录窗口内发生 episode 重置，已截断（{len(rec_t)} 个样本）。请增加 warmup 或检查稳定性。")
            truncated = True
            break
        prev_ep_len = ep_len
        rec_t.append(ep_len * step_dt)
        rec_q.append(env.robot.data.joint_pos[0].detach().cpu().numpy())
        rec_qd.append(env.robot.data.joint_vel[0].detach().cpu().numpy())
        rec_tau.append(env.robot.data.applied_torque[0].detach().cpu().numpy())
        rec_vx.append(float(env.robot.data.root_lin_vel_w[0, 0].detach().cpu().item()))

    if len(rec_t) < 10:
        raise RuntimeError(f"有效样本过少（{len(rec_t)}），无法统计。请检查策略/工况。")
    t = np.asarray(rec_t) - rec_t[0]
    q, qd, tau = np.asarray(rec_q), np.asarray(rec_qd), np.asarray(rec_tau)
    base_vx = np.asarray(rec_vx)
    stats = compute_stats(t, qd, tau, params, by_group, groups)

    # --- 绘图：每类关节 4 张 ---
    use_cn = setup_cjk_font()
    fig_files = []
    fig_no = 0
    kinds = [
        ("speed", "Speed [rad/s]", qd, "转速-时间"),
        ("torque", "Torque [N·m]", tau, "转矩-时间"),
        ("power", "Power [W]", tau * qd, "功率-时间"),
    ]
    for kind, ylabel, data, title_cn in kinds:
        for gi, gname in enumerate(groups):
            fig_no += 1
            entries = by_group[gi]
            labels = [label_of(e, use_cn) for e in entries]
            gtitle = GROUP_LABEL_CN.get(gname, gname) if use_cn else GROUP_LABEL_EN.get(gname, gname)
            fname = f"fig{fig_no:02d}_{gname}_{kind}.png"
            plot_time_series(
                t, [data[:, e["idx"]] for e in entries], labels, ylabel,
                f"{gtitle} {title_cn}" if use_cn else f"{gtitle} {kind}-time",
                os.path.join(fig_dir, fname),
            )
            fig_files.append(f"figs/{fname} — {gtitle}{title_cn if use_cn else ' ' + kind}")
    for gi, gname in enumerate(groups):
        fig_no += 1
        entries = by_group[gi]
        labels = [label_of(e, use_cn) for e in entries]
        gtitle = GROUP_LABEL_CN.get(gname, gname) if use_cn else GROUP_LABEL_EN.get(gname, gname)
        fname = f"fig{fig_no:02d}_{gname}_envelope.png"
        plot_envelope(
            qd, tau, params, entries, labels,
            f"{gtitle} 转速-转矩" if use_cn else f"{gtitle} speed-torque",
            os.path.join(fig_dir, fname),
        )
        fig_files.append(f"figs/{fname} — {gtitle}{'转速-转矩' if use_cn else ' envelope'}")
    print(f"[INFO] Saved {len(fig_files)} figures -> {fig_dir}")

    # --- 原始数据与参数 CSV ---
    save_raw_csv(os.path.join(out_dir, "joint_log.csv"), t, q, qd, tau, joint_names, base_vx)
    param_csv = os.path.join(out_dir, "joint_params.csv")
    with open(param_csv, "w", encoding="utf-8") as f:
        f.write("joint,group,side," + ",".join(PARAM_KEYS) + "\n")
        for gi in range(len(groups)):
            for e in by_group[gi]:
                i = e["idx"]
                f.write(f"{e['name']},{e['group']},{e['side']}," + ",".join(fmt(params[k][i], 4) for k in PARAM_KEYS) + "\n")

    # --- 报告 ---
    robot = env.robot
    try:
        total_mass = float(robot.root_physx_view.get_masses()[0].sum())
    except Exception:
        total_mass = float("nan")
    try:
        init_h = float(env_cfg.scene.robot.init_state.pos[2])
    except Exception:
        init_h = float("nan")
    default_pos = robot.data.default_joint_pos[0].detach().cpu().numpy()
    stance = [
        f"{label_of(e, use_cn)} {np.rad2deg(default_pos[e['idx']]):.1f}°"
        for gi in range(len(groups)) for e in by_group[gi]
        if abs(default_pos[e["idx"]]) > 1e-3
    ]
    model_rows = [
        ("关节自由度", str(n_joints), "受控关节数"),
        ("模型总质量", f"{total_mass:.2f} kg", "各刚体质量求和"),
        ("初始基座高度", f"{init_h:.2f} m", "资产 init_state"),
        ("默认站姿", "；".join(stance) if stance else "全 0°", "非零默认角关节"),
        ("动作维度", str(getattr(env, "num_actions", n_joints)), "策略输出维度"),
        ("单步观测维度", str(getattr(env, "num_obs_per_step", getattr(env, "actor_obs_dim", "?"))), "Actor 输入见 cfg"),
        ("策略频率", f"{1.0 / step_dt:.0f} Hz", f"step_dt={step_dt * 1000:.1f} ms"),
    ]
    info = {
        "task": args_cli.task, "checkpoint": resume_path, "cmd_vx": args_cli.lin_vel_x,
        "actual_vx": float(np.mean(base_vx)), "n_samples": len(t), "duration": float(t[-1]),
        "policy_hz": 1.0 / step_dt, "warmup_steps": args_cli.warmup_steps, "model_rows": model_rows,
    }
    report_path = os.path.join(out_dir, "report.md")
    write_report(report_path, info, by_group, groups, stats, params, use_cn, fig_files)

    print(f"[INFO] Report -> {report_path}")
    print(f"[INFO] Raw CSV -> {os.path.join(out_dir, 'joint_log.csv')} ({len(t)} samples, truncated={truncated})")
    all_idx = [e["idx"] for gi in range(len(groups)) for e in by_group[gi]]
    best = max(all_idx, key=lambda j: stats[j]["util"])
    print(
        f"[RESULT] 峰值转矩 {max(stats[i]['peak_tau'] for i in all_idx):.1f} N·m | "
        f"峰值功率 {max(stats[i]['p_abs'] for i in all_idx):.1f} W | "
        f"最大包络利用率 {stats[best]['util'] * 100:.1f}% | "
        f"包络外样本 {sum(stats[i]['n_out'] for i in all_idx)}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[INFO] Interrupted.")
    except Exception as exc:
        print(f"[ERROR] play_stat_report failed: {exc}")
        raise
    finally:
        simulation_app.close()
