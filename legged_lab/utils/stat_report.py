"""关节奔跑数据统计与报告生成（play / sim2sim 共用）。

输入为统计窗口内各关节的时间序列（q/qd/tau，策略步采样），输出对齐
docs/template_analysis.md 的全部仿真侧信息：

- 统计表：峰值/RMS 转速、峰值/RMS 转矩、功率（P=tau*qd 正/负/绝对峰值与 RMS）、
  转速-转矩包络最大利用率与包络外样本数；
- 图片：每类关节 4 张（转速-时间、转矩-时间、功率-时间、转速-转矩包络），
  左右同类关节同图且同纵轴范围；包络由执行器 T-N 曲线参数（X1/X2/Y1/Y2）
  按四象限对称扩展生成；
- 文件：report.md、joint_log.csv（原始序列）、joint_params.csv、figs/*.png。

调用方需提供 params: dict[str, np.ndarray]，键为 PARAM_KEYS，按 joint_names 顺序对齐。
"""

from __future__ import annotations

import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

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


# --- Joint grouping ---

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


def build_joint_index(joint_names: list[str]) -> tuple[list[list[dict]], list[str]]:
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


def label_of(e: dict, use_cn: bool) -> str:
    if use_cn:
        return SIDE_LABEL_CN[e["side"]] + GROUP_LABEL_CN.get(e["group"], e["group"])
    return SIDE_LABEL_EN[e["side"]] + GROUP_LABEL_EN.get(e["group"], e["group"])


# --- Statistics ---

def envelope_cap(abs_w: np.ndarray, y: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """T-N 曲线能力上限（对称扩展）：|w|<=X1 时为 Y，之后线性降到 X2 处为 0。"""
    slope_safe = np.maximum(x2 - x1, 1e-9)
    return np.where(abs_w <= x1, y, np.clip(y * (x2 - abs_w) / slope_safe, 0.0, None))


def compute_joint_stats(
    qd: np.ndarray, tau: np.ndarray, params: dict[str, np.ndarray],
    by_group: list[list[dict]], groups: list[str],
) -> dict[int, dict]:
    """逐关节统计峰值/RMS/功率/包络利用率；返回 {joint_idx: dict}。"""
    stats = {}
    y_env = np.nanmax(np.stack([params["y1"], params["y2"], params["effort_limit"]]), axis=0)
    cap = envelope_cap(np.abs(qd), y_env[None, :], params["x1"][None, :], params["x2"][None, :])
    power = tau * qd
    for gi in range(len(groups)):
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


# --- Plotting ---

def plot_time_series(t, series_list, labels, ylabel, title, path):
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
    ax.set_ylim(lo - margin, hi + margin)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_envelope(qd, tau, params, entries, labels, title, path):
    """转速-转矩散点 + 四象限对称扩展包络（取组内最外边界）。"""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    colors = ["tab:blue", "tab:cyan", "tab:purple"]
    for k, e in enumerate(entries):
        i = e["idx"]
        ax.scatter(qd[:, i], tau[:, i], s=4, alpha=0.5, color=colors[k % len(colors)], label=labels[k])
    y_env = max(float(np.nanmax(params["y1"][e["idx"]])) for e in entries)
    y_env = max(y_env, float(np.nanmax([max(params["y2"][e["idx"]], params["effort_limit"][e["idx"]]) for e in entries])))
    x1 = max(float(params["x1"][e["idx"]]) for e in entries)
    x2 = max(float(params["x2"][e["idx"]]) for e in entries)
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
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --- Report ---

def fmt(v, nd=3):
    return f"{v:.{nd}f}"


def write_report(path, info, by_group, groups, stats, params, use_cn, fig_files):
    """生成 markdown 报告（结构对齐 template.docx 3.3 节）。"""
    lines = []
    lines.append("# 机器人奔跑初训练与验证 — 仿真数据统计报告\n")
    lines.append(f"- 数据来源: {info['source']}")
    lines.append(f"- 策略: `{info['policy']}`")
    lines.append(f"- 前向速度指令: {info['cmd_vx']:.2f} m/s；实际质心前向速度均值: {info['actual_vx']:.2f} m/s")
    lines.append(
        f"- 统计窗口: {info['n_samples']} 个策略步采样点（{info['policy_hz']:.0f} Hz，"
        f"约 {info['duration']:.2f} s）；{info['warmup_desc']}\n"
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
        i = by_group[gi][0]["idx"]
        lines.append(
            f"| {label_of({'side': 'm', 'group': gname}, use_cn)} | {fmt(params['kp'][i], 1)} | {fmt(params['kd'][i], 1)} "
            f"| {fmt(params['effort_limit'][i], 1)} | {fmt(params['velocity_limit'][i], 1)} "
            f"| {fmt(params['x1'][i], 1)} | {fmt(params['y1'][i], 1)} |"
        )
    lines.append("")

    key_map = {
        "峰值转速 (rad/s)": "peak_qd", "RMS 转速 (rad/s)": "rms_qd",
        "峰值转矩 (N·m)": "peak_tau", "RMS 转矩 (N·m)": "rms_tau",
        "最大正功率 (W)": "p_pos", "最大负功率 (W)": "p_neg",
        "绝对峰值功率 (W)": "p_abs", "RMS 功率 (W)": "p_rms",
        "最大包络利用率": "util", "包络外样本数": "n_out",
    }

    def _stat_table(title, cols):
        lines.append(f"## {title}\n")
        lines.append("| 关节 | " + " | ".join(cols) + " |")
        lines.append("|---|---|" + "---|" * (len(cols) - 1))
        for gi in range(len(groups)):
            for e in by_group[gi]:
                s = stats[e["idx"]]
                cells = []
                for c in cols:
                    k = key_map[c]
                    cells.append(f"{s[k] * 100:.1f}%" if k == "util" else (str(s[k]) if k == "n_out" else fmt(s[k])))
                lines.append(f"| {label_of(e, use_cn)} | " + " | ".join(cells) + " |")
        lines.append("")

    _stat_table("3. 转速统计", ["峰值转速 (rad/s)", "RMS 转速 (rad/s)"])
    _stat_table("4. 转矩统计", ["峰值转矩 (N·m)", "RMS 转矩 (N·m)"])
    _stat_table("5. 功率统计（P=τ×q̇）", ["最大正功率 (W)", "最大负功率 (W)", "绝对峰值功率 (W)", "RMS 功率 (W)"])
    _stat_table("6. 转速-转矩包络", ["最大包络利用率", "包络外样本数"])

    # --- 主要结果结论 ---
    all_idx = [e["idx"] for gi in range(len(groups)) for e in by_group[gi]]

    def _best(key):
        i = max(all_idx, key=lambda j: stats[j][key])
        e = next(e for gi in range(len(groups)) for e in by_group[gi] if e["idx"] == i)
        return label_of(e, use_cn), stats[i][key]

    lines.append("## 7. 主要结果\n")
    n_out_total = sum(stats[i]["n_out"] for i in all_idx)
    lb, v = _best("peak_qd")
    lines.append(f"- 绝对峰值转速最高: {lb} = {v:.3f} rad/s")
    lb, v = _best("peak_tau")
    lines.append(f"- 绝对峰值转矩最高: {lb} = {v:.3f} N·m")
    lb, v = _best("p_abs")
    lines.append(f"- 绝对峰值功率最高: {lb} = {v:.1f} W")
    lb, v = _best("util")
    lines.append(f"- 最大包络利用率: {lb} = {v * 100:.1f}%")
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
    """保存原始时间序列（便于二次绘图/核对）。"""
    header = ["time", "base_vx"] + [f"{k}_{n}" for n in joint_names for k in ("q", "qd", "tau")]
    data = np.column_stack([t, base_vx, q, qd, tau])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        np.savetxt(f, data, delimiter=",", fmt="%.6f")


# --- Orchestrator ---

def generate_report(out_dir, t, q, qd, tau, base_vx, joint_names, params, info) -> dict:
    """统计 + 绘图 + 落盘全流程；返回 stats。info 字段见 write_report。"""
    out_dir = os.path.abspath(out_dir)
    fig_dir = os.path.join(out_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)

    by_group, groups = build_joint_index(joint_names)
    stats = compute_joint_stats(qd, tau, params, by_group, groups)
    use_cn = setup_cjk_font()

    # --- 每类关节 4 张图：转速/转矩/功率-时间 + 转速-转矩包络 ---
    fig_files = []
    fig_records = []  # {"kind", "group", "path"}，供 docx 使用
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
            fpath = os.path.join(fig_dir, fname)
            plot_time_series(
                t, [data[:, e["idx"]] for e in entries], labels, ylabel,
                f"{gtitle} {title_cn}" if use_cn else f"{gtitle} {kind}-time",
                fpath,
            )
            fig_files.append(f"figs/{fname} — {gtitle}{title_cn if use_cn else ' ' + kind}")
            fig_records.append({"kind": kind, "group": gname, "path": fpath})
    for gi, gname in enumerate(groups):
        fig_no += 1
        entries = by_group[gi]
        labels = [label_of(e, use_cn) for e in entries]
        gtitle = GROUP_LABEL_CN.get(gname, gname) if use_cn else GROUP_LABEL_EN.get(gname, gname)
        fname = f"fig{fig_no:02d}_{gname}_envelope.png"
        fpath = os.path.join(fig_dir, fname)
        plot_envelope(
            qd, tau, params, entries, labels,
            f"{gtitle} 转速-转矩" if use_cn else f"{gtitle} speed-torque",
            fpath,
        )
        fig_files.append(f"figs/{fname} — {gtitle}{'转速-转矩' if use_cn else ' envelope'}")
        fig_records.append({"kind": "envelope", "group": gname, "path": fpath})
    print(f"[INFO] Saved {len(fig_files)} figures -> {fig_dir}")

    save_raw_csv(os.path.join(out_dir, "joint_log.csv"), t, q, qd, tau, joint_names, base_vx)
    param_csv = os.path.join(out_dir, "joint_params.csv")
    with open(param_csv, "w", encoding="utf-8") as f:
        f.write("joint,group,side," + ",".join(PARAM_KEYS) + "\n")
        for gi in range(len(groups)):
            for e in by_group[gi]:
                i = e["idx"]
                f.write(f"{e['name']},{e['group']},{e['side']}," + ",".join(fmt(params[k][i], 4) for k in PARAM_KEYS) + "\n")

    info = dict(info)
    info["actual_vx"] = float(np.mean(base_vx))
    info["n_samples"] = len(t)
    info["duration"] = float(t[-1]) if len(t) > 1 else 0.0
    report_path = os.path.join(out_dir, "report.md")
    write_report(report_path, info, by_group, groups, stats, params, use_cn, fig_files)
    with open(os.path.join(out_dir, "report_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    try:
        docx_path = os.path.join(out_dir, "report.docx")
        write_docx(docx_path, info, by_group, groups, stats, params, fig_records)
        print(f"[INFO] Docx -> {docx_path}")
    except ImportError:
        print("[WARN] 未安装 python-docx，跳过 report.docx 生成（pip install python-docx）。")

    all_idx = [e["idx"] for gi in range(len(groups)) for e in by_group[gi]]
    best = max(all_idx, key=lambda j: stats[j]["util"])
    print(f"[INFO] Report -> {report_path}")
    print(
        f"[RESULT] 峰值转矩 {max(stats[i]['peak_tau'] for i in all_idx):.1f} N·m | "
        f"峰值功率 {max(stats[i]['p_abs'] for i in all_idx):.1f} W | "
        f"最大包络利用率 {stats[best]['util'] * 100:.1f}% | "
        f"包络外样本 {sum(stats[i]['n_out'] for i in all_idx)}"
    )
    return stats


# --- Docx report（结构对齐 docs/template.docx 3.3 节） ---

# 各统计小节的口径与表格列定义（docx 与结论段落共用）
DOCX_SECTIONS = [
    {
        "kind": "speed",
        "title": "转速-时间曲线",
        "desc": "表中给出各关节转速随时间的变化。左右同类关节采用相同纵轴范围，便于直接比较；峰值按绝对值最大值统计。",
        "cols": [("峰值转速 (rad/s)", "peak_qd"), ("RMS 转速 (rad/s)", "rms_qd")],
    },
    {
        "kind": "torque",
        "title": "转矩-时间曲线",
        "desc": "给出关节输出侧转矩随时间的变化。统计表采用“峰值转矩、RMS转矩”口径，峰值为时间窗内绝对值最大值。",
        "cols": [("峰值转矩 (N·m)", "peak_tau"), ("RMS 转矩 (N·m)", "rms_tau")],
    },
    {
        "kind": "power",
        "title": "功率-时间曲线",
        "desc": "机械功率按P=τ×q̇计算。正值表示驱动输出机械能，负值表示制动或潜在能量回馈；表中同时列出正、负、绝对峰值和RMS。",
        "cols": [
            ("最大正功率 (W)", "p_pos"), ("最大负功率 (W)", "p_neg"),
            ("绝对峰值功率 (W)", "p_abs"), ("RMS 功率 (W)", "p_rms"),
        ],
    },
    {
        "kind": "envelope",
        "title": "转速-力矩曲线",
        "desc": "轨迹为仿真关节工作点，虚线和浅色区域为执行器T-N曲线按四象限对称扩展后的能力包络。",
        "cols": [("最大包络利用率", "util"), ("包络外样本数", "n_out")],
    },
]
_CN_NUM = "一二三四"


def _joint_stat_conclusion(section: dict, by_group, groups, stats, n_samples: int) -> str:
    """生成该小节的“主要结果”结论段落（口径对齐 template.docx）。"""
    all_entries = [(e, stats[e["idx"]]) for gi in range(len(groups)) for e in by_group[gi]]

    def _best(key, fn=max):
        e, s = fn(all_entries, key=lambda es: es[1][key] if fn is max else -es[1][key])
        return label_of(e, True), s[key]

    kind = section["kind"]
    if kind == "speed":
        lb, v = _best("peak_qd")
        lb2, v2 = _best("rms_qd")
        return (
            f"主要结果绝对峰值转速最高的是{lb}，为{v:.3f}rad/s（{v * 9.5493:.1f}rpm）；"
            f"RMS转速最高的是{lb2}，为{v2:.3f}rad/s（{v2 * 9.5493:.1f}rpm）。"
        )
    if kind == "torque":
        lb, v = _best("peak_tau")
        lb2, v2 = _best("rms_tau")
        return f"主要结果绝对峰值转矩最高的是{lb}，为{v:.3f}N·m；RMS转矩最高的是{lb2}，为{v2:.3f}N·m。"
    if kind == "power":
        lb_pos, v_pos = _best("p_pos")
        lb_abs, v_abs = _best("p_abs")
        lb_rms, v_rms = _best("p_rms")
        lb_neg, v_neg = min(all_entries, key=lambda es: es[1]["p_neg"])[0], min(s["p_neg"] for _, s in all_entries)
        return (
            f"主要结果最大正功率、绝对峰值功率和RMS功率分别出现在{lb_pos}（{v_pos:.1f}W）、"
            f"{lb_abs}（{v_abs:.1f}W）和{lb_rms}（{v_rms:.1f}W）；最大负功率出现在{lb_neg}，为{v_neg:.1f}W。"
        )
    n_out = sum(s["n_out"] for _, s in all_entries)
    lb, v = _best("util")
    return (
        f"包络判断按执行器T-N曲线的四象限对称假设，{len(all_entries)}个关节共{n_samples * len(all_entries)}个工作点中"
        f"{n_out}个位于能力包络外；最高利用率为{lb}的{v * 100:.1f}%。该结论只表示瞬时转速-转矩工作点满足包络，"
        "仍需结合额定连续转矩、热容量、减速比、效率和母线电压完成最终选型。"
    )


def write_docx(path, info, by_group, groups, stats, params, fig_records):
    """按 docs/template.docx 的 3.3 节结构生成 docx 报告（表格 + 内嵌图 + 结论）。"""
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Cm

    doc = Document()
    doc.add_heading("3.3 机器人奔跑初训练与验证", level=1)

    # --- 3.3.1 仿真参数设置 ---
    doc.add_heading("3.3.1 仿真参数设置", level=2)
    doc.add_paragraph(
        f"本节给出机器人奔跑策略在{info['source']}中的主要仿真、训练与控制参数，"
        "并统计高速奔跑工况下的关节位置、角速度、输出转矩及功率数据，为后续关节载荷分析与电机选型提供依据。"
    )
    doc.add_paragraph(
        f"本次验证前向速度指令设定为{info['cmd_vx']:.1f}m/s（实际质心前向速度均值{info['actual_vx']:.2f}m/s）。"
        f"统计窗口包含{info['n_samples']}个{info['policy_hz']:.0f}Hz策略步采样点，"
        f"覆盖约{info['duration']:.2f}s；{info['warmup_desc']}后开始记录。"
    )
    doc.add_paragraph("（一）机器人模型与状态空间")
    table = doc.add_table(rows=1, cols=3)
    table.style = "Table Grid"
    for j, h in enumerate(("参数项", "设置值", "说明")):
        table.rows[0].cells[j].text = h
    for row in info["model_rows"]:
        cells = table.add_row().cells
        for j, v in enumerate(row):
            cells[j].text = str(v)

    doc.add_paragraph("（二）关节控制与输出限值")
    doc.add_paragraph(
        "关节采用位置目标与PD控制生成输出转矩，并按照模型中配置的关节输出侧转矩限值进行截断"
        "（执行器T-N曲线：|ω|≤X1时上限为Y1/Y2，之后线性降至X2处为0）。各类关节参数如下。"
    )
    table = doc.add_table(rows=1, cols=6)
    table.style = "Table Grid"
    for j, h in enumerate(("关节类型", "Kp", "Kd", "转矩限值（N·m）", "速度限值（rad/s）", "X1/Y1（拐点/同向峰值）")):
        table.rows[0].cells[j].text = h
    for gi, gname in enumerate(groups):
        i = by_group[gi][0]["idx"]
        cells = table.add_row().cells
        cells[0].text = label_of({"side": "m", "group": gname}, True)
        cells[1].text = fmt(params["kp"][i], 1)
        cells[2].text = fmt(params["kd"][i], 1)
        cells[3].text = fmt(params["effort_limit"][i], 1)
        cells[4].text = fmt(params["velocity_limit"][i], 1)
        cells[5].text = f"{fmt(params['x1'][i], 1)} / {fmt(params['y1'][i], 1)}"

    # --- 3.3.2 仿真结果分析 ---
    doc.add_heading("3.3.2 仿真结果分析", level=2)
    fig_no = 0
    for si, section in enumerate(DOCX_SECTIONS):
        doc.add_heading(f"（{_CN_NUM[si]}）{section['title']}", level=3)
        doc.add_paragraph(section["desc"])
        table = doc.add_table(rows=1, cols=1 + len(section["cols"]))
        table.style = "Table Grid"
        table.rows[0].cells[0].text = "关节"
        for j, (h, _) in enumerate(section["cols"]):
            table.rows[0].cells[j + 1].text = h
        for gi in range(len(groups)):
            for e in by_group[gi]:
                s = stats[e["idx"]]
                cells = table.add_row().cells
                cells[0].text = label_of(e, True)
                for j, (_, key) in enumerate(section["cols"]):
                    cells[j + 1].text = (
                        f"{s[key] * 100:.1f}%" if key == "util" else (str(s[key]) if key == "n_out" else fmt(s[key]))
                    )
        doc.add_paragraph(_joint_stat_conclusion(section, by_group, groups, stats, info["n_samples"]))
        for rec in fig_records:
            if rec["kind"] != section["kind"]:
                continue
            fig_no += 1
            doc.add_picture(rec["path"], width=Cm(14))
            caption = doc.add_paragraph(f"图{fig_no}  {GROUP_LABEL_CN.get(rec['group'], rec['group'])}{section['title']}")
            caption.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph("注：模板中“理论转速/理论转矩/理论功率”列来自设计指标，需人工补充。")
    doc.save(path)


# --- 从已有输出目录再生成（不重跑仿真） ---

def regenerate(out_dir: str) -> dict:
    """读取 out_dir 下的 joint_log.csv / joint_params.csv / report_info.json，重新生成全部产物。"""
    out_dir = os.path.abspath(out_dir)
    csv_path = os.path.join(out_dir, "joint_log.csv")
    param_path = os.path.join(out_dir, "joint_params.csv")
    info_path = os.path.join(out_dir, "report_info.json")
    for p in (csv_path, param_path, info_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"缺少文件: {p}")

    with open(csv_path, encoding="utf-8") as f:
        header = next(csv.reader(f))
    joint_names = [c[2:] for c in header if c.startswith("q_")]
    if not joint_names:
        raise RuntimeError(f"joint_log.csv 中未找到 q_<joint> 列: {csv_path}")
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data[None, :]
    col = {name: header.index(name) for name in header}
    t, base_vx = data[:, col["time"]], data[:, col["base_vx"]]
    q = np.column_stack([data[:, col[f"q_{n}"]] for n in joint_names])
    qd = np.column_stack([data[:, col[f"qd_{n}"]] for n in joint_names])
    tau = np.column_stack([data[:, col[f"tau_{n}"]] for n in joint_names])

    params = {k: np.full(len(joint_names), np.nan) for k in PARAM_KEYS}
    with open(param_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["joint"] not in joint_names:
                continue
            i = joint_names.index(row["joint"])
            for k in PARAM_KEYS:
                params[k][i] = float(row[k])
    if np.isnan(params["effort_limit"]).any():
        raise RuntimeError(f"joint_params.csv 缺少关节参数: {param_path}")

    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)
    print(f"[INFO] Regenerating report from {out_dir} ({len(t)} samples, {len(joint_names)} joints)")
    return generate_report(out_dir, t, q, qd, tau, base_vx, joint_names, params, info)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="从已有输出目录（joint_log.csv + joint_params.csv + report_info.json）重新生成统计/图/report.md/report.docx。"
    )
    parser.add_argument("out_dir", type=str, help="generate_report 的输出目录")
    args = parser.parse_args()
    regenerate(args.out_dir)
