#!/usr/bin/env python3
"""Plot joint-state CSV from sim2sim_ultraYaw.py (--log-torques).

Layout: 左列（线速度+角速度）独立铺满高度；右列 n 行关节 ω|τ（sharex）

CSV (5 ms/step, yaw frame — same as Isaac / play LCM):
  time, base_vx..vz, base_wx..wz, base_z, cmd_vx..wz,
  q_<joint>, qd_<joint>, tau_<joint>, ...limits...  (×13)

Usage:
  python legged_lab/scripts/sim2sim_ultraYaw.py --log-torques logs/sim2sim.csv --duration 30
  python legged_lab/scripts/play.py --task ultra_game_yaw --log-joints logs/play_joint_state.csv
  python legged_lab/scripts/plot_sim2sim_ultra_yaw_csv.py logs/sim2sim.csv
  python legged_lab/scripts/plot_sim2sim_ultra_yaw_csv.py logs/sim2sim.csv --all-joints --save out.png
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

_GRID_LEFT, _GRID_RIGHT = 0.05, 0.99
_GRID_BOTTOM, _GRID_TOP = 0.03, 0.95
_GRID_HSPACE, _GRID_WSPACE = 0.05, 0.20


def _monitor_pixels() -> tuple[int, int]:
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return int(w), int(h)
    except Exception:
        return 1920, 1080


def _figure_fill_screen(width_frac: float = 0.93, height_frac: float = 0.96) -> tuple[float, float]:
    dpi = float(plt.rcParams.get("figure.dpi", 100.0))
    w_px, h_px = _monitor_pixels()
    return w_px * width_frac / dpi, h_px * height_frac / dpi


def _maximize_figure_window(fig) -> None:
    try:
        win = fig.canvas.manager.window
        if hasattr(win, "showMaximized"):
            win.showMaximized()
        elif hasattr(win, "state"):
            win.state("zoomed")
    except Exception:
        pass


def _setup_maximize_on_first_draw(fig) -> None:
    def _once(_event):
        _maximize_figure_window(fig)
        fig.canvas.mpl_disconnect(_cid)

    _cid = fig.canvas.mpl_connect("draw_event", _once)


DEFAULT_PLOT_JOINTS: list[tuple[str, str]] = [
    ("L Hip Yaw", "hip_yaw_l_joint"),
    ("L Hip Roll", "hip_roll_l_joint"),
    ("L Hip Pitch", "hip_pitch_l_joint"),
    ("L Knee Pitch", "knee_pitch_l_joint"),
    ("L Ankle Pitch", "ankle_pitch_l_joint"),
    ("Waist Yaw", "waist_yaw_joint"),
    ("L Shoulder Pitch", "shoulder_pitch_l_joint"),
]

BASE_LIN_COLS = ("base_vx", "base_vy", "base_vz")
BASE_ANG_COLS = ("base_wx", "base_wy", "base_wz")
CMD_COLS = ("cmd_vx", "cmd_vy", "cmd_wz")


def _pretty_label(joint_name: str) -> str:
    return joint_name.replace("_joint", "").replace("_", " ").title()


def parse_joint_names_from_header(header: list[str]) -> list[str]:
    names: list[str] = []
    for col in header:
        m = re.match(r"^q_(?!min_|max_)(.+)$", col)
        if m:
            names.append(m.group(1))
    return names


def _has_cols(header: list[str], cols: tuple[str, ...]) -> bool:
    return all(c in header for c in cols)


class Sim2SimLog:
    def __init__(
        self,
        t: np.ndarray,
        joints: dict[str, dict[str, np.ndarray]],
        base: dict[str, np.ndarray] | None,
    ):
        self.t = t
        self.joints = joints
        self.base = base


def load_sim2sim_csv(path: Path) -> Sim2SimLog:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty or invalid CSV: {path}")
        fieldnames = list(reader.fieldnames)
        joint_names = parse_joint_names_from_header(fieldnames)
        if not joint_names:
            raise ValueError(
                f"No q_<joint> columns in {path}. "
                "Use sim2sim_ultraYaw.py --log-torques to record."
            )

        rows = list(reader)
        if not rows:
            raise ValueError(f"No data rows in {path}")

        t = np.array([float(r["time"]) for r in rows], dtype=np.float64)
        joints: dict[str, dict[str, np.ndarray]] = {}
        for jname in joint_names:
            joints[jname] = {
                "qd": np.array([float(r[f"qd_{jname}"]) for r in rows], dtype=np.float64),
                "tau": np.array([float(r[f"tau_{jname}"]) for r in rows], dtype=np.float64),
                "qd_min": np.array([float(r[f"qd_min_{jname}"]) for r in rows], dtype=np.float64),
                "qd_max": np.array([float(r[f"qd_max_{jname}"]) for r in rows], dtype=np.float64),
                "tau_min": np.array([float(r[f"tau_min_{jname}"]) for r in rows], dtype=np.float64),
                "tau_max": np.array([float(r[f"tau_max_{jname}"]) for r in rows], dtype=np.float64),
            }

        base: dict[str, np.ndarray] | None = None
        if _has_cols(fieldnames, BASE_LIN_COLS + BASE_ANG_COLS):
            base = {}
            for c in BASE_LIN_COLS + BASE_ANG_COLS + ("base_z",) + CMD_COLS:
                if c in fieldnames:
                    base[c] = np.array([float(r[c]) for r in rows], dtype=np.float64)

        return Sim2SimLog(t=t, joints=joints, base=base)


def build_plot_list(joint_names_in_csv: list[str], all_joints: bool) -> list[tuple[str, str]]:
    if all_joints:
        return [(_pretty_label(j), j) for j in joint_names_in_csv]
    out = [(lbl, jn) for lbl, jn in DEFAULT_PLOT_JOINTS if jn in joint_names_in_csv]
    return out or [(_pretty_label(j), j) for j in joint_names_in_csv[:7]]


def auto_ylim(ax, arrays: list[np.ndarray], min_margin: float) -> None:
    data = np.concatenate([a for a in arrays if len(a) > 0])
    if len(data) < 2:
        return
    y_min, y_max = float(np.min(data)), float(np.max(data))
    margin = min_margin if y_min == y_max else max(min_margin, (y_max - y_min) * 0.1)
    ax.set_ylim(y_min - margin, y_max + margin)


def plot_sim2sim_log(
    log: Sim2SimLog,
    plot_joints: list[tuple[str, str]],
    title: str,
    save: str | None,
    show: bool,
) -> None:
    t = log.t
    n = len(plot_joints)
    t0, t1 = float(t[0]), float(t[-1])
    dt_ms = (t[1] - t[0]) * 1000.0 if len(t) > 1 else 5.0
    has_base = log.base is not None

    plt.style.use("ggplot")
    fig = plt.figure(figsize=_figure_fill_screen())
    fig.suptitle(title, fontsize=12, y=0.992)

    # 左列 2 行均分整列高度；右列 n 行关节（避免与 7 行关节共用行高导致左列被压扁）
    if has_base:
        outer = gridspec.GridSpec(
            1,
            2,
            figure=fig,
            left=_GRID_LEFT,
            right=_GRID_RIGHT,
            bottom=_GRID_BOTTOM,
            top=_GRID_TOP,
            width_ratios=[1.15, 2.05],
            wspace=_GRID_WSPACE,
        )
        left_gs = outer[0].subgridspec(2, 1, hspace=_GRID_HSPACE)
        right_gs = outer[1].subgridspec(n, 2, hspace=_GRID_HSPACE, width_ratios=[1.0, 1.0])
    else:
        outer = gridspec.GridSpec(
            1,
            1,
            figure=fig,
            left=_GRID_LEFT,
            right=_GRID_RIGHT,
            bottom=_GRID_BOTTOM,
            top=_GRID_TOP,
        )
        right_gs = outer[0].subgridspec(n, 2, hspace=_GRID_HSPACE, width_ratios=[1.0, 1.0])

    ax_lin = ax_ang = None
    if has_base:
        b = log.base
        ax_lin = fig.add_subplot(left_gs[0, 0])
        ax_lin.plot(t, b["base_vx"], label="vx", color="tab:blue", linewidth=0.9)
        ax_lin.plot(t, b["base_vy"], label="vy", color="tab:green", linewidth=0.9)
        ax_lin.plot(t, b["base_vz"], label="vz", color="tab:red", linewidth=0.9)
        if "cmd_vx" in b:
            ax_lin.plot(t, b["cmd_vx"], label="cmd vx", color="tab:blue", linestyle=":", linewidth=0.8)
            ax_lin.plot(t, b["cmd_vy"], label="cmd vy", color="tab:green", linestyle=":", linewidth=0.8)
        ax_lin.set_ylabel("Lin [m/s]")
        ax_lin.set_title("Base linear vel (yaw)", fontsize=10, loc="left")
        ax_lin.legend(loc="upper right", fontsize=7)
        ax_lin.grid(True, alpha=0.3)
        auto_ylim(ax_lin, [b["base_vx"], b["base_vy"], b["base_vz"]], 0.1)

        ax_ang = fig.add_subplot(left_gs[1, 0], sharex=ax_lin)
        ax_ang.plot(t, b["base_wx"], label="wx", color="tab:blue", linewidth=0.9)
        ax_ang.plot(t, b["base_wy"], label="wy", color="tab:green", linewidth=0.9)
        ax_ang.plot(t, b["base_wz"], label="wz", color="tab:red", linewidth=0.9)
        if "cmd_wz" in b:
            ax_ang.plot(t, b["cmd_wz"], label="cmd wz", color="tab:orange", linestyle=":", linewidth=0.8)
        ax_ang.set_ylabel("Ang [rad/s]")
        ax_ang.set_title("Base angular vel (yaw)", fontsize=10, loc="left")
        ax_ang.legend(loc="upper right", fontsize=7)
        ax_ang.grid(True, alpha=0.3)
        auto_ylim(ax_ang, [b["base_wx"], b["base_wy"], b["base_wz"]], 0.1)
        plt.setp(ax_lin.get_xticklabels(), visible=False)
        ax_ang.set_xlabel(f"Time [s]  (dt≈{dt_ms:.1f} ms)")
        ax_ang.xaxis.set_major_locator(MaxNLocator(nbins=8))
    elif n > 0:
        fig.text(
            0.02,
            0.5,
            "No base velocity columns.\nRe-record with latest sim2sim --log-torques",
            fontsize=10,
            va="center",
        )

    ax_time_ref = ax_lin if ax_lin is not None else None
    ax_tau = None
    for i, (label, jname) in enumerate(plot_joints):
        d = log.joints[jname]
        if ax_time_ref is not None:
            ax_qd = fig.add_subplot(right_gs[i, 0], sharex=ax_time_ref)
            ax_tau = fig.add_subplot(right_gs[i, 1], sharex=ax_time_ref)
        else:
            ax_qd = fig.add_subplot(right_gs[i, 0])
            ax_tau = fig.add_subplot(right_gs[i, 1])
            if ax_time_ref is None and i == 0:
                ax_time_ref = ax_qd

        ax_qd.plot(t, d["qd"], color="tab:blue", linewidth=0.85, label="ω")
        ax_qd.plot(t, d["qd_min"], color="tab:blue", linestyle=":", linewidth=0.55, alpha=0.65)
        ax_qd.plot(t, d["qd_max"], color="tab:blue", linestyle=":", linewidth=0.55, alpha=0.65)
        ax_qd.set_ylabel("ω [rad/s]")
        ax_qd.set_title(f"{label}" + ("  (ω)" if i == 0 else ""), fontsize=9, loc="left")
        ax_qd.grid(True, alpha=0.3)
        auto_ylim(ax_qd, [d["qd"], d["qd_min"], d["qd_max"]], 0.5)

        ax_tau.plot(t, d["tau"], color="tab:red", linewidth=0.85, label="τ")
        ax_tau.plot(t, d["tau_min"], color="tab:red", linestyle=":", linewidth=0.55, alpha=0.65)
        ax_tau.plot(t, d["tau_max"], color="tab:red", linestyle=":", linewidth=0.55, alpha=0.65)
        ax_tau.set_ylabel("τ [N·m]")
        ax_tau.set_title(f"{label}" + ("  (τ)" if i == 0 else ""), fontsize=9, loc="left")
        ax_tau.grid(True, alpha=0.3)
        auto_ylim(ax_tau, [d["tau"], d["tau_min"], d["tau_max"]], 2.0)

        if i < n - 1:
            ax_qd.tick_params(axis="x", labelbottom=False)
            ax_tau.tick_params(axis="x", labelbottom=False)

    if ax_time_ref is not None:
        ax_time_ref.set_xlim(t0, t1)
    if ax_tau is not None and not has_base:
        ax_tau.set_xlabel(f"Time [s]  (dt≈{dt_ms:.1f} ms)")
        ax_tau.xaxis.set_major_locator(MaxNLocator(nbins=8))
    elif ax_tau is not None and has_base:
        plt.setp(ax_tau.get_xticklabels(), visible=False)

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved figure -> {save}")
    if show:
        _setup_maximize_on_first_draw(fig)
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=str, help="CSV from sim2sim_ultraYaw.py --log-torques")
    parser.add_argument("--all-joints", action="store_true", help="Plot all 13 joints")
    parser.add_argument("--save", type=str, default=None, help="Save figure path")
    parser.add_argument("--no-show", action="store_true", help="Only save (--save required)")
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.is_file():
        raise SystemExit(f"[ERROR] CSV not found: {csv_path}")

    log = load_sim2sim_csv(csv_path)
    plot_joints = build_plot_list(list(log.joints.keys()), args.all_joints)

    n_samples = len(log.t)
    duration = float(log.t[-1] - log.t[0]) if n_samples > 1 else 0.0
    print(f"[INFO] Loaded {csv_path}")
    print(f"       samples={n_samples}  duration={duration:.2f}s  joints={len(log.joints)}")
    print(f"       base_velocity(yaw)={'yes' if log.base else 'no (re-record CSV)'}")
    print(f"       plotting {len(plot_joints)} joint(s)")

    if args.no_show and not args.save:
        raise SystemExit("[ERROR] Use --save with --no-show")

    plot_sim2sim_log(
        log,
        plot_joints,
        title=f"sim2sim ultra_GameYaw — {csv_path.name}",
        save=args.save,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
