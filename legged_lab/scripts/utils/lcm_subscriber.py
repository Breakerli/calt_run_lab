import argparse
import collections
import threading

import lcm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MaxNLocator

from isaaclab_lcm.robot_state_t import robot_state_t

# GridSpec 边距：右侧留空给 twinx 的 τ 轴刻度，避免被窗口裁切
_GRID_LEFT, _GRID_RIGHT = 0.06, 0.95
_GRID_BOTTOM, _GRID_TOP = 0.04, 0.98
_GRID_HSPACE, _GRID_WSPACE = 0.06, 0.26


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


# ===================== 配置 =====================
WINDOW_SIZE = 500

# ultra_GameYaw MuJoCo MJCF sensor / actuator order (sim2sim_ultraYaw.py MUJOCO_JOINT_NAMES)
ULTRA_GAME_YAW_MUJOCO_JOINT_NAMES = [
    "hip_yaw_l_joint",
    "hip_roll_l_joint",
    "hip_pitch_l_joint",
    "knee_pitch_l_joint",
    "ankle_pitch_l_joint",
    "hip_yaw_r_joint",
    "hip_roll_r_joint",
    "hip_pitch_r_joint",
    "knee_pitch_r_joint",
    "ankle_pitch_r_joint",
    "waist_yaw_joint",
    "shoulder_pitch_l_joint",
    "shoulder_pitch_r_joint",
]

# 左腿运动链 + 腰 + 左肩（显示顺序）
LEFT_CHAIN_JOINTS = [
    ("L Hip Yaw", "hip_yaw_l_joint"),
    ("L Hip Roll", "hip_roll_l_joint"),
    ("L Hip Pitch", "hip_pitch_l_joint"),
    ("L Knee Pitch", "knee_pitch_l_joint"),
    ("L Ankle Pitch", "ankle_pitch_l_joint"),
    ("Waist Yaw", "waist_yaw_joint"),
    ("L Shoulder Pitch", "shoulder_pitch_l_joint"),
]

# MuJoCo 无 joint_names 时的回退索引（与 ultra_game_yaw.xml / sim2sim 一致）
MUJOCO_JOINT_INDEX = {name: i for i, name in enumerate(ULTRA_GAME_YAW_MUJOCO_JOINT_NAMES)}


def build_joint_plot_list(source: str) -> list[tuple[str, str | None, int | None]]:
    """(display_label, joint_name, fixed_mujoco_idx)."""
    joints = []
    for label, jname in LEFT_CHAIN_JOINTS:
        mj_idx = MUJOCO_JOINT_INDEX.get(jname) if source == "mujoco" else None
        joints.append((label, jname, mj_idx))
    return joints


parser = argparse.ArgumentParser(description="LCM robot state plotter (Isaac play / sim2sim MuJoCo)")
parser.add_argument(
    "--source",
    choices=["isaaclab", "mujoco"],
    default="mujoco",
    help="isaaclab=play.py; mujoco=sim2sim_ultraYaw.py (default)",
)
parser.add_argument("--channel", default=None, help="Override LCM channel")
parser.add_argument(
    "--show-est-vel",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Plot HIM est lin vel (default: on for isaaclab, off for mujoco)",
)
args = parser.parse_args()

if args.source == "isaaclab":
    CHANNEL = "ISAACLAB_ROBOT_STATE"
    SHOW_FOOT_PLOTS = True
    _default_show_est = True
elif args.source == "mujoco":
    CHANNEL = "MUJOCO_ROBOT_STATE"
    SHOW_FOOT_PLOTS = False
    _default_show_est = True
else:
    CHANNEL = "ISAACLAB_ROBOT_STATE"
    SHOW_FOOT_PLOTS = True
    _default_show_est = True

SHOW_EST_VEL = _default_show_est if args.show_est_vel is None else args.show_est_vel

if args.channel is not None:
    CHANNEL = args.channel
    if args.channel == "ISAACLAB_ROBOT_STATE" and args.show_est_vel is None:
        SHOW_EST_VEL = True

JOINT_PLOTS = build_joint_plot_list(args.source)
NUM_RIGHT = len(JOINT_PLOTS)
NUM_LEFT = 4 + (2 if SHOW_FOOT_PLOTS else 0)  # 左列独立子网格行数（与右列关节行数解耦）

# ===================== 数据缓存 =====================
t_buf = collections.deque(maxlen=WINDOW_SIZE)
base_vel_x_buf = collections.deque(maxlen=WINDOW_SIZE)
base_vel_y_buf = collections.deque(maxlen=WINDOW_SIZE)
base_vel_z_buf = collections.deque(maxlen=WINDOW_SIZE)
cmd_vel_x_buf = collections.deque(maxlen=WINDOW_SIZE)
cmd_vel_y_buf = collections.deque(maxlen=WINDOW_SIZE)
cmd_vel_z_buf = collections.deque(maxlen=WINDOW_SIZE)
base_ang_x_buf = collections.deque(maxlen=WINDOW_SIZE)
base_ang_y_buf = collections.deque(maxlen=WINDOW_SIZE)
base_ang_z_buf = collections.deque(maxlen=WINDOW_SIZE)
est_lin_x_buf = collections.deque(maxlen=WINDOW_SIZE)
est_lin_y_buf = collections.deque(maxlen=WINDOW_SIZE)
est_lin_z_buf = collections.deque(maxlen=WINDOW_SIZE)
base_height_buf = collections.deque(maxlen=WINDOW_SIZE)
base_roll_buf = collections.deque(maxlen=WINDOW_SIZE)
base_pitch_buf = collections.deque(maxlen=WINDOW_SIZE)

foot_l_vz_buf = collections.deque(maxlen=WINDOW_SIZE)
foot_r_vz_buf = collections.deque(maxlen=WINDOW_SIZE)
foot_l_fz_buf = collections.deque(maxlen=WINDOW_SIZE)
foot_r_fz_buf = collections.deque(maxlen=WINDOW_SIZE)

joint_vel_buf = {label: collections.deque(maxlen=WINDOW_SIZE) for label, _, _ in JOINT_PLOTS}
joint_tau_buf = {label: collections.deque(maxlen=WINDOW_SIZE) for label, _, _ in JOINT_PLOTS}

latest_msg = None
lock = threading.Lock()
start_time = None
isaac_name_to_idx: dict[str, int] | None = None
mujoco_name_to_idx: dict[str, int] | None = None


def _is_real_joint_name(name: str) -> bool:
    return bool(name) and not name.startswith("joint_")


def resolve_joint_index(joint_name: str, mujoco_idx: int | None) -> int | None:
    """Map joint name to index in LCM joint_pos/joint_vel arrays."""
    global isaac_name_to_idx, mujoco_name_to_idx
    if args.source == "mujoco":
        if mujoco_name_to_idx is not None and joint_name in mujoco_name_to_idx:
            return mujoco_name_to_idx[joint_name]
        return mujoco_idx
    if isaac_name_to_idx is not None and joint_name in isaac_name_to_idx:
        return isaac_name_to_idx[joint_name]
    return None


def auto_ylim(ax, data, min_margin=1e-3, ratio=0.1):
    if len(data) < 2:
        return

    y_min = min(data)
    y_max = max(data)

    if y_min == y_max:
        margin = min_margin
    else:
        margin = max(min_margin, (y_max - y_min) * ratio)

    ax.set_ylim(y_min - margin, y_max + margin)


LEGEND_PICK_RADIUS = 5
_legend_pick_map: dict = {}


def enable_legend_pick_toggle(legend, orig_lines):
    """Click a legend entry to show/hide the corresponding curve."""
    if legend is None:
        return
    for leg_line, orig_line in zip(legend.get_lines(), orig_lines):
        leg_line.set_picker(LEGEND_PICK_RADIUS)
        _legend_pick_map[leg_line] = orig_line


def on_legend_pick(event):
    leg_line = event.artist
    orig_line = _legend_pick_map.get(leg_line)
    if orig_line is None:
        return
    visible = not orig_line.get_visible()
    orig_line.set_visible(visible)
    leg_line.set_alpha(1.0 if visible else 0.2)
    fig.canvas.draw_idle()


# ===================== LCM 回调 =====================
_joint_map_printed = False


def robot_state_handler(channel, data):
    global latest_msg, isaac_name_to_idx, mujoco_name_to_idx, _joint_map_printed
    msg = robot_state_t.decode(data)
    with lock:
        latest_msg = msg
        if hasattr(msg, "joint_names") and msg.joint_names and _is_real_joint_name(msg.joint_names[0]):
            name_map = {name: i for i, name in enumerate(msg.joint_names)}
            if args.source == "isaaclab":
                isaac_name_to_idx = name_map
            else:
                mujoco_name_to_idx = name_map
            if not _joint_map_printed:
                _joint_map_printed = True
                src = "Isaac" if args.source == "isaaclab" else "MuJoCo"
                print(f"[LCM] {src} joint index map ({CHANNEL}):")
                for label, jname, _ in JOINT_PLOTS:
                    idx = name_map.get(jname)
                    print(f"  {label:18s} {jname:24s} -> idx={idx}")


# ===================== LCM 线程 =====================
lc = lcm.LCM()
lc.subscribe(CHANNEL, robot_state_handler)


def lcm_loop():
    while True:
        lc.handle()


threading.Thread(target=lcm_loop, daemon=True).start()
print(f"[LCM] subscribed {CHANNEL}  (source={args.source}, show_est_vel={SHOW_EST_VEL})")

# ===================== Matplotlib =====================
plt.style.use("ggplot")
fig = plt.figure(figsize=_figure_fill_screen())
outer = gridspec.GridSpec(
    1,
    2,
    figure=fig,
    left=_GRID_LEFT,
    right=_GRID_RIGHT,
    bottom=_GRID_BOTTOM,
    top=_GRID_TOP,
    width_ratios=[1.0, 1.15],
    wspace=_GRID_WSPACE,
)
left_gs = outer[0].subgridspec(NUM_LEFT, 1, hspace=_GRID_HSPACE)
right_gs = outer[1].subgridspec(NUM_RIGHT, 1, hspace=_GRID_HSPACE)

# --- 左列：机身状态（NUM_LEFT 行均分整列高度）---
ax_lin = fig.add_subplot(left_gs[0, 0])
line_base_x, = ax_lin.plot([], [], label="Yaw vx [m/s]", color="tab:blue")
line_base_y, = ax_lin.plot([], [], label="Yaw vy [m/s]", color="tab:green")
line_base_z, = ax_lin.plot([], [], label="Yaw vz [m/s]", color="tab:red")
line_cmd_x, = ax_lin.plot([], [], label="Cmd vx", color="tab:blue", linestyle=":")
line_cmd_y, = ax_lin.plot([], [], label="Cmd vy", color="tab:green", linestyle=":")
line_est_x, = ax_lin.plot([], [], label="Est vx", color="tab:blue", linestyle="--")
line_est_y, = ax_lin.plot([], [], label="Est vy", color="tab:green", linestyle="--")
line_est_z, = ax_lin.plot([], [], label="Est vz", color="tab:red", linestyle="--")
if not SHOW_EST_VEL:
    for ln in (line_est_x, line_est_y, line_est_z):
        ln.set_visible(False)
ax_lin.set_ylabel("Lin vel [m/s]")
_title_src = "sim2sim yaw" if args.source == "mujoco" else "Isaac yaw"
ax_lin.set_title(f"线速度 ({_title_src})", fontsize=11, loc="left")
_lines_lin = [line_base_x, line_base_y, line_base_z, line_cmd_x, line_cmd_y]
if SHOW_EST_VEL:
    _lines_lin.extend([line_est_x, line_est_y, line_est_z])
leg_lin = ax_lin.legend(loc="upper right", fontsize=8)
enable_legend_pick_toggle(leg_lin, _lines_lin)
ax_lin.tick_params(labelsize=9)

ax_ang = fig.add_subplot(left_gs[1, 0], sharex=ax_lin)
line_ang_x, = ax_ang.plot([], [], label="Yaw wx [rad/s]", color="tab:blue")
line_ang_y, = ax_ang.plot([], [], label="Yaw wy [rad/s]", color="tab:green")
line_ang_z, = ax_ang.plot([], [], label="Yaw wz [rad/s]", color="tab:red")
line_cmd_wz, = ax_ang.plot([], [], label="Cmd wz", color="tab:orange", linestyle=":")
ax_ang.set_ylabel("Ang vel [rad/s]")
ax_ang.set_title("角速度", fontsize=11, loc="left")
_lines_ang = [line_ang_x, line_ang_y, line_ang_z, line_cmd_wz]
leg_ang = ax_ang.legend(loc="upper right", fontsize=8)
enable_legend_pick_toggle(leg_ang, _lines_ang)
ax_ang.tick_params(labelsize=9)
plt.setp(ax_lin.get_xticklabels(), visible=False)
plt.setp(ax_ang.get_xticklabels(), visible=False)

ax_height = fig.add_subplot(left_gs[2, 0], sharex=ax_lin)
line_base_height, = ax_height.plot([], [], label="Base z [m]", color="tab:blue")
ax_height.set_ylabel("Height [m]")
ax_height.set_title("高度", fontsize=11, loc="left")
leg_height = ax_height.legend(loc="upper right", fontsize=8)
enable_legend_pick_toggle(leg_height, [line_base_height])
ax_height.tick_params(labelsize=9)
plt.setp(ax_height.get_xticklabels(), visible=False)

ax_rpy = fig.add_subplot(left_gs[3, 0], sharex=ax_lin)
line_roll, = ax_rpy.plot([], [], label="Roll [rad]", color="tab:blue")
line_pitch, = ax_rpy.plot([], [], label="Pitch [rad]", color="tab:green")
ax_rpy.set_ylabel("RP [rad]")
ax_rpy.set_title("姿态", fontsize=11, loc="left")
leg_rpy = ax_rpy.legend(loc="upper right", fontsize=8)
enable_legend_pick_toggle(leg_rpy, [line_roll, line_pitch])
ax_rpy.tick_params(labelsize=9)
plt.setp(ax_rpy.get_xticklabels(), visible=False)

ax_foot_vel = None
ax_foot_force = None
line_foot_l_vz = line_foot_r_vz = line_foot_l_fz = line_foot_r_fz = None

if SHOW_FOOT_PLOTS:
    ax_foot_vel = fig.add_subplot(left_gs[4, 0], sharex=ax_lin)
    line_foot_l_vz, = ax_foot_vel.plot([], [], label="L vz [m/s]", color="tab:blue")
    line_foot_r_vz, = ax_foot_vel.plot([], [], label="R vz [m/s]", color="tab:orange")
    ax_foot_vel.set_ylabel("vz [m/s]")
    ax_foot_vel.set_title("脚 vz", fontsize=11, loc="left")
    leg_foot_vel = ax_foot_vel.legend(loc="upper right", fontsize=8)
    enable_legend_pick_toggle(leg_foot_vel, [line_foot_l_vz, line_foot_r_vz])
    ax_foot_vel.tick_params(labelsize=9)
    plt.setp(ax_foot_vel.get_xticklabels(), visible=False)

    ax_foot_force = fig.add_subplot(left_gs[5, 0], sharex=ax_lin)
    line_foot_l_fz, = ax_foot_force.plot([], [], label="L Fz [N]", color="tab:blue")
    line_foot_r_fz, = ax_foot_force.plot([], [], label="R Fz [N]", color="tab:orange")
    ax_foot_force.set_ylabel("Fz [N]")
    ax_foot_force.set_title("足底 Fz", fontsize=11, loc="left")
    leg_foot_fz = ax_foot_force.legend(loc="upper right", fontsize=8)
    enable_legend_pick_toggle(leg_foot_fz, [line_foot_l_fz, line_foot_r_fz])
    ax_foot_force.tick_params(labelsize=9)

# --- 右列：关节（NUM_RIGHT 行铺满右列高度）---
joint_lines = {}

for i, (label, _jname, _mj_idx) in enumerate(JOINT_PLOTS):
    ax = fig.add_subplot(right_gs[i, 0], sharex=ax_lin)
    ax_tau = ax.twinx()
    line_vel, = ax.plot([], [], label="w", color="tab:blue")
    line_tau, = ax_tau.plot([], [], label="tau", color="tab:red")
    ax.set_ylabel("w [rad/s]")
    ax_tau.set_ylabel("tau [Nm]")
    ax.set_title(label, fontsize=9, loc="left")
    lines_joint = [line_vel, line_tau]
    leg_joint = ax.legend(lines_joint, [l.get_label() for l in lines_joint], loc="upper left", fontsize=6, framealpha=0.7)
    enable_legend_pick_toggle(leg_joint, lines_joint)
    if i < len(JOINT_PLOTS) - 1:
        ax.tick_params(axis="x", labelbottom=False)
    ax_tau.yaxis.tick_right()
    ax_tau.yaxis.set_label_position("right")
    ax_tau.tick_params(axis="y", labelsize=8, pad=2)
    joint_lines[label] = (ax, ax_tau, line_vel, line_tau)

# 左/右列仅最底行显示时间轴刻度（sharex 保持对齐）
_left_axes = [ax_lin, ax_ang, ax_height, ax_rpy]
if SHOW_FOOT_PLOTS and ax_foot_vel is not None and ax_foot_force is not None:
    _left_axes.extend([ax_foot_vel, ax_foot_force])
_left_bottom_ax = _left_axes[-1]
_left_bottom_ax.set_xlabel("Sim time [s]")
_left_bottom_ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
for _ax in _left_axes[:-1]:
    plt.setp(_ax.get_xticklabels(), visible=False)

_right_axes = [joint_lines[label][0] for label, _, _ in JOINT_PLOTS]
_right_bottom_ax = _right_axes[-1]
_right_bottom_ax.set_xlabel("Sim time [s]")
_right_bottom_ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
for _ax in _right_axes[:-1]:
    plt.setp(_ax.get_xticklabels(), visible=False)


# ===================== 动画更新 =====================
def update(frame):
    global start_time

    with lock:
        msg = latest_msg

    if msg is None:
        return

    if start_time is None:
        start_time = msg.timestamp

    t = (msg.timestamp - start_time) * 1e-9

    t_buf.append(t)
    base_vel_x_buf.append(msg.base_lin_vel[0])
    base_vel_y_buf.append(msg.base_lin_vel[1])
    base_vel_z_buf.append(msg.base_lin_vel[2])
    cmd_vel_x_buf.append(getattr(msg, "cmd_vel", [0.0, 0.0, 0.0])[0])
    cmd_vel_y_buf.append(getattr(msg, "cmd_vel", [0.0, 0.0, 0.0])[1])
    cmd_vel_z_buf.append(getattr(msg, "cmd_vel", [0.0, 0.0, 0.0])[2])
    base_ang_x_buf.append(msg.base_ang_vel[0])
    base_ang_y_buf.append(msg.base_ang_vel[1])
    base_ang_z_buf.append(msg.base_ang_vel[2])
    est_lin_x_buf.append(float(msg.base_lin_vel_est[0]))
    est_lin_y_buf.append(float(msg.base_lin_vel_est[1]))
    est_lin_z_buf.append(float(msg.base_lin_vel_est[2]))
    base_height_buf.append(float(msg.base_height[0]))

    rpy = getattr(msg, "base_rpy", [0.0, 0.0, 0.0])
    base_roll_buf.append(float(rpy[0]))
    base_pitch_buf.append(float(rpy[1]))

    if SHOW_FOOT_PLOTS:
        foot_vel = getattr(msg, "foot_lin_vel", [0.0] * 6)
        foot_frc = getattr(msg, "foot_contact_force", [0.0] * 6)
        foot_l_vz_buf.append(float(foot_vel[2]))
        foot_r_vz_buf.append(float(foot_vel[5]))
        foot_l_fz_buf.append(float(foot_frc[2]))
        foot_r_fz_buf.append(float(foot_frc[5]))

    for label, jname, mj_idx in JOINT_PLOTS:
        idx = resolve_joint_index(jname, mj_idx)
        if idx is not None and idx < msg.num_joints:
            joint_vel_buf[label].append(msg.joint_vel[idx])
            joint_tau_buf[label].append(msg.joint_torque[idx])
        else:
            joint_vel_buf[label].append(float("nan"))
            joint_tau_buf[label].append(float("nan"))

    t_list = list(t_buf)
    if len(t_list) == 0:
        return

    t0, t1 = t_list[0], t_list[-1]

    line_base_x.set_data(t_list, list(base_vel_x_buf))
    line_base_y.set_data(t_list, list(base_vel_y_buf))
    line_base_z.set_data(t_list, list(base_vel_z_buf))
    line_cmd_x.set_data(t_list, list(cmd_vel_x_buf))
    line_cmd_y.set_data(t_list, list(cmd_vel_y_buf))
    line_est_x.set_data(t_list, list(est_lin_x_buf))
    line_est_y.set_data(t_list, list(est_lin_y_buf))
    line_est_z.set_data(t_list, list(est_lin_z_buf))

    _left_axes = [ax_lin, ax_ang, ax_height, ax_rpy]
    if SHOW_FOOT_PLOTS and ax_foot_vel is not None and ax_foot_force is not None:
        _left_axes.extend([ax_foot_vel, ax_foot_force])
    for _ax in _left_axes:
        _ax.set_xlim(t0, t1)

    _ylim_lin = (
        list(base_vel_x_buf)
        + list(base_vel_y_buf)
        + list(base_vel_z_buf)
        + list(cmd_vel_x_buf)
        + list(cmd_vel_y_buf)
    )
    if SHOW_EST_VEL:
        _ylim_lin += list(est_lin_x_buf) + list(est_lin_y_buf) + list(est_lin_z_buf)
    auto_ylim(ax_lin, _ylim_lin, min_margin=0.1)

    line_ang_x.set_data(t_list, list(base_ang_x_buf))
    line_ang_y.set_data(t_list, list(base_ang_y_buf))
    line_ang_z.set_data(t_list, list(base_ang_z_buf))
    line_cmd_wz.set_data(t_list, list(cmd_vel_z_buf))
    auto_ylim(
        ax_ang,
        list(base_ang_x_buf) + list(base_ang_y_buf) + list(base_ang_z_buf) + list(cmd_vel_z_buf),
        min_margin=0.1,
    )

    line_base_height.set_data(t_list, list(base_height_buf))
    auto_ylim(ax_height, base_height_buf, min_margin=0.05)

    line_roll.set_data(t_list, list(base_roll_buf))
    line_pitch.set_data(t_list, list(base_pitch_buf))
    auto_ylim(ax_rpy, list(base_roll_buf) + list(base_pitch_buf), min_margin=0.05)

    if SHOW_FOOT_PLOTS and line_foot_l_vz is not None:
        line_foot_l_vz.set_data(t_list, list(foot_l_vz_buf))
        line_foot_r_vz.set_data(t_list, list(foot_r_vz_buf))
        auto_ylim(ax_foot_vel, list(foot_l_vz_buf) + list(foot_r_vz_buf), min_margin=0.1)
        line_foot_l_fz.set_data(t_list, list(foot_l_fz_buf))
        line_foot_r_fz.set_data(t_list, list(foot_r_fz_buf))
        auto_ylim(ax_foot_force, list(foot_l_fz_buf) + list(foot_r_fz_buf), min_margin=10.0)

    for label in joint_lines:
        ax, ax_tau, line_vel, line_tau = joint_lines[label]
        vel_data = joint_vel_buf[label]
        tau_data = joint_tau_buf[label]
        line_vel.set_data(t_list, list(vel_data))
        line_tau.set_data(t_list, list(tau_data))
        ax.set_xlim(t0, t1)
        auto_ylim(ax, [v for v in vel_data if not np.isnan(v)], min_margin=0.5)
        auto_ylim(ax_tau, [v for v in tau_data if not np.isnan(v)], min_margin=2.0)


# ===================== 启动 =====================
fig.canvas.mpl_connect("pick_event", on_legend_pick)
_setup_maximize_on_first_draw(fig)
ani = FuncAnimation(fig, update, interval=50)
plt.show()
