"""Isaac Sim play 奔跑数据统计与报告生成脚本。

参照 legged_lab/scripts/play.py 的流程加载策略并钉死速度指令，在固定工况下奔跑，
按策略步（decimation 后）记录各关节的位置 q、转速 qd、输出转矩 tau，然后调用
legged_lab/utils/stat_report.py 统计并生成 docs/template_analysis.md 所需的全部
仿真侧信息（参数表、统计表、每类关节 4 张图、report.md、joint_log.csv）。

注意：模板中“理论转速/理论转矩/理论功率”列来自设计指标，仿真内无法获得，报告中留空待人工填写。

运行示例（headless 推荐）：
    python legged_lab/scripts/play_stat_report.py --task yrp03_v2_run --headless \
        --lin_vel_x 10.0 --warmup_steps 200 --record_steps 100
输出目录默认为 <checkpoint_run_dir>/play_report/，内含 report.md、joint_log.csv、
joint_params.csv 与 figs/*.png。
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

from isaaclab.utils import math as math_utils
from isaaclab_tasks.utils import get_checkpoint_path

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg
from legged_lab.utils.stat_report import PARAM_KEYS, generate_report, label_of
from rsl_rl.runners import AmpOnPolicyRunner, MultiAmpOnPolicyRunner, OnPolicyRunner


# --- Isaac 侧参数提取 ---

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
    params = extract_joint_params(env, joint_names)
    step_dt = float(env.step_dt)
    n_joints = len(joint_names)
    print(f"[INFO] joints={n_joints}, step_dt={step_dt * 1000:.1f} ms")

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
        vel_yaw = math_utils.quat_apply_inverse(
            math_utils.yaw_quat(env.robot.data.root_quat_w[0:1]), env.robot.data.root_lin_vel_w[0:1]
        )[0]
        rec_vx.append(float(vel_yaw[0]))

    if len(rec_t) < 10:
        raise RuntimeError(f"有效样本过少（{len(rec_t)}），无法统计。请检查策略/工况。")
    t = np.asarray(rec_t) - rec_t[0]
    q, qd, tau = np.asarray(rec_q), np.asarray(rec_qd), np.asarray(rec_tau)
    base_vx = np.asarray(rec_vx)

    # --- 报告模型信息 ---
    robot = env.robot
    try:
        total_mass = float(robot.root_physx_view.get_masses()[0].sum())
    except Exception:
        total_mass = float("nan")
    try:
        init_h = float(env_cfg.scene.robot.init_state.pos[2])
    except Exception:
        init_h = float("nan")
    from legged_lab.utils.stat_report import build_joint_index

    by_group, groups = build_joint_index(joint_names)
    use_cn = True
    default_pos = robot.data.default_joint_pos[0].detach().cpu().numpy()
    stance = [
        f"{label_of(e, use_cn)} {np.rad2deg(default_pos[e['idx']]):.1f}°"
        for gi in range(len(groups)) for e in by_group[gi]
        if abs(default_pos[e["idx"]]) > 1e-3
    ]
    n_obs_step = 9 + 3 * n_joints  # 角速度3+重力3+指令3+关节位置/速度/动作
    hist_len = getattr(getattr(env_cfg, "robot", None), "actor_obs_history_length", "?")
    model_rows = [
        ("关节自由度", str(n_joints), "受控关节数"),
        ("模型总质量", f"{total_mass:.2f} kg", "各刚体质量求和"),
        ("初始基座高度", f"{init_h:.2f} m", "资产 init_state"),
        ("默认站姿", "；".join(stance) if stance else "全 0°", "非零默认角关节"),
        ("动作维度", str(getattr(env, "num_actions", n_joints)), "策略输出维度"),
        ("单步观测维度", str(n_obs_step), f"历史 {hist_len} 步，Actor 输入 {getattr(env, 'actor_obs_dim', '?')}"),
        ("策略频率", f"{1.0 / step_dt:.0f} Hz", f"step_dt={step_dt * 1000:.1f} ms"),
    ]
    out_dir = args_cli.output_dir or os.path.join(log_dir, "play_report")
    info = {
        "source": f"Isaac Lab play（task={args_cli.task}）",
        "policy": resume_path,
        "cmd_vx": args_cli.lin_vel_x,
        "policy_hz": 1.0 / step_dt,
        "warmup_desc": f"热身 {args_cli.warmup_steps} 步",
        "model_rows": model_rows,
    }
    generate_report(out_dir, t, q, qd, tau, base_vx, joint_names, params, info)
    print(f"[INFO] Raw CSV -> {os.path.join(os.path.abspath(out_dir), 'joint_log.csv')} ({len(t)} samples, truncated={truncated})")


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
