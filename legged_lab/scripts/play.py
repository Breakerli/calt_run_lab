# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

import argparse
import os
import shutil

import numpy as np
import torch
from isaaclab.app import AppLauncher

from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.utils import task_registry
from rsl_rl.runners import AmpOnPolicyRunner, MultiAmpOnPolicyRunner, OnPolicyRunner

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--video", action="store_true", default=False, help="Record viewport RGB to mp4 under log_dir/videos/play/.")
parser.add_argument(
    "--video_length",
    type=int,
    default=2000,
    help="Number of env steps to record when --video is set (default: 2000).",
)
parser.add_argument(
    "--log-joints",
    nargs="?",
    const="",
    default=None,
    help="Log joint state CSV (same format as sim2sim --log-torques). "
    "Optional path; default: <log_dir>/play_logs/joint_state.csv",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# Start camera rendering
if args_cli.task and "sensor" in args_cli.task:
    args_cli.enable_cameras = True
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs.ui import ViewportCameraController
from legged_lab.utils.play_viewport_camera import apply_viewport_camera_patch, setup_camera_follow_toggle

apply_viewport_camera_patch()
from legged_lab.utils.exporter import export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab.utils import math as math_utils

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg

from threading import Thread

from legged_lab.scripts.lcm_publisher import get_him_estimated_lin_vel, publish_robot_state
from legged_lab.utils.joint_state_log import PlayJointStateLogger
from legged_lab.utils.play_video import PlayVideoRecorder


# ---------------------------------------------------------------------------
# Joystick 开关（默认关）。改成 True 即可恢复手柄遥控；
# 主循环里是按 `joystick_opened` 判断是否覆盖 command 的，所以即便
# joystick_use=True 但手柄没插，也不会把 command 强制覆盖成 0，而是
# 回退到 env.command_generator 的 resample 结果（cfg 里的 ranges）。
# ---------------------------------------------------------------------------
joystick_use = False
joystick_opened = False
x_vel_cmd = y_vel_cmd = yaw_vel_cmd = 0.0

if joystick_use:
    import pygame

    pygame.init()
    try:
        # get joystick
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        joystick_opened = True
    except Exception as e:
        print(f"无法打开手柄：{e}")
    # joystick thread exit flag
    exit_flag = False

    def handle_joystick_input():
        global exit_flag, x_vel_cmd, y_vel_cmd, yaw_vel_cmd
        
        while not exit_flag:
            # get joystick input
            pygame.event.get()
            # update robot command
            x_vel_cmd = -joystick.get_axis(1) * 6
            y_vel_cmd = -joystick.get_axis(0) * 0.5
            yaw_vel_cmd = -joystick.get_axis(2) * 1
            pygame.time.delay(100)

    if joystick_opened:
        joystick_thread = Thread(target=handle_joystick_input)
        joystick_thread.start()

def play():
    runner: OnPolicyRunner
    env_cfg: BaseEnvCfg  # noqa:F405

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.domain_rand.events.randomize_rigid_body_inertia = None
    env_cfg.domain_rand.events.add_base_mass = None
    env_cfg.domain_rand.events.add_others_mass = None
    env_cfg.domain_rand.events.randomize_rigid_body_com = None
    env_cfg.domain_rand.events.randomize_joint_armature = None
    env_cfg.domain_rand.events.randomize_joint_friction = None
    env_cfg.domain_rand.events.randomize_actuator_friction = None
    env_cfg.domain_rand.events.randomize_motor_strength = None
    env_cfg.domain_rand.events.randomize_actuator_gains = None
    env_cfg.domain_rand.events.physics_material = None
    env_cfg.scene.max_episode_length_s = 1000.0
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 2.5
    env_cfg.commands.debug_vis = True
    env_cfg.commands.heading_command = False
    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.cmd_bucket_probs = None  # play 固定速度，不用分桶
    env_cfg.commands.use_curriculum = False
    env_cfg.commands.run_high_speed_sample_frac = None  # 否则 run 态 70% 重采样在 [7,10] 而非固定 10
    env_cfg.commands.resampling_time_range = (0.1, 0.1)
    env_cfg.commands.ranges.lin_vel_x = (12.0, 12.0)
    # env_cfg.commands.ranges.lin_vel_x = (1.0, 1.0) # (0.3, 0.3) # (4.5, 4.5)
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.ranges.heading = (0.0, 0.0)
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)
    # Inference / export: actor-only (no critic, no height/feet scan). sim2sim: num_obs_per_step=48.
    env_cfg.scene.height_scanner.enable_height_scan = False
    env_cfg.scene.height_scanner.enable_feet_height_scan = False

    env_cfg.scene.terrain_generator = None
    env_cfg.scene.terrain_type = "plane"

    env_cfg.sim.physx.gpu_max_rigid_patch_count = 2 * 2**15

    if env_cfg.scene.terrain_generator is not None:
        env_cfg.scene.terrain_generator.num_rows = 5
        env_cfg.scene.terrain_generator.num_cols = 5
        env_cfg.scene.terrain_generator.curriculum = False
        env_cfg.scene.terrain_generator.difficulty_range = (0.4, 0.4)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    def _sample_play_cmd_from_ranges():
        r = env_cfg.commands.ranges
        return (
            float(torch.empty(1).uniform_(r.lin_vel_x[0], r.lin_vel_x[1]).item()),
            float(torch.empty(1).uniform_(r.lin_vel_y[0], r.lin_vel_y[1]).item()),
            float(torch.empty(1).uniform_(r.ang_vel_z[0], r.ang_vel_z[1]).item()),
        )

    play_cmd = dict(zip(("vx", "vy", "wz"), _sample_play_cmd_from_ranges()))
    cmd_manual = [False]  # 键盘改过指令后置 True，之后不再每局重采样（list 包一层便于闭包修改）

    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)

    expected_actor_dim = getattr(env, "actor_obs_dim", None)
    if expected_actor_dim is not None:
        print(
            f"[Play] actor_obs_dim={expected_actor_dim} (history×per-step); "
            f"critic_obs_dim={getattr(env, 'critic_obs_dim', 'n/a')} (training only, not exported)"
        )

    camera = ViewportCameraController(env, env_cfg.viewer)  # noqa: F841

    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    print(f"[INFO] Loading checkpoint: {os.path.abspath(resume_path)}")

    runner_class: OnPolicyRunner | AmpOnPolicyRunner | MultiAmpOnPolicyRunner = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path)

    if expected_actor_dim is not None and getattr(runner, "empirical_normalization", False):
        norm = runner.obs_normalizer
        if hasattr(norm, "_mean"):
            norm_dim = int(norm._mean.shape[0])
        elif hasattr(norm, "mean"):
            norm_dim = int(norm.mean.shape[0])
        else:
            norm_dim = None
        if norm_dim is not None and norm_dim != expected_actor_dim:
            print(
                f"[WARN] obs normalizer dim ({norm_dim}) != env actor_obs_dim ({expected_actor_dim}). "
                "Checkpoint may be from a different obs layout; sim2sim/deploy can be wrong."
            )

    policy = runner.get_inference_policy(device=env.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(runner.alg.policy, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        runner.alg.policy, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )
    exported_pt_path = os.path.join(export_model_dir, "policy.pt")
    exported_onnx_path = os.path.join(export_model_dir, "policy.onnx")
    deploy_policy_dir = os.path.join(LEGGED_LAB_ROOT_DIR, "Exported_policy")
    os.makedirs(deploy_policy_dir, exist_ok=True)
    shutil.copy2(exported_pt_path, os.path.join(deploy_policy_dir, "policy.pt"))
    if os.path.isfile(exported_onnx_path):
        shutil.copy2(exported_onnx_path, os.path.join(deploy_policy_dir, "policy.onnx"))

    # Export the Isaac internal joint order (robot.data.joint_names). The policy obs
    # (joint_pos/vel) and actions are ordered by this internal order, which is only
    # known at runtime. sim2sim reads this file to reorder MuJoCo <-> Isaac correctly.
    try:
        isaac_joint_names = list(env.robot.data.joint_names)
        joint_names_txt = "\n".join(isaac_joint_names) + "\n"
        for _dst in (
            os.path.join(export_model_dir, "isaac_joint_names.txt"),
            os.path.join(deploy_policy_dir, "isaac_joint_names.txt"),
        ):
            with open(_dst, "w", encoding="utf-8") as _f:
                _f.write(joint_names_txt)
        print(
            f"[Play] Wrote Isaac joint order ({len(isaac_joint_names)} joints) -> "
            f"{deploy_policy_dir}/isaac_joint_names.txt (sim2sim joint mapping)."
        )
    except Exception as _e:  # noqa: BLE001
        print(f"[Play][WARN] Could not export isaac_joint_names.txt: {_e}")

    if expected_actor_dim is not None:
        print(
            f"[Play] Exported policy (obs history dim={expected_actor_dim}) -> {export_model_dir}. "
            "HIM: policy.onnx + policy.pt; critic/height-scan not exported."
        )
    print(f"[Play] Copied -> {deploy_policy_dir}/policy.onnx (sim2sim default), policy.pt")

    if not args_cli.headless:
        from legged_lab.utils.keyboard import Keyboard

        # --- 键盘实时调速：↑/↓ vx±0.5，←/→ wz±0.2；改过后不再每局重采样指令 ---
        CMD_LIMITS = {"vx": (-1.0, 15.0, 0.5), "wz": (-1.0, 1.0, 0.2)}

        def _bump_cmd(key, sign):
            lo, hi, step = CMD_LIMITS[key]
            play_cmd[key] = float(np.clip(play_cmd[key] + sign * step, lo, hi))
            cmd_manual[0] = True
            print(f"[Play] command -> vx={play_cmd['vx']:.2f} vy={play_cmd['vy']:.2f} wz={play_cmd['wz']:.2f}")

        keyboard = Keyboard(  # noqa:F841
            env,
            key_callbacks={
                "UP": lambda: _bump_cmd("vx", 1.0),
                "DOWN": lambda: _bump_cmd("vx", -1.0),
                "LEFT": lambda: _bump_cmd("wz", 1.0),
                "RIGHT": lambda: _bump_cmd("wz", -1.0),
            },
        )
        # 必须把订阅 handle 存到一个变量里，否则会被 GC 取消订阅。
        # 默认 follow=False：进 play 时鼠标可自由拖动，按 F 切到自动跟随。
        camera_follow_sub = setup_camera_follow_toggle(default_on=False)  # noqa:F841
        print("[Play] 键盘: ↑/↓ vx±0.5, ←/→ wz±0.2, R 重置, F 相机跟随")

    obs, _ = env.get_observations()

    # --- 速度指令 vs 实际质心速度打印（每 0.5s 一次，EMA 平滑） ---
    step_count = 0
    vel_ema = np.zeros(3)  # yaw 系实际速度 EMA: vx, vy, wz
    log_every = max(1, int(0.5 / float(env.step_dt)))

    video_recorder = None
    if args_cli.video:
        video_recorder = PlayVideoRecorder(env, log_dir, args_cli.video_length)

    joint_logger = None
    if args_cli.log_joints is not None:
        if args_cli.log_joints:
            joint_log_path = os.path.abspath(args_cli.log_joints)
        else:
            joint_log_path = os.path.join(log_dir, "play_logs", "joint_state.csv")
        joint_logger = PlayJointStateLogger(env, joint_log_path)
        print(f"[INFO] Joint state logging enabled (incremental flush) -> {joint_log_path}")

    try:
        if joint_logger is not None:
            joint_logger.record()

        while simulation_app.is_running():
            with torch.inference_mode():
                actions = policy(obs)
                # 仅当手柄真的打开时才覆盖 command；否则保留 env.command_generator
                # resample 出来的速度（cfg.commands.ranges 里设的范围）。
                if joystick_opened:
                    env.command_generator.command[:, 0] = x_vel_cmd
                    env.command_generator.command[:, 1] = y_vel_cmd
                    env.command_generator.command[:, 2] = yaw_vel_cmd
                obs, _, _, _ = env.step(actions)
                # 每步钉死；episode 首步按 ranges 重新 uniform 采样（键盘手动改过则不再重采样）
                if not joystick_opened:
                    if env.episode_length_buf[0] == 1 and not cmd_manual[0]:
                        play_cmd.update(zip(("vx", "vy", "wz"), _sample_play_cmd_from_ranges()))
                        print(f"[Play] resampled command (vx, vy, wz) = {tuple(play_cmd.values())}")
                    env.command_generator.vel_command_b[:, 0] = play_cmd["vx"]
                    env.command_generator.vel_command_b[:, 1] = play_cmd["vy"]
                    env.command_generator.vel_command_b[:, 2] = play_cmd["wz"]
                    if hasattr(env.command_generator, "cmd_target_b"):
                        env.command_generator.cmd_target_b[:, 0] = play_cmd["vx"]
                        env.command_generator.cmd_target_b[:, 1] = play_cmd["vy"]
                        env.command_generator.cmd_target_b[:, 2] = play_cmd["wz"]

            est_vel = None
            if getattr(runner.alg.policy, "estimator", None) is not None:
                est_vel = get_him_estimated_lin_vel(runner, obs)
            publish_robot_state(
                env.robot,
                command=env.command_generator.command,
                est_vel=est_vel,
                contact_sensor=getattr(env, "contact_sensor", None),
                feet_body_ids=getattr(env, "feet_body_ids", None),
                feet_sensor_ids=getattr(getattr(env, "feet_cfg", None), "body_ids", None),
                sim_time_s=float(env.episode_length_buf[0].item()) * float(env.step_dt),
            )

            # 打印速度指令 vs 实际质心速度（yaw 系，EMA 平滑）
            step_count += 1
            vel_yaw = math_utils.quat_apply_inverse(
                math_utils.yaw_quat(env.robot.data.root_quat_w[0:1]), env.robot.data.root_lin_vel_w[0:1]
            )[0]
            vel_ema = 0.9 * vel_ema + 0.1 * np.array(
                [float(vel_yaw[0]), float(vel_yaw[1]), float(env.robot.data.root_ang_vel_w[0, 2])]
            )
            if step_count % log_every == 0:
                print(
                    f"[Play] t={step_count * float(env.step_dt):7.2f}s | "
                    f"cmd(vx,vy,wz)=({play_cmd['vx']:5.2f},{play_cmd['vy']:5.2f},{play_cmd['wz']:5.2f}) | "
                    f"actual(vx,vy,wz)=({vel_ema[0]:5.2f},{vel_ema[1]:5.2f},{vel_ema[2]:5.2f}) | "
                    f"err_vx={play_cmd['vx'] - vel_ema[0]:5.2f}"
                )

            if joint_logger is not None:
                joint_logger.record()

            if video_recorder is not None and video_recorder.capture_step():
                break
    except KeyboardInterrupt:
        print("[INFO] Ctrl+C: saving recorded outputs (video / joint CSV)...")
        if joint_logger is not None:
            joint_logger.close()
        if video_recorder is not None:
            video_recorder.save()
    finally:
        if video_recorder is not None:
            video_recorder.save()
        if joint_logger is not None:
            joint_logger.close()


if __name__ == "__main__":
    try:
        play()
    except KeyboardInterrupt:
        print("[INFO] Interrupted.")
    except Exception as exc:
        print(f"[ERROR] play failed: {exc}")
        raise
    finally:
        simulation_app.close()
