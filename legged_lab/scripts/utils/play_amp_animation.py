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

from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip
import numpy as np

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--save_path", type=str, default=None, help="Path to save the txt file")
parser.add_argument("--fps", type=float, default=30.0, help="Target fps")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# Start camera rendering
if "sensor" in args_cli.task:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg
from isaaclab.envs.ui import ViewportCameraController


from legged_lab.utils.amp_motion_io import save_amp_motion_txt  # noqa: E402


def play_amp_animation():
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
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 2.5
    env_cfg.scene.terrain_generator = None
    env_cfg.scene.terrain_type = "plane"
    env_cfg.commands.debug_vis = False

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)

    # visualize_motion() 走的是直接 write_data + sim.step 的路径，articulation 的
    # root_physx_view 在初始化时会有短暂为 None 的窗口，若 viewer 用 origin_type="asset_body"
    # 跟随 pelvis，会让 ViewportCameraController 的 stage callback 反复访问 body_names 而刷屏报错。
    # 此处覆盖为世界系固定视角，对动画播放与导出 txt 无影响，只影响相机跟随。
    env_cfg.viewer.origin_type = "world"
    env_cfg.viewer.eye = (5.0, 0.0, 2.0)
    env_cfg.viewer.lookat = (0.0, 0.0, 1.0)
    camera = ViewportCameraController(env, env_cfg.viewer)  # noqa:F841

    num_frames = int(env.motion_len)
    frame_duration = 1.0 / args_cli.fps
    all_frames = []

    export_mode = args_cli.save_path is not None
    if export_mode:
        save_dir = os.path.dirname(os.path.abspath(args_cli.save_path))
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        print(
            f"Export mode: {num_frames - 1} frames -> {args_cli.save_path}\n"
            f"  Note: --fps only sets output FrameDuration, not export frame count."
        )

    frame_cnt = 0
    while simulation_app.is_running():
        if export_mode:
            frame = env.visualize_motion(frame_idx=frame_cnt)
            all_frames.append(frame.detach().cpu().numpy().reshape(-1))
            if frame_cnt % 500 == 0:
                print(f"  export progress: {frame_cnt}/{num_frames - 1}")
        else:
            time = frame_cnt * frame_duration
            env.visualize_motion(time=time)
        frame_cnt += 1
        if frame_cnt >= num_frames - 1:
            break

    if export_mode:
        if len(all_frames) == 0:
            raise RuntimeError("No frames collected; did you interrupt (Ctrl+C) before export finished?")
        print(f"Playback/collection finished ({len(all_frames)} frames). Stacking arrays...")
        all_frames_np = np.stack(all_frames, axis=0)
        save_amp_motion_txt(args_cli.save_path, all_frames_np, frame_duration)
        print(f"✅ Successfully saved to {args_cli.save_path}")
if __name__ == "__main__":
    play_amp_animation()
    simulation_app.close()
