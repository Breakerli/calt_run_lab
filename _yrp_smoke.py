"""Smoke test for yrp03_v2_run env: verify joint order, dims, AMP, one step."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
from legged_lab.envs import *  # noqa: F401,F403,E402  (registers tasks)
from legged_lab.utils import task_registry  # noqa: E402

env_cfg, agent_cfg = task_registry.get_cfgs("yrp03_v2_run")
env_cfg.scene.num_envs = 16
env_cfg.scene.seed = agent_cfg.seed
env_class = task_registry.get_task_class("yrp03_v2_run")
env = env_class(env_cfg, True)

robot = env.robot
print("=== JOINT NAMES (Isaac internal order) ===")
for i, n in enumerate(robot.data.joint_names):
    print(f"{i:2d} {n}")
print("num_joints:", robot.num_joints)
print("num_actions:", env.num_actions)
print("left_leg_ids :", list(env.left_leg_ids), [robot.data.joint_names[i] for i in env.left_leg_ids])
print("right_leg_ids:", list(env.right_leg_ids), [robot.data.joint_names[i] for i in env.right_leg_ids])
print("ankle_roll_ids:", list(env.ankle_roll_ids), [robot.data.joint_names[i] for i in env.ankle_roll_ids])
print("hip_roll_ids:", list(env.hip_roll_ids), "hip_yaw_ids:", list(env.hip_yaw_ids), "waist_yaw_ids:", list(env.waist_yaw_ids))
print("feet bodies:", [robot.data.body_names[i] for i in env.feet_body_ids])
print("default_joint_pos[0]:", robot.data.default_joint_pos[0].tolist())

# AMP dims
amp = env.amp_loader_dict[2]
print("AMP loader obs_dim (style_2):", amp.observation_dim)
policy_amp = env.get_amp_obs_for_expert_trans()
print("policy AMP obs shape:", tuple(policy_amp.shape))
expert = env.get_amp_observations_pos()
print("expert AMP sample shape:", tuple(expert.shape))

# one step
actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
obs, rew, reset, extras = env.step(actions)
print("actor obs shape:", tuple(obs.shape), "expected actor_obs_dim:", env.actor_obs_dim)
print("critic obs shape:", tuple(extras["observations"]["critic"].shape), "expected critic_obs_dim:", env.critic_obs_dim)
print("reward finite:", bool(torch.isfinite(rew).all()))
print("SMOKE_OK")
simulation_app.close()
