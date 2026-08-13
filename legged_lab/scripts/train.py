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

from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry
from rsl_rl.runners import AmpOnPolicyRunner, MultiAmpOnPolicyRunner, OnPolicyRunner

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# Start camera rendering
if "sensor" in args_cli.task:
    args_cli.enable_cameras = True

import os
from datetime import datetime

import torch

if args_cli.distributed:
    _local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    _global_rank = int(os.environ.get("RANK", str(_local_rank)))
    _world_size = int(os.environ.get("WORLD_SIZE", "1"))
    from rsl_rl.utils import init_distributed_training

    init_distributed_training(_local_rank, _global_rank, _world_size)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def train():
    runner: OnPolicyRunner | AmpOnPolicyRunner | MultiAmpOnPolicyRunner

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)
    env_class = task_registry.get_task_class(env_class_name)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    if args_cli.distributed:
        _local_rank = int(app_launcher.local_rank)
        env_cfg.device = f"cuda:{_local_rank}"
        env_cfg.sim.device = f"cuda:{_local_rank}"
        agent_cfg.device = f"cuda:{_local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + _local_rank
        env_cfg.scene.seed = seed
        agent_cfg.seed = seed

    _rank = int(app_launcher.local_rank) if args_cli.distributed else 0
    env = env_class(env_cfg, args_cli.headless)

    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    if _rank == 0:
        print(f"[INFO] Logging experiment in directory: {log_root_path}", flush=True)

    if args_cli.distributed:
        from rsl_rl.utils import broadcast_log_dir_name

        log_dir = broadcast_log_dir_name(log_root_path, agent_cfg.run_name or "", _rank)
    else:
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, log_dir)

    runner_class: OnPolicyRunner | AmpOnPolicyRunner | MultiAmpOnPolicyRunner = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}", flush=True)
        runner.load(resume_path)
        print(
            f"[INFO]: Resume from iteration {runner.current_learning_iteration} "
            f"(actor_history={runner.actor_obs_history_length}, critic_history={runner.critic_obs_history_length})",
            flush=True,
        )

    if _rank == 0:
        os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    try:
        train()
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass
