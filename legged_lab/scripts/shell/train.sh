#!/usr/bin/env bash
cd /home/bobby/run_ws/ultra_run_lab
/home/bobby/miniconda3/envs/isaaclab/bin/python legged_lab/scripts/train.py --task=ultra_game_yaw_run --logger=tensorboard --headless --num_envs=4096 "$@"