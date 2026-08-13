#!/usr/bin/env bash
cd /home/bobby/run_ws/ultra_run_lab
/home/bobby/miniconda3/envs/isaaclab/bin/python legged_lab/scripts/sim2sim_ultraYaw.py "$@" --model /home/bobby/run_ws/ultra_run_lab/legged_lab/assets/ultra_GameYaw/mjcf/ultra_game_yaw.xml