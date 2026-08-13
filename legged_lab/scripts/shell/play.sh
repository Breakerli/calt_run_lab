#!/usr/bin/env bash
cd /home/bobby/run_ws/ultra_run_lab
/home/bobby/miniconda3/envs/isaaclab/bin/python legged_lab/scripts/play.py --task=ultra_game_yaw_run --num_envs=1 "$@"
# /home/bobby/miniconda3/envs/isaaclab/bin/python legged_lab/scripts/play.py --task=ultra_game_yaw_run --headless --num_envs=1 --load_run 2026-06-03_22-16-39  --checkpoint model_15000.pt