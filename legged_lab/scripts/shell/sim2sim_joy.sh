#!/usr/bin/env bash
cd /home/bobby/run_ws/ultra_run_lab
/home/bobby/miniconda3/envs/isaaclab/bin/python legged_lab/scripts/sim2sim/sim2sim_joy.py \
  --policy /home/bobby/run_ws/ultra_run_lab/Exported_policy/policy.pt \
  --model /home/bobby/run_ws/ultra_run_lab/legged_lab/assets/ultra_GameYaw/mjcf/ultra_game_yaw.xml \
  "$@"
# 可选参数
# --plot_TN 绘制T-N曲线
# --explicit-actuator 使用explicit-actuator