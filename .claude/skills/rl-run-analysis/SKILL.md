---
name: rl-run-analysis
description: 分析 ultra_run_lab 机器人大步幅跑步训练（MULTIAMPPPO + WGAN AMP + 速度课程）的 tensorboard 日志，定位课程学习/奖励/判别器/收敛问题并给出"又稳又快"的改进思路。当用户要求"分析试验结果/tensorboard/奖励曲线/课程学习/判别器学习"时使用。
---

# RL 跑步试验分析（tensorboard → 诊断 → 改进）

面向赛跑赛事目标：**又稳又快**，重点关注步频（cadence）与步幅（stride）。

## 前置：环境与日志

- 用 `ultra_run` conda 环境解析 tensorboard（有 `tensorboard/numpy/matplotlib`，无 pandas/tbparse/tf）：
  `source /home/robot02/miniconda3/bin/activate ultra_run`
- 日志在 `logs/<experiment>/<run_time>/events.out.tfevents.*`，同目录有 `params/env.yaml`、`params/agent.yaml`、`git/*.diff`。
- 一次"完整"试验 = 有多个 `model_*.pt` 的那个 run（体积最大）。

## 解析事件文件（唯一可靠来源）

```python
from tensorboard.backend.event_processing import event_accumulator
ea = event_accumulator.EventAccumulator(path, size_guidance={event_accumulator.SCALARS: 0})
ea.Reload()
tags = ea.Tags()['scalars']          # 74 个标量 tag
ev = ea.Scalars(tag)                 # .step / .value
```

把全部序列 dump 成 `scalars.json`，再画平滑曲线（`np.convolve`，窗口 ~101）。

## 指标分组与判读（本代码库 MULTIAMPPPO + WGAN）

| 分组 | tag | 健康信号 | 异常信号 |
|---|---|---|---|
| 总奖励 | `Train/mean_reward` | 单调上升不崩 | 发散/崩溃/骤降 |
| 稳定性 | `Train/mean_episode_length`（满 2000=20s） | 收敛到接近满值 | 长期偏低 |
| 课程 | `Curriculum/lin_vel_x_max` | 阶梯式升到上限 15 | **全程不变 = 卡死** |
| 课程门槛 | `Curriculum/run_track_lin_vel_x_exp` vs `run_track_gate` | track > gate | track < gate |
| 课程门槛 | `Curriculum/run_mean_episode_length` vs gate 1600 | >1600 | <1600 |
| 判别器 | `Loss/amp`（= -(expert−policy)） | 温和下降 | 越负越强，追不上 = style 退化 |
| 判别器 | `Loss/amp_policy_pred` / `Loss/amp_expert_pred` | 差距小 | 差距越拉越大（critic 碾压） |
| 判别器 | `Loss/amp_grad_pen` | 收敛到小值(~1) | 爆大 |
| 收敛 | `Policy/mean_noise_std` / `Loss/entropy` / `Loss/learning_rate` | std↓、entropy↓ | std 高(>0.5)未收敛 |

## 关键公式（解读判别器/课程必须会）

- **WGAN style 奖励**（`wgan_discriminator.py:165`）：
  `score = (D(policy) − EMA[D(expert)]).clamp(max=0)`，`style_r = coef · exp(score/10)`。
  gap 越大 → style_r 越小；gap≈10 时 style_r≈coef·0.36。`amp_reward_coef_dict` 里 run=0.6。
- **总奖励**（`task_reward_lerp=0.8`）：`r = 0.2·style_r + 0.8·task_r` → 任务奖励主导。
- **run 跟踪奖励**（`rewards.py:92`）：`15·exp(-|err|/std²)`，std 随速度 smoothstep 从 0.6 放宽到 1.0。
  由终值反推误差：`err = -std²·ln(reward/15)`。
- **课程门槛**（`yrp03_motion_run_env.py:993`）：`rwd_ratio` 在 [5,15] m/s 从 0.43 线性降到 0.35，
  `gate = rwd_ratio × 15`。升速需同时满足 `mean_ep_len>1600` 且 `run_track>gate`。

## 诊断流程（按顺序）

1. **稳不稳**：`mean_episode_length` 是否接近满值、`termination_penalty` 是否≈0。
2. **快不快**：`lin_vel_x_max` 是否在涨；若卡死，看 `run_track` vs `gate` 哪个没过 → 定位是"跟踪精度"还是"稳定性"瓶颈。
3. **判别器是否健康**：expert/policy gap 是否持续拉大导致 style 奖励退化（36% 饱和）。
4. **是否收敛**：noise_std / entropy / learning_rate。
5. **步频/步幅**：当前**没有直接指标**，只能从 `gait_feet_distance`/`knee_y_distance`/`feet_collision`/`feet_slide` 间接看；要显式埋点。

## 改进思路模板（面向"又稳又快"）

- P0 解除课程卡点：降 `curriculum_rwd_threshold`、放宽 run 跟踪 `std`、拉长课程评价窗口。
- P1 让 AMP 风格生效：提高 run `amp_reward_coef`、降 `amp_task_reward_lerp`、判别器正则（grad_pen 权重/限制 critic 更新步数/one-sided smoothing）。
- P2 显式优化步频步幅：新增 `stride_length`/`cadence`/`flight_ratio` 埋点与奖励；用专家 run 运动数据（`run_17200_9mps`、`vxp15`）标定目标。
- P3 收敛/稳定：继续训练、noise 衰减、关注 `feet_slide`（随速度变大）。

## 参考实现

本 skill 的完整示例见一次真实分析：
`logs/yrp03_v2_run/2026-08-14_21-01-35/analysis/README.md`（结论 + 6 张图 + `scalars.json`）。
