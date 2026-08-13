# Ultra Run Lab (Isaac Lab + RSL-RL)

基于 [TienKung-Lab](https://github.com/Open-X-Humanoid/TienKung-Lab) / [Legged Lab](https://github.com/Hellod035/LeggedLab) 的腿足人形 RL 框架，当前主线为 **Ultra GameYaw** 高速跑步：**Multi-AMP (WGAN)** + **HIM 速度估计** + **速度课程** + **Sim2Sim (MuJoCo)** + **LCM 调试**。

[IsaacSim](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[Isaac Lab](https://isaac-sim.github.io/IsaacLab)
[RSL_RL](https://github.com/leggedrobotics/rsl_rl)
[License](https://opensource.org/licenses/BSD-3-Clause)

---

## Ultra GameYaw Run 流水线


| 阶段      | 脚本                    | 说明                                 |
| ------- | --------------------- | ---------------------------------- |
| 训练      | `train.py`            | Isaac Lab 4096 env，Multi-AMP + PPO |
| 回放      | `play.py`             | 加载 checkpoint，导出 `policy.pt`       |
| Sim2Sim | `sim2sim_ultraYaw.py` | MuJoCo 验证 TorchScript 策略           |
| 曲线      | `lcm_subscriber.py`   | 实时关节/速度/估计速度                       |


**任务名：** `ultra_game_yaw_run`  
**配置：** `legged_lab/envs/ultra_game_yaw/motion_run_cfg.py`  
**环境：** `legged_lab/envs/ultra_game_yaw/ultra_motion_run_env.py`  
**日志：** `logs/ultra_game_yaw_run/<timestamp>/`

### 特性摘要

- **机器人：** Ultra GameYaw，13 DoF（`legged_lab/assets/ultra_GameYaw/`）
- **仿真：** 物理 400 Hz（`dt=0.0025`, `decimation=4`），策略 100 Hz
- **策略：** HIM 估计线速度 + latent；`actor_recent_frames` 可配置（默认 5 帧进 actor MLP）
- **HIM 编码器：** `him_encoder_type` 可选 `mlp` / `gru`（默认 `gru`）
- **AMP：** 三风格 WGAN 判别器 — stand / walk / run（run 双专家轨迹）
- **命令：** 按 `|cmd_x|` 自动切 style；run 高速重采样 + `max_lin_accel` 斜坡
- **速度课程：** `lin_vel_x` 上限 5→15 m/s；跟踪门槛随速度线性降低（5 m/s→0.6，15 m/s→0.4）
- **域随机：** 摩擦/质量/惯量/COM、电机强度、PD 增益、关节位置 bias ±0.03、push（含 z/roll/pitch）、动作延迟
- **对称：** 数据增强 + mirror loss
- **部署导出：** `play.py` 导出 JIT `policy.pt` 并复制到 `Exported_policy/policy.pt`（供 sim2sim）

### Actor / Critic 观测（概要）

- **Actor（每步 48D × 历史 10）：** 角速度、重力、命令、关节 pos/vel、上步动作、步态相位等；**无**本体线速度（由 HIM 估计）
- **Critic（特权）：** 含真实线速度、域随机参数、足端力/高等（见 `CriticPrivilegeCfg`）

---

## 安装

需已安装 **Isaac Sim 5.x + Isaac Lab**（与本地 `isaaclab` 版本一致）。在**与 Isaac Lab 分离的目录**克隆本仓库：

```bash
cd ultra_run_2026lab
pip install -e .

cd rsl_rl
pip install -e .
```

验证（任选）：

```bash
python legged_lab/scripts/train.py --task=ultra_game_yaw_run --headless --logger=tensorboard --num_envs=64
```

---

## 训练

```bash
python legged_lab/scripts/train.py \
  --task=ultra_game_yaw_run \
  --headless \
  --logger=tensorboard \
  --num_envs=4096
```

多卡（示例）：

```bash
python -m torch.distributed.run --nproc_per_node=2 legged_lab/scripts/train.py \
  --task=ultra_game_yaw_run --headless --distributed
```

**Resume：** 在 `motion_run_cfg.py` → `UltraGameYawMotionRunAgentCfg` 中设置：

```python
resume = True
load_run = "2026-05-31_00-03-26"   # logs 下子目录名
load_checkpoint = "model_.*.pt"
```

**注意：** `him_encoder_type=gru` 与旧版 MLP checkpoint **不兼容**，换编码器需重新训或匹配 checkpoint。

**TensorBoard：**

```bash
tensorboard --logdir logs/ultra_game_yaw_run
```

常用标量：`Train/mean_reward`、`Curriculum/lin_vel_x_max`、`Curriculum/run_track_ok`、`Curriculum/curriculum_rwd_threshold`、`AMP/style_*_reward`。

---

## Play（Isaac Lab）

```bash
python legged_lab/scripts/play.py --task=ultra_game_yaw_run --num_envs=1
```

- 从 `logs/ultra_game_yaw_run/...` 加载 checkpoint（由 agent cfg 的 `load_run` / `load_checkpoint` 决定）
- 导出目录：`logs/ultra_game_yaw_run/<run>/exported/policy.pt`
- 自动复制：`Exported_policy/policy.pt`（sim2sim 默认路径）
- **键盘：** `F` 切换相机跟随/自由拖动
- **录像**（需 `pip install "gymnasium[other]"`）：

```bash
python legged_lab/scripts/play.py --task=ultra_game_yaw_run --num_envs=1 --video --video_length=200
```

输出：`logs/ultra_game_yaw_run/<run>/videos/play/rl-video-step-0.mp4`。`--video` 自动开启 `--enable_cameras`；**Ctrl+C** 会保存已录帧。

---

## Sim2Sim（MuJoCo）

先 `play.py` 生成 `Exported_policy/policy.pt`，再：

```bash
python legged_lab/scripts/sim2sim_ultraYaw.py
# 或指定策略
python legged_lab/scripts/sim2sim_ultraYaw.py --policy path/to/policy.pt
# 录像（与 play 相同，需 moviepy；按 policy 步抓帧，默认 100 Hz）
python legged_lab/scripts/sim2sim_ultraYaw.py --video --video_length=200
```

输出默认：`logs/sim2sim_ultra_yaw/videos/rl-video-step-0.mp4`；**Ctrl+C** 会保存已录帧。


| 按键         | 功能            |
| ---------- | ------------- |
| NumPad 8/2 | vx ±0.5       |
| NumPad 4/6 | vy ∓/±0.2     |
| NumPad 7/9 | yaw ±0.2      |
| p          | 手动切 style（调试） |
| BACKSPACE  | 重置            |
| ESC        | 退出            |


默认 MJCF：`legged_lab/assets/ultra_GameYaw/mjcf/ultra_game_yaw.xml`  
关节顺序与 Isaac **USD** 一致；若不一致，使用 `--isaac-joint-names-file`。

---

## LCM 实时曲线

**终端 1** — 仿真（play 或 sim2sim）  
**终端 2** — 订阅：

```bash
# sim2sim（默认）
python legged_lab/scripts/lcm_subscriber.py --source mujoco

# Isaac play
python legged_lab/scripts/lcm_subscriber.py --source isaaclab
```

`--show-est-vel` / `--no-show-est-vel` 控制 HIM 估计线速度曲线（play/sim2sim 默认开启）。

---

## AMP 专家数据


| 用途   | 目录                                                              | 字段                       |
| ---- | --------------------------------------------------------------- | ------------------------ |
| 训练专家 | `legged_lab/envs/ultra_game_yaw/datasets/motion_amp_expert/`    | dof_pos, dof_vel, EE pos |
| 动画预览 | `legged_lab/envs/ultra_game_yaw/datasets/motion_visualization/` | 含 root 位姿/速度             |


配置映射（`motion_run_cfg.py`）：

- `style_0` → stand  
- `style_1` → walk  
- `style_2` → run（`run_17200_9mps.txt`, `run_ultrayaw_14mps.txt`）

新增轨迹：按 GMR/转换流程生成 txt，写入上述目录并更新 `amp_motion_files_dict`。

---

## 配置调参入口


| 模块                       | 文件                                                        |
| ------------------------ | --------------------------------------------------------- |
| 环境、奖励、命令、课程、DR           | `legged_lab/envs/ultra_game_yaw/motion_run_cfg.py`        |
| PPO / AMP / HIM / resume | 同文件 `UltraGameYawMotionRunAgentCfg`                       |
| 课程逻辑                     | `ultra_motion_run_env.py` → `update_command_curriculum()` |
| HIM 导出                   | `legged_lab/utils/exporter.py` → `JitPolicyExporterHIM`   |
| WGAN 判别器                 | `rsl_rl/rsl_rl/modules/wgan_discriminator.py`             |


**速度课程（当前默认）：**

- 初始 `lin_vel_x` 上限见 `ranges.lin_vel_x[1]`（勿与 `curriculum_max` 设成相同，避免 resume 顶速）
- 升速条件：run env 平均 episode 长度 > 80% max，且跟踪 reward 超过随速度变化的门槛
- `curriculum_update_step=0.2` m/s

---

## 其他任务（TienKung / Ultra 旧任务）

仍可通过 `--task` 使用，例如：


| Task                                   | 说明             |
| -------------------------------------- | -------------- |
| `walk` / `run`                         | TienKung 单 AMP |
| `walk_with_sensor` / `run_with_sensor` | 带感知            |
| `ultra_motion_run`                     | Ultra JS WB    |
| `ultra88p_run`                         | Ultra88p       |


```bash
python legged_lab/scripts/train.py --task=walk --headless --num_envs=4096
python legged_lab/scripts/play.py --task=run --num_envs=1
python legged_lab/scripts/sim2sim.py --task walk --policy Exported_policy/walk.pt
```

### Motion Retargeting（TienKung）

使用 [GMR](https://github.com/YanjieZe/GMR) 重定向 SMPLX → 机器人，再经 `gmr_data_conversion.py` / `play_amp_animation.py` 生成 visualization 与 amp_expert 数据。详见原 TienKung 流程文档。

---

## Code formatting

```bash
pip install pre-commit
pre-commit run --all-files
```

---

## Acknowledgement

- [TienKung-Lab](https://github.com/Open-X-Humanoid/TienKung-Lab) / [Legged Lab](https://github.com/Hellod035/LeggedLab)
- [RSL RL](https://github.com/leggedrobotics/rsl_rl)
- [AMP_for_hardware](https://github.com/Alescontrela/AMP_for_hardware)
- [GMR](https://github.com/YanjieZe/GMR)
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)

## License

BSD-3-Clause（见各文件头声明）。