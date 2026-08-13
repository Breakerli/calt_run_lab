# UltraGameYaw Reward 解读

本文解读当前任务里启用的 reward 配置，主要对应：

- [`motion_run_cfg.py`](motion_run_cfg.py) 中的 `UltraGameYawRewardCfg`
- [`rewards.py`](rewards.py) 中的 ultra 专属 reward 占位说明
- [`../../mdp/rewards.py`](../../mdp/rewards.py) 中实际被 `mdp.<func>` 调用的共享 reward 实现

`ultra_game_yaw/rewards.py` 目前没有定义具体 reward 函数。这个文件只是说明：只有当 reward 依赖本机器人特殊关节或 link 布局、无法通过 `SceneEntityCfg` 参数化时，才应把函数放到这里。当前配置里所有启用项都通过 `import legged_lab.mdp as mdp` 调到了共享 mdp reward。

## 1. 整体结构

`UltraGameYawMotionRunEnvCfg.reward = UltraGameYawRewardCfg()`，环境每步通过 `RewardManager` 计算这些 `RewTerm`。每个条目可以理解成：

```text
term_reward = weight * func(env, **params)
```

通常 `func` 返回非负分数或非负 penalty 值：

- `weight > 0`：鼓励项，函数值越大越好。
- `weight < 0`：惩罚项，函数值越大扣分越多。
- `style_mask`：按 `env.style_ids` 屏蔽不适用的 gait style，不匹配时该项输出为 0。

当前 style 约定：

| style id | 名称 | 典型命令区间 |
|---:|---|---|
| 0 | stand | `|vx|, |vy|, |yaw|` 都很小 |
| 1 | walk | 非站立，且 `|vx|` 未进入 run |
| 2 | run | `|vx|` 足够大 |

style 由 `ultra_motion_run_env.py::_update_style_from_command()` 根据命令自动更新：

- 进入站立：`|vx|, |vy|, |yaw| < stand_enter = 0.05`
- 离开站立：任一命令分量达到 `stand_exit = 0.08`
- 进入跑步：`|vx| >= run_enter = 1.0`
- 从跑步退回走路：`|vx| < run_exit = 0.85`

需要特别注意平均速度窗口：配置里一些 reward 传了 `avg_vel_window_s=0.6`，但在当前 `UltraMotionRunEnv` 里，`mdp.rewards` 会优先使用环境维护的 `avg_lin_vel_x_yaw`、`avg_lin_vel_y_yaw` 和 `avg_ang_vel_z`。实际 EMA 窗口由 `UltraGameYawMotionRunEnvCfg.track_avg_vel_window_s = 0.5` 控制。也就是说，这里的 `avg_vel_window_s` 更像“启用平均速度”的开关，实际窗口看 env cfg。

## 2. 生存和站立

| term | 函数 | 权重 | style | 逻辑 | 训练意图 |
|---|---|---:|---|---|---|
| `stand_still` | `stand_still` | `10.0` | `[0]` | `exp(-sum((q - target_angles)^2))` | 站立 style 下，把 13 个关节拉回指定静态姿态。 |
| `alive` | `is_alive` | `10.0` | 全部 | `(~reset_buf) * episode_progress` | 活得越久奖励越高，后半段 episode 的 alive 权重更明显。 |
| `termination_penalty` | `is_terminated` | `-200.0` | 全部 | `reset_buf & ~time_out_buf` | 非超时终止重罚，主要压摔倒、非法接触等提前 reset。 |

`stand_still` 只在 `style_id=0` 生效。它不是简单保持默认关节，而是使用配置里的 `target_angles`，其中膝、踝给了轻微弯曲姿态。

## 3. 速度跟踪

这些项让机器人按照命令 `[vx, vy, yaw_rate]` 行走或奔跑。线速度会转到 yaw frame，即只去掉朝向 yaw，避免 roll/pitch 干扰前后和横向速度解释。

| term | 函数 | 权重 | style | 逻辑 | 训练意图 |
|---|---|---:|---|---|---|
| `track_lin_vel_x_walk_exp` | `track_lin_vel_x_yaw_frame_exp` | `5.0` | `[1]` | `exp(-abs(cmd_x - avg_vx_yaw) / std^2)`，`std=0.3` | walk 下跟踪前向速度，使用平均速度抑制抖动。 |
| `track_lin_vel_x_run_exp` | `track_lin_vel_x_yaw_frame_exp` | `15.0` | `[2]` | `exp(-abs(cmd_x - vx_yaw) / std_eff^2)` | run 下更强地跟踪前向速度，`std_eff` 随高速从 `0.6` 放宽到 `1.0`。 |
| `track_lin_vel_y_exp` | `track_lin_vel_y_yaw_frame_exp` | `5.0` | `[1, 2]` | `exp(-abs(cmd_y - avg_vy_yaw) / 0.3^2)` | 走和跑都要跟踪横向速度。 |
| `track_lin_vel_z_exp` | `track_lin_vel_z_yaw_frame_exp` | `0.05` | `[0, 1, 2]` | `exp(-abs(vz_yaw) / 0.3^2)` | 很弱地奖励低竖直速度，减少上下弹跳。 |
| `track_ang_vel_z_exp` | `track_ang_vel_z_world_exp` | `5.0` | `[1, 2]` | `exp(-(cmd_yaw - avg_yaw_rate)^2 / 0.3^2)` | 跟踪 yaw 角速度命令。 |
| `track_lin_vel_y_walk_neg` | `track_lin_vel_y_yaw_frame_neg` | `-5.0` | `[1]` | `abs(cmd_y - avg_vy_yaw)` | walk 下额外惩罚横向误差，防止只靠前向速度拿奖励。 |
| `track_ang_vel_z_neg` | `track_ang_vel_z_world_neg` | `-3.0` | `[1, 2]` | `abs(cmd_yaw - avg_yaw_rate)` | 额外惩罚 yaw 跟踪误差，让转向命令更难被忽略。 |

`track_lin_vel_x_run_exp` 的权重是 walk x 跟踪的 3 倍，说明当前任务明显偏向高速跑。高速时放宽 `std` 是为了避免 10 m/s 以上速度下微小绝对误差被过度扣分。

## 4. 跑步速度 shaping

| term | 函数 | 权重 | style | 逻辑 | 训练意图 |
|---|---|---:|---|---|---|
| `track_lin_vel_x_bonus` | `track_lin_vel_x_progress_bonus` | `15.0` | `[2]` | `max(min(actual_vx, target_vx) - 2.0, 0) / (15.0 - 2.0)` | run 专属进度奖励，只奖励超过 2 m/s 且没有超过目标的有效前进速度。 |
| `track_lin_vel_x_neg` | `track_lin_vel_x_yaw_frame_neg` | `-1.0` | `[0, 1, 2]` | `abs(cmd_x - vx_yaw)` | 全 style 的前向速度误差惩罚，补充 exp reward 的尾部梯度。 |
| `track_ang_vel_z_bonus` | `track_ang_vel_z_world_bonus` | `1.0` | `[2]` | `exp(-yaw_error^2 / 0.5^2) * abs(cmd_x)` | 跑得越快，越鼓励 yaw 跟踪稳定。 |

这组和上一组配合：`track_lin_vel_x_run_exp` 负责“准不准”，`track_lin_vel_x_bonus` 负责“真的跑起来”，`track_lin_vel_x_neg` 保留线性误差梯度。

## 5. 动作和能耗正则

| term | 函数 | 权重 | 逻辑 | 训练意图 |
|---|---|---:|---|---|
| `energy` | `energy` | `-3e-4` | `norm(abs(torque * joint_vel))` | 惩罚机械功率，减少高能耗动作。 |
| `dof_acc_l2` | `joint_acc_l2` | `-3e-7` | `sum(joint_acc^2)` | 惩罚关节加速度，抑制高频抖动。 |
| `action_rate_weighted_l2` | `action_rate_weighted_l2` | `-0.01` | 加权 `sum((a_t - a_{t-1})^2)`，上限 `200` | 惩罚一阶动作变化，提升动作平滑性。 |
| `action_rate2_weighted_l2` | `action_rate2_weighted_l2` | `-0.005` | 加权 `sum((a_t - 2a_{t-1} + a_{t-2})^2)`，上限 `400` | 惩罚二阶动作变化，减少动作加加速度。 |
| `torques_weighted_neg` | `torques_weighted_neg` | `-7e-6` | 按关节加权的 torque L2 | 抑制高扭矩，尤其脚踝、腰、肩等高权重关节。 |
| `wholebody_vel_weighted_neg` | `wholebody_vel_weighted_neg` | `-0.007` | 按关节加权的 joint velocity L2 | 抑制全身关节高速摆动。 |

`action_rate_*`、`torques_weighted_neg` 和 `wholebody_vel_weighted_neg` 都使用类似速度缩放：低速时 penalty scale 接近 `max_speed_scale=2.0`，随着 `|cmd_x|` 接近 `speed_relax=6.0`，scale 平滑下降到 `min_speed_scale=0.4`。意思是站立和慢走要更稳，高速跑允许动作更激烈。

关节权重来自 `mdp.rewards` 里的 xhumanoid 映射，例如：

- torque penalty 中 ankle pitch 权重 `5.0`，waist yaw `3.0`，shoulder pitch `2.0`
- action rate penalty 中 hip yaw、hip roll 权重 `3.0`，ankle pitch、waist、shoulder 权重 `2.0`
- whole-body velocity penalty 中 waist yaw `3.0`，hip yaw/roll 和 ankle pitch `1.0`

## 6. 关节限制和姿态约束

| term | 函数 | 权重 | style | 逻辑 | 训练意图 |
|---|---|---:|---|---|---|
| `dof_pos_limits` | `joint_pos_limits` | `-10.0` | 全部 | 超出关节位置 limit 的距离和 | 避免关节打到硬限位。 |
| `joint_vel_limits_neg` | `joint_vel_limits_neg` | `-2.0` | 全部 | 超过 `0.9 * joint_vel_limit` 后才罚，单关节 excess clamp 到 `3.0` | 防止关节速度逼近上限。 |
| `torque_limits_neg` | `torque_limits_neg` | `-1.0` | 全部 | 超过 `0.85 * effort_limit` 后才罚 | 防止 actuator 长期顶满。 |
| `ankle_pitch_pos_limits` | `ankle_pitch_pos_limits_neg` | `-100.0` | 全部 | 脚踝 pitch 超 soft limit 的距离和 | 脚踝 pitch 限位重罚，保护落地和蹬地姿态。 |
| `hip_yaw_pos_neg` | `hip_yaw_pos_neg` | `-5.0` | `[1, 2]` | `sum(abs(q_hip_yaw)) * (1 - clamp(abs(cmd_yaw)/0.5, 0, 1))` | 非强转向时抑制髋 yaw 外八或乱扭；大 yaw 命令时放松。 |
| `shoulder_pitch_pos_neg` | `shoulder_pitch_pos_neg` | `-1.0` | 全部 | 肩 pitch 位置平方和，随 `|cmd_x|/15` 放松 | 低速时收住肩部姿态，高速跑允许摆臂。 |

`dof_pos_limits` 名字来自 `mdp.joint_pos_limits`。当前 `legged_lab/mdp/rewards.py` 没有这个定义，项目里 `rewards_gym.py` 有同名通用实现；如果运行时来自 IsaacLab 的 mdp，它的语义也通常是“超出关节位置限制才惩罚”。这一项需要在调试导入链时留意。

## 7. 稳定性和身体姿态

| term | 函数 | 权重 | style | 逻辑 | 训练意图 |
|---|---|---:|---|---|---|
| `base_height_neg` | `base_height_neg` | `-10.0` | `[0, 1, 2]` | 只惩罚低于 `target_height=1.18` 的高度误差，最多看低 `0.5 m` | 防止蹲太低或塌腰，不惩罚高于目标。 |
| `lin_vel_z_l2` | `lin_vel_z_l2` | `-1.0` | `[0, 1]` | `root_lin_vel_b.z^2` | 站立和走路时抑制上下跳动；跑步时关闭，给奔跑弹性。 |
| `ang_vel_xy_l2` | `ang_vel_xy_l2` | `-0.5` | 全部 | `root_ang_vel_b.roll^2 + pitch^2` | 抑制身体绕 roll/pitch 快速旋转。 |
| `body_orientation_exp` | `body_orientation_exp` | `1.0` | `[0, 1]` | waist yaw link 的重力投影 xy 越小越好，`exp(-100 * err)` | 站立和走路保持躯干直立。 |
| `body_orientation_speed_aware` | `body_orientation_speed_aware` | `3.0` | `[2]` | 目标 pitch 随 `cmd_x` 前倾，目标 roll 随 `cmd_x * cmd_yaw` 侧倾 | 跑步时允许并鼓励合理前倾和转弯侧倾。 |

跑步没有使用普通直立 `body_orientation_exp`，而是用 speed-aware 版本。这能避免高速时“必须竖直”的约束和自然跑姿冲突。

## 8. 接触、脚滑和落地

| term | 函数 | 权重 | style | 逻辑 | 训练意图 |
|---|---|---:|---|---|---|
| `body_contact_neg` | `body_contact_neg` | `-1.0` | 全部 | 非 ankle pitch body 的接触数量，阈值 `1.0` | 惩罚膝、肩、base、腰等非脚部触地。 |
| `feet_air_time_walk` | `feet_air_time_positive_biped` | `10.0` | `[1]` | 单脚支撑时奖励当前接触或腾空时长，最多 `0.4 s` | walk 下鼓励交替步态和足够摆动时间。 |
| `feet_clearance_walk` | `feet_clearance` | `2.0` | `[1]` | 摆动脚高度接近 `0.10 m` 得 Gaussian 奖励，`std=0.04` | walk 下鼓励抬脚，不拖脚。 |
| `feet_slide` | `feet_slide` | `-2.0` | `[0, 1, 2]` | 接触力 `fz > 50` 时惩罚脚速度超阈值 | 抑制脚在地面上滑。高速时前后向阈值从 `0.05` 放宽到 `0.45`。 |
| `feet_lin_vel_stand` | `feet_lin_vel_stand_neg` | `-2.0` | `[0]` | 站立时惩罚脚水平速度平方，实际 base 速度接近 `1.0` 时放松 | 站立脚要钉住，不要挪脚。 |
| `feet_contact_ang_vel_z` | `feet_contact_ang_vel_z` | `-0.1` | `[1, 2]` | 接触力 `fz > 150` 时惩罚脚绕 z 的角速度平方，上限 `10` | 防止脚掌接地时原地旋转磨地。 |
| `contact_impact_vel` | `contact_impact_vel` | `-2.0` | `[1, 2]` | touchdown 上升沿和落地前若干 policy step 的脚速度 | 降低落地冲击，跑得越快权重越高。 |
| `feet_descent_proximity_neg` | `feet_descent_proximity_neg` | `-2.0` | `[2]` | `max(0, -foot_vz) * exp(-clearance / 0.05)` | 跑步时脚越接近地面，越不希望高速向下砸。 |
| `gait_feet_force_max_neg` | `gait_feet_force_max_neg` | `-0.01` | `[1, 2]` | 脚接触力超过 `2000 N` 的部分，单脚 excess clamp 到 `100` | 抑制过大地面反力。 |
| `feet_stumble` | `feet_stumble` | `-2.0` | 全部 | 水平接触力大于 `5 * abs(vertical_force)` 时触发 | 惩罚绊脚或侧向撞击。 |

这组 reward 解决的是足端接触质量：不滑、不砸、不绊、非脚部不接触。`feet_slide` 的 `high_speed_slip_threshold` 很关键，因为高速跑时脚相对地面的前后向速度本来会更大，完全使用低速阈值会过分限制蹬地。

## 9. 足距、膝距和碰撞

| term | 函数 | 权重 | style | 逻辑 | 训练意图 |
|---|---|---:|---|---|---|
| `gait_feet_distance` | `gait_feet_distance` | `-8.0` | `[1, 2]` | body frame 下左右脚横向跨度应接近 `2 * target_y = 0.30 m`，随 `|cmd_vx|` 变强，横移命令大时放松 | 保持合理步宽，速度越快越重视。 |
| `knee_y_distance` | `knee_y_distance` | `-5.0` | `[1, 2]` | 左膝靠 `+0.15`，右膝靠 `-0.15`，随 `|cmd_vx|` 变强，横移命令大时放松 | 防止膝盖内扣或互相靠太近。 |
| `feet_collision` | `feet_collision` | `-30.0` | `[1, 2]` | 若左右脚 `y < 0.14` 且 `x < 0.24`，二值惩罚 | 强惩罚双脚太近导致碰撞或交叉步。 |

`feet_collision` 是二值项且权重很大，主要是硬约束味道；`gait_feet_distance` 和 `knee_y_distance` 是连续 shaping，帮助策略提前远离碰撞区。

## 10. AMP reward 配置

除 `RewardTerm` 之外，agent 还启用了 Multi-AMP：

```python
use_wgan_discriminator = True
amp_reward_coef_dict = {
    "style_0": 1.0,
    "style_1": 0.8,
    "style_2": 0.6,
}
amp_task_reward_lerp = 0.8
```

每个 style 有独立专家数据：

- `style_0`：stand motion
- `style_1`：walk motion
- `style_2`：run motion，当前有两段 run 数据

在 WGAN discriminator 中，AMP 原始奖励大致是：

```text
disc_reward = amp_reward_coef[style] * exp(clamp(D(policy) - mean(D(expert)), max=0) / 10)
final_reward = (1 - amp_task_reward_lerp) * disc_reward + amp_task_reward_lerp * task_reward
```

因此当前 `amp_task_reward_lerp=0.8` 表示最终 rollout reward 更偏任务 reward：约 80% task reward，加 20% AMP imitation reward。`amp_reward_coef_dict` 又让 stand 的 AMP imitation 最强，run 最弱，避免高速跑被专家风格过度束缚。

## 11. 调参时最值得盯的项

- 跑不起来：优先看 `track_lin_vel_x_run_exp`、`track_lin_vel_x_bonus`，以及 `action_rate_*`、`wholebody_vel_weighted_neg` 是否过强。
- 横向或 yaw 命令被忽略：看 `track_lin_vel_y_exp`、`track_lin_vel_y_walk_neg`、`track_ang_vel_z_exp`、`track_ang_vel_z_neg`。
- 脚滑：看 `feet_slide` 的 `slip_speed_threshold`、`high_speed_slip_threshold`、`min_contact_fz`。
- 落地太重：看 `contact_impact_vel`、`feet_descent_proximity_neg`、`gait_feet_force_max_neg`。
- 姿态太僵或跑姿不自然：看 `body_orientation_speed_aware`、`lin_vel_z_l2` 是否对 run 生效，以及 shoulder/hip 相关 penalty。
- 双脚或膝盖打架：看 `feet_collision`、`gait_feet_distance`、`knee_y_distance`。

## 12. 当前配置的奖励倾向

这套 reward 的核心倾向是：

1. 站立时稳定、不挪脚、贴近指定关节姿态。
2. 走路时重视横向、yaw 命令跟踪，并用 air time 和 clearance 塑造正常步态。
3. 跑步时强推前向速度和高速进度，同时通过落地、脚滑、足距、躯干前倾来约束可用跑姿。
4. 所有 style 都强惩罚提前终止、非脚部接触、关节限位和脚部碰撞。

整体上，它不是单纯“速度越快越好”的 reward，而是“在不摔、不滑、不撞、不过度用力的前提下，按 style 跟踪命令并模仿对应 AMP 风格”。
