# Ultra Game Yaw 奖励函数说明

本文档解释当前项目 `UltraGameYawMotionRunEnv` 使用的奖励函数。核心代码位置：

- 奖励配置：`legged_lab/envs/ultra_game_yaw/motion_run_cfg.py`
- 奖励实现：`legged_lab/mdp/rewards.py`
- 环境 step / style 更新 / AMP 观测：`legged_lab/envs/ultra_game_yaw/ultra_motion_run_env.py`
- Multi-AMP 奖励混合：`rsl_rl/rsl_rl/runners/multi_amp_on_policy_runner.py`
- WGAN AMP 奖励：`rsl_rl/rsl_rl/modules/wgan_discriminator.py`

`legged_lab/mdp/rewards_gym.py` 里有大量从旧 IsaacGym 版本迁移来的奖励函数，但当前 `UltraGameYawRewardCfg` 没有直接挂这些函数；本说明只聚焦当前配置里实际启用的奖励项。

## 1. 总体计算流程

每个策略步的环境流程大致是：

1. policy 输出动作 `actions`。
2. 环境用动作延迟 buffer 取 delayed action，并裁剪到 `[-clip_actions, clip_actions]`。
3. 关节目标位置：
   `q_target = action * action_scale + default_joint_pos + joint_pos_bias`。
4. 物理仿真执行 `decimation=4` 个子步。
5. 更新命令、style、跟踪速度 EMA。
6. 检查是否 reset / timeout。
7. 调 `RewardManager.compute(step_dt)` 计算任务奖励。
8. 训练时 runner 再按当前 style 用对应 AMP discriminator 计算 AMP 奖励，并与任务奖励混合。

当前配置：

- 物理步长 `dt = 0.0025s`
- 策略步长 `step_dt = dt * decimation = 0.01s`
- episode 最大长度 `20s`，即约 `2000` 个策略步
- 初始命令范围 `vx in [-0.5, 8.6]`，课程学习最高扩到 `15.0m/s`
- 命令采样桶比例 `stand : walk : run = 1 : 3 : 6`

IsaacLab 的 `RewardManager` 会把每个奖励项乘上 `dt`：

```text
task_reward_step = sum_i raw_i(env) * weight_i * step_dt
```

所以配置里的 weight 可以理解成“每秒尺度”的权重；单步真实贡献还要再乘 `0.01`。TensorBoard 中 `Episode_Reward/<term>` 是 episode 累计后除以 `max_episode_length_s` 的平均值。

举个最常见的读法：如果某项 raw 值是 `0.8`、weight 是 `15.0`，那么它对单个策略步的任务奖励贡献是：

```text
0.8 * 15.0 * 0.01 = 0.12
```

如果某个惩罚项 raw 值是 `0.3`、weight 是 `-8.0`，单步贡献是：

```text
0.3 * (-8.0) * 0.01 = -0.024
```

所以看配置时不要只看 weight 大小，也要看 raw 的量纲和典型范围。比如 `termination_penalty` 的 weight 是 `-200`，但 raw 通常只有终止那一个 step 为 `1`；`track_lin_vel_x_run_exp` 的 weight 是 `15`，但只要一直跑，它每个 step 都在持续贡献。

## 2. Style 分流机制

这个环境有 3 个 style：

- `style_0`: stand
- `style_1`: walk
- `style_2`: run

奖励项里经常有 `style_mask`，表示该项只在某些 style 下生效。没有 `style_mask` 的项对所有 style 生效。

style 不是 policy 直接输出的，也不进入 actor obs；它由命令 `cmd = [vx, vy, yaw_rate]` 推断：

- 三个命令绝对值都 `< 0.05`，强制 `stand`。
- 从 `stand` 出发：
  - `|vx| >= 1.0` 进入 `run`
  - 命令活跃且 `|vx| < 1.0` 进入 `walk`
- 从 `walk` 出发：
  - `|vx| >= 1.0` 进入 `run`
  - 命令不活跃回到 `stand`
- 从 `run` 出发：
  - `|vx| < 0.85` 降到 `walk`
  - 命令不活跃回到 `stand`

这里 `run_enter=1.0`，`run_exit=0.85` 是滞回，避免速度在边界附近抖动导致 style 频繁跳变。

几个具体例子：

- `cmd=[0.00, 0.00, 0.00]`：三个命令都小于 `0.05`，进入 `style_0 stand`。
- `cmd=[0.40, 0.00, 0.00]`：命令活跃，但 `|vx| < 1.0`，进入 `style_1 walk`。
- `cmd=[0.00, 0.30, 0.00]`：虽然前向速度是 0，但有横向命令，仍是 `style_1 walk`。
- `cmd=[0.00, 0.00, 0.30]`：只有 yaw 命令，也不是站立，会进入 `style_1 walk`。
- `cmd=[1.20, 0.00, 0.00]`：`|vx| >= 1.0`，进入 `style_2 run`。
- 如果当前已经是 run，`vx` 降到 `0.90` 时仍保持 run；降到 `0.80` 才因低于 `run_exit=0.85` 转回 walk。

## 3. 任务奖励与 AMP 奖励的关系

环境 `step()` 先返回任务奖励 `task_reward`。训练 runner 中又把它替换成 Multi-AMP 后的总奖励：

```text
total_reward = 0.8 * task_reward + 0.2 * amp_reward
```

因为 `amp_task_reward_lerp = 0.8`。

当前使用 WGAN discriminator。对某个 style：

```text
score = D(policy_transition) - moving_mean(D(expert_transition))
score = clamp(score, max=0)
amp_reward = amp_coef[style] * exp(score / 10)
```

其中：

- `style_0`: `amp_coef = 1.0`
- `style_1`: `amp_coef = 0.8`
- `style_2`: `amp_coef = 0.6`

因此 AMP 奖励越接近专家动作分布越高，上限大约是对应 style 的 `amp_coef`。最终给 PPO 的 reward 不是纯任务奖励，而是任务奖励为主、AMP 风格模仿为辅。

数值例子：

- run style 下 `amp_coef=0.6`，如果 discriminator 认为策略动作非常像专家，`score≈0`，则 `amp_reward≈0.6`。
- 如果此时任务奖励单步为 `0.10`，AMP 奖励为 `0.60`，最终 PPO reward 是 `0.8*0.10 + 0.2*0.60 = 0.20`。
- 如果策略动作不像专家，假设 `score=-10`，则 `amp_reward=0.6*exp(-1)=0.221`，同样任务奖励 `0.10` 时总奖励是 `0.8*0.10 + 0.2*0.221 = 0.124`。

这说明 AMP 不会完全替代任务奖励，但会明显偏好“像专家库里的站/走/跑动作”的解。

## 4. 启用的任务奖励总览

下表中的 weight 是配置权重，真实单步贡献为 `raw * weight * 0.01`。

| 奖励项 | style | weight | 作用 |
|---|---:|---:|---|
| `alive` | all | `10.0` | 活着且未 reset，随 episode 进度增加 |
| `termination_penalty` | all | `-200.0` | 非 timeout 摔倒/碰撞终止时重罚 |
| `stand_still` | 0 | `10.0` | 站立时靠近目标关节角 |
| `feet_lin_vel_stand` | 0 | `-2.0` | 站立时脚不要乱动 |
| `track_lin_vel_x_walk_exp` | 1 | `5.0` | 走路时跟踪前向速度 |
| `track_lin_vel_y_walk_neg` | 1 | `-5.0` | 走路时惩罚横向速度误差 |
| `feet_air_time_walk` | 1 | `10.0` | 走路时鼓励单脚支撑/迈步 |
| `feet_clearance_walk` | 1 | `2.0` | 走路摆动脚接近目标抬脚高度 |
| `track_lin_vel_x_run_exp` | 2 | `15.0` | 跑步时主要前向速度跟踪 |
| `track_lin_vel_x_bonus` | 2 | `15.0` | 高速前进额外进度奖励 |
| `track_ang_vel_z_bonus` | 2 | `1.0` | 跑步时转向跟踪奖励，随前向命令放大 |
| `body_orientation_speed_aware` | 2 | `3.0` | 高速跑时允许/鼓励前倾和转弯侧倾 |
| `feet_descent_proximity_neg` | 2 | `-2.0` | 跑步落脚接近地面时惩罚过快下落 |
| `track_lin_vel_y_exp` | 1,2 | `5.0` | 走/跑横向速度指数跟踪 |
| `track_ang_vel_z_exp` | 1,2 | `5.0` | 走/跑 yaw 角速度指数跟踪 |
| `track_ang_vel_z_neg` | 1,2 | `-3.0` | 走/跑 yaw 角速度绝对误差惩罚 |
| `hip_yaw_pos_neg` | 1,2 | `-5.0` | 非大转向时抑制髋 yaw 偏角 |
| `feet_contact_ang_vel_z` | 1,2 | `-0.1` | 脚接触地面时抑制脚 yaw 角速度 |
| `contact_impact_vel` | 1,2 | `-2.0` | 惩罚触地瞬间/触地前脚速度过大 |
| `gait_feet_force_max_neg` | 1,2 | `-0.01` | 惩罚足端接触力超过阈值 |
| `gait_feet_distance` | 1,2 | `-8.0` | 跑走时保持左右脚横向间距 |
| `knee_y_distance` | 1,2 | `-5.0` | 跑走时保持左右膝横向间距 |
| `feet_collision` | 1,2 | `-30.0` | 左右脚太近时二值碰撞风险惩罚 |
| `lin_vel_z_l2` | 0,1 | `-1.0` | 站/走时抑制 base 竖直速度 |
| `body_orientation_exp` | 0,1 | `1.0` | 站/走时腰部保持直立 |
| `track_lin_vel_z_exp` | all | `0.05` | 所有 style 抑制竖直速度 |
| `track_lin_vel_x_neg` | all | `-1.0` | 所有 style 前向速度绝对误差惩罚 |
| `base_height_neg` | all | `-10.0` | base 低于目标高度时惩罚 |
| `feet_slide` | all | `-2.0` | 接触时足端打滑惩罚 |
| `energy` | all | `-3e-4` | 功率/能耗惩罚 |
| `dof_acc_l2` | all | `-3e-7` | 关节加速度 L2 惩罚 |
| `action_rate_weighted_l2` | all | `-0.01` | 动作一阶差分平滑惩罚 |
| `action_rate2_weighted_l2` | all | `-0.005` | 动作二阶差分平滑惩罚 |
| `torques_weighted_neg` | all | `-7e-6` | 分关节加权 torque L2 |
| `wholebody_vel_weighted_neg` | all | `-0.007` | 分关节加权速度 L2 |
| `dof_pos_limits` | all | `-10.0` | 关节位置超限惩罚 |
| `joint_vel_limits_neg` | all | `-2.0` | 关节速度超过 soft limit 惩罚 |
| `torque_limits_neg` | all | `-1.0` | torque 超过 soft limit 惩罚 |
| `ankle_pitch_pos_limits` | all | `-100.0` | 踝 pitch 超 soft limit 重罚 |
| `shoulder_pitch_pos_neg` | all | `-1.0` | 低速时抑制肩 pitch 偏角 |
| `ang_vel_xy_l2` | all | `-0.5` | 抑制 base roll/pitch 角速度 |
| `body_contact_neg` | all | `-1.0` | 非脚部 body 接触惩罚 |
| `feet_stumble` | all | `-2.0` | 足端水平撞击/绊脚惩罚 |

### 4.1 参数速查表

这里把当前配置中最重要的参数单独列出来。调奖励时，通常先看这些参数，而不是直接改函数实现。

| 参数 | 出现位置 | 当前值 | 含义 | 调大/调小的效果 |
|---|---|---:|---|---|
| `style_mask` | 多数奖励项 | `[0]` / `[1]` / `[2]` / `[1,2]` | 控制奖励只在哪些 style 生效 | 改错会让站/走/跑互相干扰，是最先检查的开关 |
| `std` | 速度/姿态指数奖励 | `0.3/0.5/0.6` 等 | 指数奖励宽度，误差除以 `std^2` | 越大越宽松；越小越严格，奖励下降更快 |
| `std_speed_max` | `track_lin_vel_x_run_exp` | `1.0` | 高速时把 `std` 从 `0.6` 放宽到 `1.0` | 越大越允许高速误差；太大可能跟踪松 |
| `speed_walk` | `track_lin_vel_x_run_exp` | `1.0` | 开始按速度放宽 `std` 的命令速度 | 越小越早放宽；越大低速更严格 |
| `speed_max` | 多个 run 项 | `15.0` | 归一化/截断使用的最高速度 | 应与课程最高速度一致，否则高速段尺度会偏 |
| `avg_vel_window_s` | walk/yaw 跟踪 | `0.6` | 使用 EMA 平均速度的窗口 | 越大越平滑但响应慢；越小更敏感但噪声大 |
| `speed_threshold` | `track_lin_vel_x_bonus` | `2.0` | 前向 bonus 开始给分的速度 | 越高越强调真正跑起来；越低更早给 bonus |
| `normalize_speed` | `track_lin_vel_x_bonus` | `True` | bonus 是否除以速度范围归一化 | 开启后 raw 在 0 到 1 左右，更好控权重 |
| `pitch_coef` | `body_orientation_speed_aware` | `0.025` | 跑步前倾目标系数 | 越大越鼓励高速前倾 |
| `roll_coef` | `body_orientation_speed_aware` | `0.07` | 转弯侧倾目标系数 | 越大转弯越倾斜；过大可能歪身 |
| `foot_sole_offset` | `feet_descent_proximity_neg` | `0.074` | 踝 link 到脚底的高度偏移 | 偏大时认为脚更接近地面，落脚惩罚更早出现 |
| `proximity_height_scale` | `feet_descent_proximity_neg` | `0.05` | 接近地面权重衰减尺度 | 越大，离地较高时也罚下落；越小只贴地时罚 |
| `threshold` | `feet_air_time_walk` | `0.4` | 单脚支撑/摆动时间奖励上限 | 越大越鼓励长步态周期；过大可能拖脚 |
| `target_height` | `feet_clearance_walk` | `0.10` | 走路摆动脚目标高度 | 越大抬脚越高；过大耗能、动作夸张 |
| `min_contact_fz` | `feet_slide` | `50.0` | 认为脚真正着地的竖直力阈值 | 越大越少触发滑动罚；越小轻触地也罚 |
| `slip_speed_threshold` | `feet_slide` | `0.05` | 小滑动死区 | 越大越容忍脚滑；越小更严格 |
| `high_speed_slip_threshold` | `feet_slide` | `0.45` | 高速时纵向滑动放宽阈值 | 越大越允许前后推蹬；过大可能脚底乱滑 |
| `high_speed_relax` | `feet_slide` | `3.0` | 多快达到高速滑动放宽 | 越小越早放宽；越大低中速更严格 |
| `min_contact_fz` | `feet_contact_ang_vel_z` | `150.0` | 脚 yaw 转动惩罚的接触力阈值 | 越大只在重接触时罚；越小更早罚 |
| `max_penalty` | `feet_contact_ang_vel_z` | `10.0` | 单项 raw 上限 | 防止异常角速度导致巨大惩罚 |
| `pre_contact_policy_steps` | `contact_impact_vel` | `2` | 触地前多少个策略步也惩罚 | 越大越早压低落脚速度 |
| `contact_force_threshold` | `contact_impact_vel` | `1.0` | 判断足端接触的力阈值 | 越低越敏感；越高可能漏掉轻触地 |
| `max_force` | `gait_feet_force_max_neg` | `2000.0` | 足端竖直力超过多少开始罚 | 越小越保护硬件但可能限制爆发力 |
| `target_y` | `gait_feet_distance` / `knee_y_distance` | `0.15` | 单侧脚/膝期望横向距离 | 越大站姿/跑姿更宽；越小更窄 |
| `speed_scale` | 间距项 | `0.5` | 前向速度对间距惩罚的放大 | 越大高速时越强制脚/膝分开 |
| `threshold_y` | `feet_collision` | `0.14` | 左右脚横向距离危险阈值 | 越大越保守，脚稍近就罚 |
| `threshold_x` | `feet_collision` | `0.24` | 左右脚前后距离危险阈值 | 越大越保守，交错步也更容易被罚 |
| `target_height` | `base_height_neg` | `1.18` | base 目标高度下限 | 越大越不允许蹲低；过大可能逼出踮脚 |
| `min_speed_scale/max_speed_scale` | 动作/torque/速度正则 | `0.4/2.0` | 速度相关正则强度范围 | 低速强、高速弱；增大 `max` 会让低速更稳但更僵 |
| `speed_relax` | 动作/torque/速度正则 | `6.0` | 多快从强正则过渡到弱正则 | 越小越早放松；越大中高速也保持保守 |
| `soft_limit` | `joint_vel_limits_neg` | `0.9` | 速度限制软阈值比例 | 越小越早罚速度超限 |
| `soft_limit` | `torque_limits_neg` | `0.85` | 力矩限制软阈值比例 | 越小越早罚力矩接近上限 |
| `relax_speed` | `shoulder_pitch_pos_neg` | `15.0` | 肩部姿态惩罚随前向速度放松的速度 | 越小越早允许摆臂；越大肩更受约束 |

常见调参原则：

- 指数奖励的 `std` 决定“容忍误差”。如果训练初期怎么都拿不到跟踪奖励，先适当增大 `std`；如果学会后跟踪很松，再收小。
- 接触阈值类参数要和机器人质量、脚底接触力尺度一起看。`min_contact_fz` 太低会把擦碰也当作着地，太高会漏掉真实轻接触。
- 速度相关的 `speed_relax`、`speed_scale`、`speed_max` 要和命令课程一致。课程最高速度改了，最好同步检查这些参数。
- 强惩罚项如 `feet_collision`、`ankle_pitch_pos_limits`、`termination_penalty` 的 raw 通常是稀疏或超限才出现，weight 大是为了“一旦出现就明显不划算”。

## 5. Episode / 生存项

### `alive`

```text
raw = 1(not reset) * episode_progress
episode_progress = episode_length / max_episode_length
```

权重 `+10`。它不是固定每步 +1，而是 episode 越往后越大。刚开始接近 0，接近 20s 时接近 1。意图是鼓励机器人撑完整局。

例子：最大 episode 是 `2000` 步。第 `500` 步时 `episode_progress=0.25`，raw 为 `0.25`，单步贡献 `0.25*10*0.01=0.025`。第 `1800` 步时 raw 为 `0.9`，单步贡献 `0.09`。所以越接近完整跑完一局，活着奖励越值钱。

### `termination_penalty`

```text
raw = reset_buf AND NOT time_out_buf
```

权重 `-200`。只有因为摔倒/身体接触等提前终止时惩罚；正常超时结束不罚。由于还乘 `0.01`，单步终止惩罚约 `-2.0`。

提前终止由 `check_reset()` 判断：`knee_pitch.*`、`shoulder_pitch.*`、`base_link`、`waist_yaw_link` 等非足部关键 body 的接触力历史超过 `1.0N` 即 reset。

例子：机器人跑到第 `300` 步时腰部触地，`reset_buf=True` 且 `time_out_buf=False`，raw 为 `1`，这一项给 `-200*0.01=-2.0`。如果是正常跑满 `2000` 步 timeout，`time_out_buf=True`，raw 为 `0`，没有这个终止惩罚。

## 6. Stand 奖励

### `stand_still`

只对 `style_0` 生效。目标关节角在配置里显式给出：

```text
[0, 0, 0, 0, 0, 0, 0, -0.25, -0.25, 0.5, 0.5, -0.25, -0.25]
```

公式：

```text
raw = exp(-sum((q - q_target)^2))
```

越接近站立姿态越接近 1。

参数说明：

- `target_angles`：站立目标关节角，顺序与机器人关节顺序一致。任何关节偏离都会以平方误差加入总误差。
- `style_mask=[0]`：只在 stand style 生效，walk/run 不会拿这个站姿奖励。

例子：如果 13 个关节总平方误差 `sum((q-q_target)^2)=0.05`，raw=`exp(-0.05)=0.951`，单步贡献 `0.951*10*0.01=0.0951`。如果总平方误差变成 `1.0`，raw=`0.368`，单步贡献降到 `0.0368`。

### `feet_lin_vel_stand`

只对 `style_0` 生效。惩罚两个踝部 link 的水平速度：

```text
penalty = sum(foot_vel_xy^2)
relax = smoothstep(actual_base_speed / 1.0)
raw = penalty * (1 - relax)
```

站立时脚越静越好；如果机器人已经被扰动到移动，惩罚会逐渐放松，避免恢复过程中被过度限制。

参数说明：

- `horizontal_only=True`：只看脚的水平速度，不看上下方向速度。
- `relax_speed=1.0`：base 实际水平速度接近 `1m/s` 时，这个站立脚速惩罚基本放松。

例子：站立时两只脚水平速度平方和为 `0.20`，base 速度几乎为 0，raw≈`0.20`，单步贡献 `0.20*(-2)*0.01=-0.004`。如果机器人被推到 base 速度 `1m/s`，`relax≈1`，raw 接近 0，这项不会阻碍机器人调整脚步恢复。

## 7. Walk 奖励

### `track_lin_vel_x_walk_exp`

只对 `style_1` 生效。使用 yaw frame 下的前向速度，`std=0.3`，并使用约 `0.6s` 的速度平均窗口：

```text
err = abs(cmd_vx - avg_vx_yaw)
raw = exp(-err / std^2)
```

注意这里是 `abs(err)` 放进指数，不是 `err^2`。

参数说明：

- `std=0.3`：误差衰减很快，`std^2=0.09`。误差 `0.1m/s` 还可以，误差 `0.5m/s` 奖励会很低。
- `avg_vel_window_s=0.6`：用 0.6 秒 EMA 速度，降低瞬时速度抖动对奖励的影响。

例子：walk 命令 `cmd_vx=0.6m/s`，实际平均 `avg_vx_yaw=0.5m/s`，误差 `0.1`，raw=`exp(-0.1/0.09)=0.329`，单步贡献 `0.329*5*0.01=0.0165`。如果实际只有 `0.1m/s`，误差 `0.5`，raw=`exp(-5.56)=0.0039`，几乎拿不到这项奖励。

### `track_lin_vel_y_walk_neg`

只对 `style_1` 生效：

```text
raw = abs(cmd_vy - avg_vy_yaw)
```

权重为负，直接惩罚横向速度误差。

例子：walk 命令 `cmd_vy=0.2m/s`，实际平均 `0.05m/s`，raw=`0.15`，单步贡献 `0.15*(-5)*0.01=-0.0075`。如果没有横向命令但机器人横向漂 `0.2m/s`，同样会吃 `-0.01` 左右的单步惩罚。

### `feet_air_time_walk`

只对 `style_1` 生效。它鼓励双足步态里的“单脚支撑”：

```text
in_contact = contact_time > 0
single_stance = exactly_one_foot_in_contact
raw = min(mode_time_of_single_stance_foot, threshold=0.4)
```

当命令几乎为零时不奖励，避免原地乱抬脚刷分。

参数说明：

- `threshold=0.4`：单脚支撑时间最多按 `0.4s` 计分，再长不会继续增加。
- `sensor_cfg` 指向两只 `ankle_pitch.*`，即用脚踝/足端接触状态判断左右脚是否着地。

例子：当前只有左脚接触，右脚摆动，单脚支撑相关时间是 `0.18s`，raw=`0.18`，单步贡献 `0.18*10*0.01=0.018`。如果时间是 `0.60s`，会被截断成 `0.4`，单步贡献最多 `0.04`。

### `feet_clearance_walk`

只对 `style_1` 生效。对摆动脚给抬脚高度奖励：

```text
in_swing = foot_contact_force <= 1.0
height_score = exp(-(foot_z - 0.10)^2 / 0.04^2)
raw = sum(height_score * in_swing)
```

同样要求命令活跃，避免站立时抬脚刷分。

参数说明：

- `target_height=0.10`：摆动脚理想高度约 `10cm`。
- `std=0.04`：高度偏差的容忍范围。偏离 `4cm` 时 raw 约为 `exp(-1)=0.368`。

例子：摆动脚高度 `0.10m`，raw 接近 `1`；高度 `0.14m` 或 `0.06m`，单脚 raw 约 `0.368`；高度 `0.18m`，raw=`exp(-4)=0.018`，几乎没有奖励。两只脚都摆动时会求和，但正常步态多数时候只有一只脚在摆动。

## 8. Run 奖励

### `track_lin_vel_x_run_exp`

只对 `style_2` 生效，是跑步最核心的速度跟踪奖励。基础 `std=0.6`，当命令速度从 `1.0` 增长到 `15.0` 时，`std` 通过 smoothstep 放宽到 `1.0`：

```text
std_eff = lerp(0.6, 1.0, smoothstep((|cmd_vx|-1)/(15-1)))
err = abs(cmd_vx - vx_yaw)
raw = exp(-err / std_eff^2)
```

高速下跟踪误差允许更大，避免 10m/s 以上训练过于苛刻。

参数说明：

- `std=0.6`：低速 run 的基础容忍度。
- `std_speed_max=1.0`：高速时最大容忍度。
- `speed_walk=1.0`、`speed_max=15.0`：当 `|cmd_vx|` 从 `1` 到 `15` 增大时，`std_eff` 从 `0.6` 平滑过渡到 `1.0`。

例子 1：`cmd_vx=3m/s`，假设 `std_eff≈0.62`，实际 `vx=2.6m/s`，误差 `0.4`，raw≈`exp(-0.4/0.62^2)=0.353`，单步贡献 `0.353*15*0.01=0.053`。

例子 2：`cmd_vx=12m/s`，`std_eff` 接近 `0.94`，实际 `vx=11.2m/s`，误差 `0.8`，raw≈`exp(-0.8/0.94^2)=0.404`，单步贡献约 `0.061`。如果高速仍用 `std=0.6`，同样误差 raw 只有 `0.108`，训练会更难。

### `track_lin_vel_x_bonus`

只对 `style_2` 生效。奖励真正跑起来的前向进度：

```text
target = clamp(cmd_vx, 0, 15)
actual = clamp(vx_yaw, 0, 15)
rewardable = min(actual, target)
raw = clamp(rewardable - 2.0, 0, 13) / 13
```

低于 `2m/s` 没有 bonus；越接近目标高速，bonus 越大。`min(actual, target)` 防止跑得比命令更快刷分。

参数说明：

- `speed_threshold=2.0`：只有达到 `2m/s` 后才开始给高速进度 bonus。
- `speed_max=15.0`：bonus 的归一化上限。
- `normalize_speed=True`：把 raw 控制在 0 到 1 左右，便于用 weight 调强度。

例子：`cmd_vx=8m/s`，实际 `vx=6m/s`，`rewardable=6`，raw=`(6-2)/(15-2)=0.308`，单步贡献 `0.308*15*0.01=0.046`。如果实际 `vx=10m/s` 但命令只有 `8m/s`，`rewardable=min(10,8)=8`，raw=`0.462`，不会因为超速到 `10m/s` 继续加分。

### `track_ang_vel_z_bonus`

只对 `style_2` 生效：

```text
err = (cmd_yaw - avg_yaw_rate)^2
raw = exp(-err / 0.5^2) * abs(cmd_vx)
```

跑得越快，转向跟踪奖励越重要。

参数说明：

- `std=0.5`：yaw rate 误差的指数奖励宽度。
- `avg_vel_window_s=0.6`：使用平均 yaw rate，减少瞬时角速度尖峰。
- 乘 `abs(cmd_vx)`：同样 yaw 跟踪误差，高速跑时影响更大。

例子：`cmd_vx=6m/s`、`cmd_yaw=0.3rad/s`、实际平均 `0.2rad/s`，误差平方 `0.01`，raw=`exp(-0.01/0.25)*6=5.765`，weight 只有 `1`，单步贡献约 `0.0577`。如果 `cmd_vx=1m/s`，同样 yaw 误差 raw 约 `0.961`，贡献只有 `0.0096`。

### `body_orientation_speed_aware`

只对 `style_2` 生效。高速时不强行要求完全直立，而是给一个随命令变化的姿态目标：

```text
pitch_target = 0.025 * max(cmd_vx, 0)
roll_target = 0.07 * abs(cmd_vx) * cmd_yaw
err = (pitch_target - projected_gravity_x)^2 + (roll_target - projected_gravity_y)^2
raw = exp(-100 * err)
```

直观上：跑得快允许前倾，转弯时允许侧倾。

参数说明：

- `pitch_coef=0.025`：每 `1m/s` 前向命令，对应约 `0.025` 的 projected-gravity x 目标。
- `roll_coef=0.07`：转弯侧倾目标与 `|cmd_vx| * cmd_yaw` 成正比。
- `exp(-100*err)`：姿态目标很敏感，误差 `0.01` 时 raw 约 `0.368`。

例子：`cmd_vx=8m/s`、直行 `cmd_yaw=0`，`pitch_target=0.20`，`roll_target=0`。如果当前 projected gravity 为 `[0.18, 0.02]`，err=`(0.02)^2+(0.02)^2=0.0008`，raw=`exp(-0.08)=0.923`，单步贡献 `0.923*3*0.01=0.0277`。如果机器人太直，projected gravity 为 `[0.02,0]`，err=`0.0324`，raw=`0.039`，几乎拿不到这项跑姿奖励。

### `feet_descent_proximity_neg`

只对 `style_2` 生效。落脚接近地面时，如果脚还在快速向下，会被惩罚：

```text
clearance = foot_z - ground_z - foot_sole_offset
proximity_weight = exp(-clearance / 0.05)
descent = max(0, -foot_vz)
raw = sum(descent * proximity_weight)
```

它主要用于降低高速跑触地冲击。

参数说明：

- `foot_sole_offset=0.074`：脚底相对踝 link 的偏移，用来估算真正脚底离地高度。
- `proximity_height_scale=0.05`：离地越近惩罚越强；`clearance=0.05m` 时权重 `exp(-1)=0.368`。

例子：脚底离地 `clearance=0.02m`，脚向下速度 `1.5m/s`，单脚 raw=`1.5*exp(-0.02/0.05)=1.005`，单步贡献约 `1.005*(-2)*0.01=-0.020`。如果同样下落速度但离地 `0.15m`，权重 `exp(-3)=0.050`，raw 只有 `0.075`，惩罚很小。

## 9. Walk/Run 共享跟踪与步态约束

### 横向速度与 yaw 跟踪

`track_lin_vel_y_exp`：

```text
raw = exp(-abs(cmd_vy - avg_vy_yaw) / 0.3^2)
```

`track_ang_vel_z_exp`：

```text
raw = exp(-(cmd_yaw - avg_yaw_rate)^2 / 0.3^2)
```

`track_ang_vel_z_neg`：

```text
raw = abs(cmd_yaw - avg_yaw_rate)
```

这里同时有指数奖励和线性惩罚，是“给对的动作糖，也给错的动作压力”的组合。

例子：`cmd_yaw=0.4rad/s`，实际平均 `0.1rad/s`，yaw 误差 `0.3`。`track_ang_vel_z_exp` raw=`exp(-0.09/0.09)=0.368`，单步正贡献 `0.368*5*0.01=0.0184`；`track_ang_vel_z_neg` raw=`0.3`，单步负贡献 `0.3*(-3)*0.01=-0.009`。两项合起来仍然鼓励接近命令，但误差变大时正项快速消失，负项继续施压。

### `hip_yaw_pos_neg`

走/跑时抑制髋 yaw 关节偏角，但 yaw 命令越大越放松：

```text
penalty = sum(abs(q_hip_yaw))
relax = clamp(abs(cmd_yaw) / 0.5, 0, 1)
raw = penalty * (1 - relax)
```

直行时不希望髋 yaw 晃太大；转弯时允许它参与动作。

例子：左右髋 yaw 绝对角度和为 `0.20rad`。直行 `cmd_yaw=0` 时 raw=`0.20`，单步贡献 `0.20*(-5)*0.01=-0.010`。大转弯 `cmd_yaw=0.5` 时 `relax=1`，raw=0，这项完全放松。

### `feet_contact_ang_vel_z`

脚接触地面且 `fz > 150N` 时，惩罚足端 yaw 角速度：

```text
raw = clamp(sum(foot_ang_vel_z^2 * in_contact), max=10)
```

减少脚底在地上拧转。

例子：一只脚重接触，足端 yaw 角速度 `4rad/s`，另一只脚离地，raw=`min(4^2,10)=10`，单步贡献 `10*(-0.1)*0.01=-0.01`。如果没有 `max_penalty=10`，高速异常角速度会产生很大尖峰惩罚。

### `contact_impact_vel`

惩罚触地瞬间和触地前的足端速度：

```text
impact = foot just changed from no_contact to contact
pre_contact = recent history all in air, but older history有过接触
raw = sum((impact + pre_contact) * foot_speed) * cmd_weight
```

`cmd_weight = 0.5 + clamp(cmd_vx - 3, 0, vmax) / vmax`，高速跑时这个惩罚更强。

参数说明：

- `pre_contact_policy_steps=2`：除了真正触地瞬间，还看约 `2*step_dt=0.02s` 的触地前窗口。
- `contact_force_threshold=1.0`：足端接触判定阈值。

例子：`cmd_vx=8m/s`，`vmax=8.6`，`cmd_weight≈0.5+(8-3)/8.6=1.081`。如果触地瞬间脚速 `3m/s`，raw≈`3.24`，单步贡献 `3.24*(-2)*0.01=-0.0648`。这项会强烈推动策略在落地前减速脚。

### `gait_feet_force_max_neg`

足端竖直力超过 `2000N` 的部分才罚：

```text
raw = sum(clamp(fz - 2000, 0, 100))
```

例子：两只脚竖直力分别 `1800N` 和 `2300N`，第一只不罚，第二只超出 `300N` 但被 `max=100` 截断，所以 raw=`100`，单步贡献 `100*(-0.01)*0.01=-0.01`。这个截断让它更像“超过阈值就提醒”，而不是让极端接触力完全主导奖励。

### `gait_feet_distance`

在 base 坐标系下保持左右脚横向跨度接近 `2 * target_y = 0.30m`：

```text
lateral_span = left_foot_y - right_foot_y
penalty = abs(lateral_span - 0.30)
relax = clamp(abs(cmd_vy) / 0.5, 0, 1)
raw = penalty * (1 - relax) * 0.5 * clamp(abs(cmd_vx), max=15)
```

前向速度越高，脚间距越重要；横向命令越大，间距约束越放松。

参数说明：

- `target_y=0.15`：期望左脚在身体左侧约 `+0.15m`、右脚在 `-0.15m`，合计跨度 `0.30m`。
- `speed_scale=0.5`：间距惩罚乘 `0.5*|cmd_vx|`，高速时更重。
- `lateral_cmd_relax_at=0.5`：横向命令达到 `0.5m/s` 时，脚间距惩罚基本放松。

例子：`cmd_vx=6m/s`、`cmd_vy=0`，左右脚跨度只有 `0.20m`，误差 `0.10`，raw=`0.10*1*0.5*6=0.30`，单步贡献 `0.30*(-8)*0.01=-0.024`。如果 `cmd_vy=0.5m/s`，`relax=1`，raw=0，允许横移时脚距变化。

### `knee_y_distance`

类似足端间距，但对象是左右膝：

```text
score_left = 2 * abs(0.15 - left_knee_y)
score_right = 2 * abs(0.15 + right_knee_y)
raw = (score_left + score_right) * 0.5 * abs(cmd_vx) * (0.5 - clamp(abs(cmd_vy), max=0.5))
```

用于减少跑动时膝盖内扣或交叉。

例子：`cmd_vx=6m/s`、`cmd_vy=0`，左膝 y=`0.10`、右膝 y=`-0.08`，目标是 `+0.15/-0.15`。`score_left=0.10`，`score_right=0.14`，合计 `0.24`；再乘 `0.5*6*(0.5-0)=1.5`，raw=`0.36`，单步贡献 `0.36*(-5)*0.01=-0.018`。

### `feet_collision`

左右脚太近时给二值惩罚：

```text
feet_distance_y < 0.14 AND feet_distance_x < 0.24
```

raw 为 `0/1`，weight `-30`，单步实际约 `-0.3`，属于比较强的防碰撞项。

例子：左右脚横向距离 `0.10m`，前后距离 `0.18m`，同时小于 `threshold_y=0.14` 和 `threshold_x=0.24`，raw=1，单步惩罚 `-0.3`。如果横向距离仍是 `0.10m`，但前后距离 `0.35m`，raw=0，因为两脚虽然横向近，但前后错开较多，不按碰撞风险处理。

## 10. Stand/Walk 共享姿态项

### `lin_vel_z_l2`

只对 `style_0/1` 生效：

```text
raw = root_lin_vel_b_z^2
```

站和走时抑制 base 上下跳。

例子：站立或走路时 base 竖直速度 `0.3m/s`，raw=`0.09`，单步贡献 `0.09*(-1)*0.01=-0.0009`。这项单步不大，但长期会压制上下弹跳。

### `body_orientation_exp`

只对 `style_0/1` 生效，使用 `waist_yaw_link` 的 projected gravity：

```text
raw = exp(-100 * (g_body_x^2 + g_body_y^2))
```

腰部越直立越接近 1。

例子：如果 projected gravity 的水平分量平方和是 `0.0025`，raw=`exp(-0.25)=0.779`，单步贡献 `0.00779`。如果身体明显倾斜，平方和 `0.04`，raw=`exp(-4)=0.018`，几乎没有这项奖励。

## 11. 全 style 共享任务项

### `track_lin_vel_z_exp`

所有 style 生效：

```text
raw = exp(-abs(vz_yaw) / 0.3^2)
```

权重很小 `0.05`，是轻微鼓励竖直速度小。

例子：`vz=0.1m/s`，raw=`exp(-0.1/0.09)=0.329`，但 weight 只有 `0.05`，单步贡献 `0.00016`。它只是微弱偏好，不是主要控制项。

### `track_lin_vel_x_neg`

所有 style 生效：

```text
raw = abs(cmd_vx - vx_yaw)
```

权重 `-1`。它和 walk/run 的正向速度跟踪一起约束前向速度误差。站立时 `cmd_vx≈0`，它也会抑制机器人乱跑。

例子：站立命令 `cmd_vx=0`，机器人自己往前漂 `0.4m/s`，raw=`0.4`，单步惩罚 `-0.004`。run 命令 `8m/s`、实际 `6m/s`，raw=`2`，单步惩罚 `-0.02`，它会和 run 的指数奖励/bonus 一起推动追上命令。

### `base_height_neg`

只惩罚低于目标高度 `1.18m`，高于目标不罚：

```text
height_error = clamp(root_z - 1.18, min=-0.5, max=0)
raw = abs(height_error)
```

参数说明：

- `target_height=1.18`：期望 base 不要低于这个高度。
- `clamp(min=-0.5,max=0)`：最多按低 `0.5m` 惩罚，避免摔倒瞬间高度误差无限放大。

例子：base 高度 `1.10m`，低了 `0.08m`，raw=`0.08`，单步贡献 `0.08*(-10)*0.01=-0.008`。如果高度 `1.25m`，因为只罚低不罚高，raw=0。

### `feet_slide`

脚接触地面时惩罚打滑。当前配置：

- `min_contact_fz = 50N`
- `slip_speed_threshold = 0.05m/s`
- `high_speed_slip_threshold = 0.45m/s`
- `high_speed_relax = 3.0m/s`

高前向速度下，足端纵向 slip 阈值从 `0.05` 平滑放宽到 `0.45`，横向 slip 仍用低阈值。这样允许高速跑的前后方向推蹬/滚动，但不允许横向滑。

例子 1：低速 `cmd_vx=0`，接触脚横向速度 `0.20m/s`，横向超出 `0.20-0.05=0.15`，单脚 raw=`0.15`，单步贡献 `0.15*(-2)*0.01=-0.003`。

例子 2：高速 `cmd_vx>=3m/s`，纵向 slip 阈值放宽到约 `0.45m/s`。如果足端纵向速度 `0.40m/s`，纵向不罚；但横向速度 `0.40m/s` 仍按 `0.05` 阈值罚 `0.35`。这就是“允许推蹬，不允许侧滑”。

## 12. 全 style 共享正则与限制项

### 能耗与平滑

`energy`：

```text
raw = norm(abs(torque * joint_vel))
```

`dof_acc_l2`：

```text
raw = sum(joint_acc^2)
```

`action_rate_weighted_l2`：

```text
raw = clamp(sum(joint_weight * (a_t - a_{t-1})^2) * speed_scale, max=200)
```

`action_rate2_weighted_l2`：

```text
raw = clamp(sum(joint_weight * (a_t - 2a_{t-1} + a_{t-2})^2) * speed_scale, max=400)
```

`speed_scale` 随 `|cmd_vx|` 从 `2.0` 平滑降到 `0.4`，也就是低速时更要求动作平滑，高速跑时适当放松。

参数说明：

- `min_speed_scale=0.4`：高速时的最低正则倍数。
- `max_speed_scale=2.0`：低速/站立时的最高正则倍数。
- `speed_relax=6.0`：`|cmd_vx|` 接近 `6m/s` 时基本过渡到较低正则。
- `max_penalty=200/400`：动作一阶/二阶差分 raw 的截断上限，避免动作尖峰让奖励爆炸。

例子：某一步加权动作一阶差分平方和为 `10`。站立时 `speed_scale≈2.0`，raw=`20`，单步贡献 `20*(-0.01)*0.01=-0.002`。高速 `cmd_vx=6m/s` 时 `speed_scale≈0.4`，raw=`4`，单步贡献 `-0.0004`，允许跑步动作更有幅度。

### 分关节 torque / velocity 正则

`torques_weighted_neg` 使用按关节名映射的 torque 权重：

```text
hip_yaw: 1.5
hip_roll: 1.0
hip_pitch: 0.6
knee_pitch: 0.6
ankle_pitch: 5.0
waist_yaw: 3.0
shoulder_pitch: 2.0
```

raw：

```text
sum(weight_j * torque_j^2) * speed_scale
```

`wholebody_vel_weighted_neg` 类似，但作用于关节速度，权重里腰 yaw、踝 pitch、髋 yaw 比较重。低速时更强，高速时放松。

例子：如果某一时刻加权 torque 平方和是 `5000`，低速 `speed_scale≈2`，`torques_weighted_neg` raw=`10000`，单步贡献 `10000*(-7e-6)*0.01=-0.0007`。数值看着小，但它每步都有，长期会影响策略是否偏向省力动作。

对 `wholebody_vel_weighted_neg`，如果加权关节速度平方和是 `30`，低速 raw=`60`，单步贡献 `60*(-0.007)*0.01=-0.0042`；高速 raw≈`12`，贡献 `-0.00084`。所以它对低速抖动和站立乱动压制更明显。

### 关节/力矩限制

`dof_pos_limits`：

```text
raw = sum(max(lower - q, 0) + max(q - upper, 0))
```

`joint_vel_limits_neg`：

```text
soft_limit = 0.9 * joint_vel_limit
raw = 0.1 * sum(clamp(abs(qd) - soft_limit, 0, 3))
```

`torque_limits_neg`：

```text
soft_limit = 0.85 * effort_limit
raw = 0.005 * sum(max(abs(torque) - soft_limit, 0))
```

`ankle_pitch_pos_limits` 只看两个踝 pitch 的 soft joint position limits，weight `-100`，是非常强的防超限项。

例子：某关节速度 limit 是 `20rad/s`，`soft_limit=0.9` 后阈值是 `18rad/s`。实际速度 `19rad/s`，超出 `1`，该关节贡献 `0.1*1=0.1` raw；如果多个关节都超限会累加。

力矩限制类似：某电机 effort limit 是 `100Nm`，`soft_limit=0.85` 后阈值 `85Nm`。实际 torque `95Nm`，超出 `10Nm`，该关节 raw 贡献 `0.005*10=0.05`。

踝 pitch 超限更严：如果两个踝 pitch 对 soft limit 的越界量合计 `0.03rad`，raw=`0.03`，单步贡献 `0.03*(-100)*0.01=-0.03`。这已经比很多普通正则项大，说明踝 pitch 被设计成绝不能长期顶限。

### 上身与 base 稳定

`shoulder_pitch_pos_neg`：

```text
raw = sum(q_shoulder_pitch^2) * (1 - clamp(abs(cmd_vx) / 15, 0, 1))
```

低速时抑制肩 pitch 偏角，高速时允许摆臂。

参数说明：

- `relax_speed=15.0`：当前向命令达到 `15m/s` 时，肩 pitch 位置惩罚几乎完全放松。

例子：左右肩 pitch 平方和 `0.25`。站立时 raw=`0.25`，单步贡献 `0.25*(-1)*0.01=-0.0025`。`cmd_vx=7.5m/s` 时放松一半，raw=`0.125`。`cmd_vx=15m/s` 时 raw≈0，允许充分摆臂。

`ang_vel_xy_l2`：

```text
raw = root_ang_vel_b_x^2 + root_ang_vel_b_y^2
```

用于抑制 roll/pitch 方向角速度。

例子：base roll/pitch 角速度分别 `0.4` 和 `0.3rad/s`，raw=`0.16+0.09=0.25`，单步贡献 `0.25*(-0.5)*0.01=-0.00125`。

### 接触异常

`body_contact_neg` 对所有非踝足 body 生效：

```text
raw = number_of_non_foot_bodies_with_contact_force > 1N
```

注意这个奖励项是惩罚，而 `check_reset()` 也会因为关键 body 接触提前终止。也就是说，轻微/非终止接触会被持续罚；严重接触直接 reset 并吃 `termination_penalty`。

例子：如果一个非足部 body 接触力历史超过 `1N`，raw=`1`，单步贡献 `-0.01`。如果三个非足部 body 同时接触，raw=`3`，单步贡献 `-0.03`。若这些接触属于终止列表里的关键 body，还会触发 reset，并额外吃 `termination_penalty` 的 `-2.0`。

`feet_stumble`：

```text
raw = any(horizontal_contact_force > 5 * abs(vertical_contact_force))
```

足端水平撞击远大于竖直支撑力时视为绊脚。

例子：某脚水平接触力 `80N`，竖直力 `10N`，因为 `80 > 5*10`，raw=1，单步贡献 `1*(-2)*0.01=-0.02`。如果水平力 `40N`、竖直力 `100N`，则不算 stumble。

## 13. 奖励设计的整体意图

可以把当前奖励理解成四层：

1. **活着与不摔倒**：`alive`、`termination_penalty`、`body_contact_neg`。
2. **按命令完成任务**：`track_*`、`track_lin_vel_x_bonus`、yaw / lateral tracking。
3. **跑得像人形机器人**：脚间距、膝间距、抬脚、落脚冲击、足端接触约束。
4. **动作可部署**：能耗、动作平滑、关节/力矩限制、打滑、绊脚惩罚。

再加上 AMP：

5. **动作风格像专家数据**：站/走/跑分别用对应 motion 文件训练 discriminator，给策略一个风格先验。

因此它不是单纯“速度越快越好”。`track_lin_vel_x_bonus` 鼓励高速，但速度跟踪、姿态、接触力、落脚冲击、打滑、关节限制会共同限制不合理的刷分方式。

## 14. 调参时最值得关注的项

如果机器人跑不快，优先看：

- `track_lin_vel_x_run_exp`
- `track_lin_vel_x_bonus`
- `track_lin_vel_x_neg`
- `body_orientation_speed_aware`
- `contact_impact_vel`
- `feet_descent_proximity_neg`
- `action_rate_weighted_l2`
- `torques_weighted_neg`

如果机器人跑起来但脚很乱，优先看：

- `feet_slide`
- `feet_contact_ang_vel_z`
- `gait_feet_distance`
- `knee_y_distance`
- `feet_collision`
- `gait_feet_force_max_neg`
- `feet_stumble`

如果站立/走路不稳，优先看：

- `stand_still`
- `feet_lin_vel_stand`
- `body_orientation_exp`
- `lin_vel_z_l2`
- `base_height_neg`
- `track_lin_vel_y_walk_neg`

如果动作太抖或 sim2sim 不好，优先看：

- `action_rate_weighted_l2`
- `action_rate2_weighted_l2`
- `dof_acc_l2`
- `wholebody_vel_weighted_neg`
- `energy`
- `joint_vel_limits_neg`

## 15. 日志阅读提示

常看的 TensorBoard 标量：

- `Train/mean_reward`: PPO 实际使用的总奖励，已包含 AMP 混合后的 reward。
- `Episode_Reward/<term>`: 单个任务奖励项的 episode 平均贡献。
- `AMP/style_*_reward`: 如果日志里开启，对应各 style 的 AMP 奖励。
- `Curriculum/lin_vel_x_max`: 当前课程学习速度上限。
- `Curriculum/run_track_lin_vel_x_exp`: run env 上一局平均前向跟踪奖励。
- `Curriculum/run_track_gate`: 升速要求的前向跟踪门槛。
- `Curriculum/run_mean_episode_length_ratio`: run env 平均 episode 长度比例。

课程学习升速条件主要是：

```text
run_mean_episode_length > 0.8 * max_episode_length
run_track_lin_vel_x_exp > curriculum_threshold(vmax) * track_lin_vel_x_run_exp.weight
```

当前 `curriculum_threshold` 随速度上限从 `0.43` 逐渐降到 `0.35`，速度每次增加 `0.2m/s`，最高到 `15.0m/s`。
