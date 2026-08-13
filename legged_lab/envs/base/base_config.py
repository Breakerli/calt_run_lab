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

import math
from dataclasses import MISSING

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from legged_lab.sensors.camera import TiledCameraCfg
from legged_lab.sensors.lidar import LidarCfg


@configclass
class RewardCfg:
    pass


@configclass
class HeightScannerCfg:
    enable_height_scan: bool = False
    prim_body_name: str = MISSING
    resolution: float = 0.1
    size: tuple = (1.2, 1.6)  # 13x17 grid @ res=0.1 → 221 (nogait n_scan)
    debug_vis: bool = False
    drift_range: tuple = (0.0, 0.0)
    enable_feet_height_scan: bool = False
    feet_body_names: tuple[str, str] = ("ankle_pitch_l_link", "ankle_pitch_r_link")
    feet_scan_resolution: float = 0.05
    feet_scan_size: tuple[float, float] = (0.30, 0.10)  # 7x3 @ 0.05 (nogait foot grid)


@configclass
class BaseSceneCfg:
    max_episode_length_s: float = 20.0
    num_envs: int = 4096
    env_spacing: float = 2.5
    robot: ArticulationCfg = MISSING
    terrain_type: str = MISSING
    terrain_generator: TerrainGeneratorCfg = None
    max_init_terrain_level: int = 5
    terrain_restitution: float = 0.0
    height_scanner: HeightScannerCfg = HeightScannerCfg()
    lidar: LidarCfg = LidarCfg()
    depth_camera: TiledCameraCfg = TiledCameraCfg()


@configclass
class RobotCfg:
    actor_obs_history_length: int = 10
    critic_obs_history_length: int = 10
    action_scale: float = 0.25
    terminate_contacts_body_names: list = []
    feet_body_names: list = []


@configclass
class ObsScalesCfg:
    lin_vel: float = 1.0
    ang_vel: float = 1.0
    projected_gravity: float = 1.0
    commands: float | tuple[float, float, float] = 1.0  # (lin_vel_x, lin_vel_y, ang_vel_z)
    joint_pos: float = 1.0
    joint_vel: float = 1.0
    actions: float = 1.0
    height_scan: float = 1.0


@configclass
class NormalizationCfg:
    obs_scales: ObsScalesCfg = ObsScalesCfg()
    clip_observations: float = 100.0
    clip_actions: float = 100.0
    height_scan_offset: float = 0.5


@configclass
class CommandRangesCfg:
    lin_vel_x: tuple = (-0.6, 1.0)
    lin_vel_y: tuple = (-0.5, 0.5)
    ang_vel_z: tuple = (-1.0, 1.0)
    heading: tuple = (-math.pi, math.pi)


@configclass
class CommandsCfg:
    resampling_time_range: tuple = (10.0, 10.0)
    rel_standing_envs: float = 0.2
    rel_heading_envs: float = 1.0
    heading_command: bool = True
    heading_control_stiffness: float = 0.5
    debug_vis: bool = True
    ranges: CommandRangesCfg = CommandRangesCfg()
    use_curriculum: bool = False
    curriculum_rwd_threshold: float | None = None
    # 速度课程顶速时的跟踪 reward 比例门槛（低于 curriculum_rwd_threshold）；None 表示不随速度变化
    curriculum_rwd_threshold_min: float | None = None
    curriculum_rwd_threshold_v_lo: float = 5.0
    curriculum_rwd_threshold_v_hi: float = 15.0
    curriculum_update_step: float | None = None
    curriculum_max_lin_vel_x: tuple | None = None
    curriculum_max_lin_vel_y: tuple | None = None
    curriculum_min_episode_length: int | None = None
    curriculum_min_long_ratio: float | None = None
    # 全 run env 上局平均步长 / max_episode_length 须超过该比例才升速（对齐 legged_gym）
    curriculum_min_mean_episode_ratio: float = 0.8
    curriculum_min_episode_steps: int = 300
    curriculum_min_speed_ratio: float = 0.7
    curriculum_ema_alpha: float = 0.9
    curriculum_retreat_below_ratio: float = 0.5
    curriculum_log_interval_steps: int = 256
    # 站/走/跑分桶采样比例 (stand, walk, run)；None 表示关闭，仍用均匀采样 + rel_standing_envs
    cmd_bucket_probs: tuple[float, float, float] | None = None
    # 走桶 vx：在 [lo, hi] 内均匀采样；站=0、跑>=run_min 由分桶决定
    cmd_bucket_walk_lin_vel_x: tuple[float, float] = (-0.5, 1.0)
    # 走桶中抽取一部分样本，强制 vy/yaw 至少一项为有效命令
    cmd_bucket_walk_lateral_yaw_frac: float = 0.0
    # 跑桶 vx 下限（含），U[min, vmax]；style 滞回仍用 StyleFromCommandCfg
    cmd_bucket_run_lin_vel_x_min: float = 1.0
    # run style (style_id=2): 重采样时高速命令占比与峰值比例
    run_high_speed_sample_frac: float | None = None
    run_high_speed_threshold_frac: float = 0.7
    # 重采样：阶跃(立即到位) vs 斜坡(保持当前 cmd，由 max_lin_accel 爬升)
    cmd_step_sample_frac: float = 0.5
    # 线速度 / 非 heading 角速度命令变化率上限 [m/s^2]（每控制步）
    max_lin_accel: float = 5.0


@configclass
class NoiseScalesCfg:
    lin_vel: float = 0.2
    ang_vel: float = 0.2
    projected_gravity: float = 0.05
    joint_pos: float = 0.01
    joint_vel: float = 1.5
    height_scan: float = 0.1


@configclass
class NoiseCfg:
    add_noise: bool = True
    noise_scales: NoiseScalesCfg = NoiseScalesCfg()


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.4, 0.8),
            "restitution_range": (0.0, 0.005),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )
    add_others_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "mass_distribution_params": (-0.02, 0.08),
            "operation": "add",
        },
    )
    randomize_rigid_body_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "com_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
        },
    )
    randomize_rigid_body_inertia = EventTerm(
        func=mdp.randomize_rigid_body_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "inertia_distribution_params": (1.0, 1.0),
            "operation": "scale",
        },
    )
    randomize_motor_strength = EventTerm(
        func=mdp.randomize_motor_strength,
        mode="startup",
        params={
            "motor_strength_range": (1.0, 1.0),
        },
    )
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )


@configclass
class ActionDelayCfg:
    enable: bool = False
    params: dict = {"max_delay": 5, "min_delay": 0}


@configclass
class DomainRandCfg:
    events: EventCfg = EventCfg()
    action_delay: ActionDelayCfg = ActionDelayCfg()


@configclass
class PhysxCfg:
    gpu_max_rigid_patch_count: int = 10 * 2**15


@configclass
class SimCfg:
    dt: float = 0.005
    decimation: int = 4
    physx: PhysxCfg = PhysxCfg()
