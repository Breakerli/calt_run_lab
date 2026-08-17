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

from isaaclab.envs import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (  # noqa:F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRndCfg,
    RslRlSymmetryCfg,
)

import legged_lab.mdp as mdp
from legged_lab.assets.yrp03_v2.yrp03_v2 import YRP03_V2_CFG, AMP_LOADER_KWARGS
from legged_lab.envs.base.base_config import (
    ActionDelayCfg,
    BaseSceneCfg,
    CommandRangesCfg,
    CommandsCfg,
    DomainRandCfg,
    EventCfg,
    HeightScannerCfg,
    NoiseCfg,
    NoiseScalesCfg,
    NormalizationCfg,
    ObsScalesCfg,
    PhysxCfg,
    RobotCfg,
    SimCfg,
)
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG  # noqa:F401

_AMP_EXPERT = "legged_lab/envs/yrp03_v2/datasets/motion_amp_expert"
_AMP_VIS = "legged_lab/envs/yrp03_v2/datasets/motion_visualization"

# Shared by Yrp03V2MotionRunEnvCfg (env expert sampling) and Yrp03V2MotionRunAgentCfg (discriminator).
_AMP_MOTION_STAND = [f"{_AMP_EXPERT}/stand_ultra_yaw2.txt"]
_AMP_MOTION_WALK = [f"{_AMP_EXPERT}/walk_ultra_yaw2.txt"]
_AMP_MOTION_RUN = [
    f"{_AMP_EXPERT}/run_17200_9mps.txt",
    f"{_AMP_EXPERT}/collected_20260602_160257/env000_vxp15p00_vyp0p00_wp0p00.txt",
]
AMP_MOTION_FILES_DICT = {
    "style_0": _AMP_MOTION_STAND,
    "style_1": _AMP_MOTION_WALK,
    "style_2": _AMP_MOTION_RUN,
}


@configclass
class Yrp03V2RewardCfg:
    # ── Episode / survival ──
    alive = RewTerm(func=mdp.is_alive, weight=10.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # ── style_0: stand ──
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=10.0,
        params={
            # yrp03-v2 has 15 joints and Isaac's internal joint order is only known
            # at runtime, so we can't hardcode a positional target vector here.
            # None -> stand_still uses default_joint_pos (correct Isaac order), which
            # matches the original standing pose for legs/waist (arms stay at init 0.5).
            "target_angles": None,
            "style_mask": [0],
        },
    )
    feet_lin_vel_stand = RewTerm(
        func=mdp.feet_lin_vel_stand_neg,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_roll.*"),
            "horizontal_only": True,
            "relax_speed": 1.0,
            "style_mask": [0],
        },
    )

    # ── style_1: walk ──
    track_lin_vel_x_walk_exp = RewTerm(
        func=mdp.track_lin_vel_x_yaw_frame_exp,
        weight=5.0,
        params={
            "std": 0.3,
            "avg_vel_window_s": 0.6,
            "style_mask": [1],
        },
    )
    track_lin_vel_y_walk_neg = RewTerm(
        func=mdp.track_lin_vel_y_yaw_frame_neg,
        weight=-5.0,
        params={"avg_vel_window_s": 0.6, "style_mask": [1]},
    )
    feet_air_time_walk = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=10.0,
        params={
            "threshold": 0.4,
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["ankle_roll.*"]),
            "style_mask": [1],
        },
    )
    feet_clearance_walk = RewTerm(
        func=mdp.feet_clearance,
        weight=2.0,
        params={
            "target_height": 0.10,
            "std": 0.04,
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["ankle_roll.*"]),
            "style_mask": [1],
        },
    )

    # ── style_2: run ──
    track_lin_vel_x_run_exp = RewTerm(
        func=mdp.track_lin_vel_x_yaw_frame_exp,
        weight=15.0,
        params={
            "std": 0.8,
            "std_speed_max": 1.0,
            "speed_walk": 1.0,
            "speed_max": 15.0,
            "style_mask": [2],
        },
    )
    track_lin_vel_x_bonus = RewTerm(
        func=mdp.track_lin_vel_x_progress_bonus,
        weight=15.0,
        params={
            "speed_threshold": 2.0,
            "speed_max": 15.0,
            "normalize_speed": True,
            "style_mask": [2],
        },
    )
    track_ang_vel_z_bonus = RewTerm(
        func=mdp.track_ang_vel_z_world_bonus,
        weight=1.0,
        params={"std": 0.5, "avg_vel_window_s": 0.6, "style_mask": [2]},
    )
    body_orientation_speed_aware = RewTerm(
        func=mdp.body_orientation_speed_aware,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
            "pitch_coef": 0.025,
            "roll_coef": 0.07,
            "style_mask": [2],
        },
    )
    feet_descent_proximity_neg = RewTerm(
        func=mdp.feet_descent_proximity_neg,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_roll.*"),
            "foot_sole_offset": 0.074,
            "proximity_height_scale": 0.05,
            "style_mask": [2],
        },
    )
    # 前脚落脚点相对质心的前向位移：逼近目标（大步幅），落在质心后硬惩罚（会前倾失衡摔倒）。
    # 替换旧的 feet_air_time_run（奖励腾空时间会诱导"原地跳跃"退化解）。
    # target_stride：正向奖励的平移门控——本局平均步幅达到该值才全额发放，防"原地踏步领奖"。
    foot_strike_com_offset = RewTerm(
        func=mdp.foot_strike_com_offset,
        weight=5.0,
        params={
            "target_offset": 0.2,
            "std": 0.1,
            "behind_penalty": 2.0,
            "target_stride": 0.5,
            "style_mask": [2],
        },
    )
    # 腾空占比约束：超过 0.7 惩罚（腾空只是手段、落脚点才是目的），防跳跃/坠落。
    flight_ratio_neg = RewTerm(
        func=mdp.flight_ratio_neg,
        weight=-2.0,
        params={"max_flight_ratio": 0.7, "style_mask": [2]},
    )
    feet_clearance_run = RewTerm(
        func=mdp.feet_clearance,
        weight=2.0,
        params={
            "target_height": 0.15,
            "std": 0.05,
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["ankle_roll.*"]),
            "style_mask": [2],
        },
    )
    # 直接步频惩罚：局内平均步频超过 target_cadence（步/秒）时线性惩罚，压掉高频小碎步。
    cadence_neg = RewTerm(
        func=mdp.cadence_neg,
        weight=-2.0,
        params={"target_cadence": 3.0, "style_mask": [2]},
    )

    # ── style_1/style_2: locomotion command tracking ──
    track_lin_vel_y_exp = RewTerm(
        func=mdp.track_lin_vel_y_yaw_frame_exp,
        weight=5.0,
        params={"std": 0.3, "avg_vel_window_s": 0.6, "style_mask": [1, 2]},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=5.0,
        params={"std": 0.3, "avg_vel_window_s": 0.6, "style_mask": [1, 2]},
    )
    track_ang_vel_z_neg = RewTerm(
        func=mdp.track_ang_vel_z_world_neg,
        weight=-3.0,
        params={"avg_vel_window_s": 0.6, "style_mask": [1, 2]},
    )
    hip_yaw_pos_neg = RewTerm(
        func=mdp.hip_yaw_pos_neg,
        weight=-5.0,
        params={"yaw_cmd_relax_at": 0.5, "style_mask": [1, 2]},
    )
    # hip_yaw_action = RewTerm(
    #     func=mdp.hip_yaw_action,
    #     weight=-1.0,
    #     params={"yaw_cmd_relax_at": 0.5, "style_mask": [1, 2]},
    # )
    # hip_yaw_vel_cmd_neg = RewTerm(func=mdp.hip_yaw_vel_cmd_neg, weight=-0.05)

    # ── style_1/style_2: locomotion contact and spacing ──
    feet_contact_ang_vel_z = RewTerm(
        func=mdp.feet_contact_ang_vel_z,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_roll.*"),
            "min_contact_fz": 150.0,
            "max_penalty": 10.0,
            "style_mask": [1, 2],
        },
    )
    contact_impact_vel = RewTerm(
        func=mdp.contact_impact_vel,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_roll.*"),
            "style_mask": [1, 2],
            "pre_contact_policy_steps": 2,
        },
    )
    gait_feet_force_max_neg = RewTerm(
        func=mdp.gait_feet_force_max_neg,
        weight=-0.01,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["ankle_roll.*"]),
            "max_force": 2000.0,
            "style_mask": [1, 2],
        },
    )
    gait_feet_distance = RewTerm(
        func=mdp.gait_feet_distance,
        weight=-8.0,
        params={
            "target_y": 0.15,
            "speed_scale": 0.5,
            "speed_max": 15.0,
            "style_mask": [1, 2],
        },
    )
    knee_y_distance = RewTerm(
        func=mdp.knee_y_distance,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["knee_pitch_l_link", "knee_pitch_r_link"]),
            "target_y": 0.15,
            "speed_scale": 0.5,
            "speed_max": 15.0,
            "style_mask": [1, 2],
        },
    )
    feet_collision = RewTerm(
        func=mdp.feet_collision,
        weight=-30.0,
        params={"threshold_y": 0.14, "threshold_x": 0.24, "style_mask": [1, 2]},
    )

    # ── style_0/style_1: non-run posture ──
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0, params={"style_mask": [0, 1]})
    body_orientation_exp = RewTerm(
        func=mdp.body_orientation_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
            "style_mask": [0, 1],
        },
    )

    # ── style_0/style_1/style_2: shared task terms ──
    track_lin_vel_z_exp = RewTerm(
        func=mdp.track_lin_vel_z_yaw_frame_exp,
        weight=0.05,
        params={"std": 0.3, "style_mask": [0, 1, 2]},
    )
    track_lin_vel_x_neg = RewTerm(
        func=mdp.track_lin_vel_x_yaw_frame_neg,
        weight=-1.0,
        params={"style_mask": [0, 1, 2]},
    )
    base_height_neg = RewTerm(
        func=mdp.base_height_neg,
        weight=-10.0,
        params={"target_height": 1.15, "style_mask": [0, 1, 2]},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_roll.*"),
            "min_contact_fz": 50.0,
            "slip_speed_threshold": 0.05,
            "high_speed_slip_threshold": 0.45,
            "high_speed_relax": 3.0,
            "style_mask": [0, 1, 2],
        },
    )

    # ── Shared regularization / limits ──
    energy = RewTerm(func=mdp.energy, weight=-3e-4)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-3.0e-7)
    action_rate_weighted_l2 = RewTerm(
        func=mdp.action_rate_weighted_l2,
        weight=-0.01,
        params={"min_speed_scale": 0.4, "max_speed_scale": 2.0, "speed_relax": 6.0, "max_penalty": 200.0},
    )
    action_rate2_weighted_l2 = RewTerm(
        func=mdp.action_rate2_weighted_l2,
        weight=-0.005,
        params={"min_speed_scale": 0.4, "max_speed_scale": 2.0, "speed_relax": 6.0, "max_penalty": 400.0},
    )
    torques_weighted_neg = RewTerm(
        func=mdp.torques_weighted_neg,
        weight=-7.0e-6,
        params={"min_speed_scale": 0.4, "max_speed_scale": 2.0, "speed_relax": 6.0},
    )
    wholebody_vel_weighted_neg = RewTerm(
        func=mdp.wholebody_vel_weighted_neg,
        weight=-0.007,
        params={"min_speed_scale": 0.4, "max_speed_scale": 2.0, "speed_relax": 6.0},
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    joint_vel_limits_neg = RewTerm(
        func=mdp.joint_vel_limits_neg,
        weight=-2.0,
        params={"soft_limit": 0.9},
    )
    torque_limits_neg = RewTerm(
        func=mdp.torque_limits_neg,
        weight=-1.0,
        params={"soft_limit": 0.85},
    )
    ankle_pitch_pos_limits = RewTerm(func=mdp.ankle_pitch_pos_limits_neg, weight=-100.0)
    shoulder_pitch_pos_neg = RewTerm(
        func=mdp.shoulder_pitch_pos_neg,
        weight=-1.0,
        params={"relax_speed": 15.0},
    )
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)
    body_contact_neg = RewTerm(
        func=mdp.body_contact_neg,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(?!.*ankle_roll.*).*"),
            "threshold": 1.0,
        },
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["ankle_roll.*"])},
    )



@configclass
class StyleFromCommandCfg:
    """stand: |vx|,|vy|,|vyaw| 均 < stand_enter；否则 walk/run 按 |vx| 滞回。"""

    stand_enter: float = 0.05
    stand_exit: float = 0.08
    run_enter: float = 1.0  # 与 cmd_bucket_run_lin_vel_x_min 对齐
    run_exit: float = 0.85


@configclass
class CriticPrivilegeCfg:
    """Critic-only privileged features (aligned with ultra_run_nogait n_critic_extra)."""

    base_height_target: float = 1.15
    foot_height_offset: float = 0.074
    height_scale: float = 1.0
    foot_force_scale: float = 0.001
    refresh_material_on_reset: bool = False


@configclass
class Yrp03V2EventCfg(EventCfg):
    randomize_joint_pos_bias = EventTerm(
        func=mdp.randomize_joint_pos_bias,
        mode="reset",
        params={"bias_range": (-0.03, 0.03)},
    )
    randomize_joint_armature = None
    randomize_joint_friction = None
    randomize_actuator_friction = None


@configclass
class Yrp03V2MotionRunEnvCfg:
    viewer: ViewerCfg = ViewerCfg(
        origin_type="asset_body",
        env_index=0,
        asset_name="robot",
        body_name="pelvis",
        eye=(5.0, 0.0, 2.0),
        lookat=(0.0, 0.0, 0.0),
    )
    num_styles = 3
    # y / yaw 速度跟踪奖励用 step_dt/window 的 EMA（有效窗口约 window 秒）
    track_avg_vel_window_s: float = 0.5
    style_from_command: StyleFromCommandCfg = StyleFromCommandCfg()
    amp_motion_files_display = [f"{_AMP_VIS}/run_14mps_play_newultra.txt"]
    amp_motion_files_dict = AMP_MOTION_FILES_DICT
    amp_mirror_expert_motion: bool = True
    # 15-DoF / 42-dim AMP obs: 6-DoF legs (ankle_roll per leg), 与 agent cfg 保持一致。
    amp_loader_kwargs: dict = AMP_LOADER_KWARGS
    device: str = "cuda:0"
    critic_privilege: CriticPrivilegeCfg = CriticPrivilegeCfg()
    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=20.0,
        num_envs=4096,
        env_spacing=2.5,
        robot=YRP03_V2_CFG,
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=0,
        terrain_restitution=0.03,
        # 平地：关 base 221 维扫描省算力；开脚下小网格算 foot_clearance。
        # 粗糙地形：enable_height_scan=True，并可设 terrain_generator。
        height_scanner=HeightScannerCfg(
            enable_height_scan=False,
            prim_body_name="pelvis",
            resolution=0.1,
            size=(1.2, 1.6),
            debug_vis=False,
            drift_range=(0.0, 0.0),
            enable_feet_height_scan=True,
            feet_body_names=("ankle_roll_l_link", "ankle_roll_r_link"),
            feet_scan_resolution=0.05,
            feet_scan_size=(0.30, 0.10),
        ),
    )
    robot: RobotCfg = RobotCfg(
        # actor_obs_history_length=20,
        actor_obs_history_length=10,
        critic_obs_history_length=5,
        action_scale=0.25,
        terminate_contacts_body_names=["knee_pitch.*", "shoulder_pitch.*", "pelvis", "waist_yaw_link"],
        feet_body_names=[".*ankle_roll.*"],
    )
    reward = Yrp03V2RewardCfg()
    normalization: NormalizationCfg = NormalizationCfg(
        obs_scales=ObsScalesCfg(
            lin_vel=0.2,
            ang_vel=0.5,
            projected_gravity=1.0,
            commands=(0.2, 1.0, 1.0),  # x, y, yaw
            joint_pos=1.0,
            joint_vel=0.05,
            actions=1.0,
            height_scan=1.0,
        ),
        clip_observations=100.0,
        clip_actions=100.0,
        height_scan_offset=0.5,
    )
    commands: CommandsCfg = CommandsCfg(
        resampling_time_range=(7.0, 13.0),
        rel_standing_envs=0.0,  # 分桶采样时由 cmd_bucket_probs 控制站定比例
        cmd_bucket_probs=(1.0, 3.0, 6.0),  # stand : walk : run = 1 : 3 : 6
        cmd_bucket_walk_lin_vel_x=(-0.5, 1.0),
        cmd_bucket_walk_lateral_yaw_frac=0.7,
        cmd_bucket_run_lin_vel_x_min=1.0,
        rel_heading_envs=0.6,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=CommandRangesCfg(
            # 初始 vmax；勿与 curriculum_max 相同，否则 resume 易顶速开训
            lin_vel_x=(-0.5, 8.6),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-0.5, 0.5),
            heading=(-math.pi / 2, math.pi / 2),
        ),
        use_curriculum=True,
        # P3: 回调升速门槛（0.35→0.38 / 0.30→0.32），避免过松把速度推到策略无法稳定维持的水平。
        # 上次 12.4m/s 处 track≈4.5 已贴着 gate≈4.70，升速后崩溃。
        curriculum_rwd_threshold=0.38,
        curriculum_rwd_threshold_min=0.32,
        curriculum_rwd_threshold_v_lo=5.0,
        curriculum_rwd_threshold_v_hi=15.0,
        curriculum_update_step=0.2,
        curriculum_max_lin_vel_x=(-0.5, 15.0),
        curriculum_min_mean_episode_ratio=0.8,
        run_high_speed_sample_frac=0.5,
        run_high_speed_threshold_frac=0.7,
        cmd_step_sample_frac=0.3,
        max_lin_accel=5.0,
    )
    noise: NoiseCfg = NoiseCfg(
        add_noise=True,
        noise_scales=NoiseScalesCfg(
            ang_vel=0.3,
            projected_gravity=0.05,
            joint_pos=0.01,
            joint_vel=1.5,
            height_scan=0.02,
        ),
    )
    domain_rand: DomainRandCfg = DomainRandCfg(
        events=Yrp03V2EventCfg(
            physics_material=EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "static_friction_range": (0.4, 1.0),
                    "dynamic_friction_range": (0.4, 0.8),
                    "restitution_range": (0.0, 0.3),
                    "num_buckets": 64,
                },
            ),
            add_base_mass=EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
                    "mass_distribution_params": (-3.0, 3.0),
                    "operation": "add",
                },
            ),
            add_others_mass=EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names="^(?!.*waist_yaw_link).*"),
                    "mass_distribution_params": (0.8, 1.2),
                    "operation": "scale",
                },
            ),
            randomize_rigid_body_com=EventTerm(
                func=mdp.randomize_rigid_body_com,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
                    "com_range": {"x": (-0.05, 0.05), "y": (-0.03, 0.03), "z": (-0.05, 0.05)},
                },
            ),
            randomize_rigid_body_inertia=EventTerm(
                func=mdp.randomize_rigid_body_inertia,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "inertia_distribution_params": (0.8, 1.2),
                    "operation": "scale",
                },
            ),
            randomize_joint_armature=EventTerm(
                func=mdp.randomize_joint_armature,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                    "armature_distribution_params": (0.8, 1.2),
                },
            ),
            randomize_actuator_friction=EventTerm(
                func=mdp.randomize_actuator_friction,
                mode="startup",
                params={
                    "static_friction_range": (0.8, 1.2),
                    "dynamic_friction_range": (0.8, 1.2),
                },
            ),
            randomize_motor_strength=EventTerm(
                func=mdp.randomize_motor_strength,
                mode="startup",
                params={"motor_strength_range": (0.7, 1.2)},
            ),
            randomize_actuator_gains=EventTerm(
                func=mdp.randomize_actuator_gains,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "stiffness_distribution_params": (0.7, 1.2),
                    "damping_distribution_params": (0.7, 1.2),
                    "operation": "scale",
                    "distribution": "uniform",
                },
            ),
            reset_base=EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                    "velocity_range": {
                        "x": (-1.0, 1.0),
                        "y": (-0.5, 0.5),
                        "z": (-0.5, 0.5),
                        "roll": (-0.5, 0.5),
                        "pitch": (-0.5, 0.5),
                        "yaw": (-0.5, 0.5),
                    },
                },
            ),
            reset_robot_joints=EventTerm(
                func=mdp.reset_joints_by_scale,
                mode="reset",
                params={
                    "position_range": (0.5, 1.5),
                    "velocity_range": (0.0, 0.0),
                },
            ),
            push_robot=EventTerm(
                func=mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(5.0, 10.0),
                params={
                    "velocity_range": {
                        "x": (-1.0, 1.0),
                        "y": (-0.5, 0.5),
                        "z": (-0.5, 0.5),
                        "roll": (-0.5, 0.5),
                        "pitch": (-0.5, 0.5),
                        "yaw": (-0.5, 0.5),
                    }
                },
            ),
        ),
        action_delay=ActionDelayCfg(enable=True, params={"max_delay": 1, "min_delay": 0}),
    )
    # physics 200 Hz (dt=0.005), policy/env step 100 Hz (step_dt=0.01)
    sim: SimCfg = SimCfg(dt=0.0025, decimation=4, physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15))


@configclass
class Yrp03V2PpoActorCriticCfg(RslRlPpoActorCriticCfg):
    """HIM actor: last N frames from obs history fed to actor MLP (0..history_length)."""

    actor_recent_frames: int = 5
    him_encoder_type: str = "gru"
    him_gru_hidden_dim: int = 64
    him_gru_num_layers: int = 1
    him_gru_head_hidden_dims: tuple[int, ...] = (64,)


@configclass
class Yrp03V2PpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Extra knobs for Multi-AMP/HIM update batch sizes."""

    amp_mini_batch_size: int = 4096
    estimator_mini_batch_size: int = 4096


@configclass
class Yrp03V2MotionRunAgentCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 32
    max_iterations = 20000
    empirical_normalization = False
    use_wgan_discriminator = True
    policy = Yrp03V2PpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = Yrp03V2PpoAlgorithmCfg(
        class_name="MULTIAMPPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg={
            "use_data_augmentation": False,
            "use_mirror_loss": True,
            "data_augmentation_func": "legged_lab.mdp.symmetry:data_augmentation_func",
            "mirror_loss_coeff": 10.0,
        },
        rnd_cfg=None,
    )
    clip_actions = None
    save_interval = 200
    runner_class_name = "MultiAmpOnPolicyRunner"
    experiment_name = "yrp03_v2_run"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "yrp03_v2_run"
    wandb_project = "yrp03_v2_run"

    # Multi-AMP (motion list: AMP_MOTION_FILES_DICT at module top)
    amp_motion_files_dict = AMP_MOTION_FILES_DICT
    # yrp03-v2 AMP obs is 15-DoF / 42-dim, so the runner must build its expert
    # loader + discriminators with the 6-DoF-leg layout (not the default 5-DoF),
    # otherwise the 42 vs 38 dim mismatch crashes on the first rollout.
    amp_loader_kwargs = AMP_LOADER_KWARGS
    amp_mirror_expert_motion = True
    amp_reward_coef_dict = {
        "style_0": 1.0,
        "style_1": 0.8,
        "style_2": 0.4,
    }
    amp_num_preload_transitions = 200000
    amp_task_reward_lerp = 0.8
    amp_discr_hidden_dims = [512, 256]
    min_normalized_std = [0.05] * 15

    resume = False  # GRU encoder is incompatible with MLP checkpoints
    # load_run = ".*" # '2026-05-31_00-03-26' # '2026-05-29_11-03-16' #
    # load_checkpoint = "model_5000.pt"
    load_run="2026-08-15_13-33-08"
    load_checkpoint = "model_.*.pt"