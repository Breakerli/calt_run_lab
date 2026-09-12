# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""tiangong2plus (天工2.0plus) 人形机器人 Articulation 配置。

21 DoF: 腿 6x2 (roll/pitch/yaw + knee_pitch + ankle_pitch/roll, 链序 roll->pitch->yaw,
与 yrp03/g1 的 yaw->roll->pitch 不同, 但 AMP 特征段按语义序 [yaw,roll,pitch,knee,
ankle_pitch,ankle_roll] 取, 不受影响) + 腰 waist_yaw 1 + 臂 4x2 (shoulder
pitch/roll/yaw + elbow_pitch)。root body ``pelvis``, 脚为 ``ankle_roll_*_link``。

effort/velocity limit 取自 tiangong2_plus_urdf.urdf (与关键参数 CSV 一致);
stiffness/damping/armature/力矩曲线暂无实测整定, 沿用 yrp03_v2 的基线 (可调)。
站立 pelvis 高 0.96 = 站立角 (hip_pitch -15°/knee +30°/ankle -15°) 腿 FK
(ankle_roll 原点 -0.9044) + 脚掌网格偏置 (sole 在 ankle_roll 原点下 0.057)。
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab.actuators import CustomDelayedPDActuatorCfg
from legged_lab.assets import ISAAC_ASSET_DIR

MIN_ACTUATOR_DELAY = 0
MAX_ACTUATOR_DELAY = 4

# --- armature (tunable baseline, 沿用 yrp03_v2) ---
ARMATURE_HIP_YAW = 0.048865
ARMATURE_HIP_ROLL = 0.40318
ARMATURE_HIP_PITCH = 0.40318
ARMATURE_KNEE_PITCH = 0.40318
ARMATURE_ANKLE_PITCH = 0.21448
ARMATURE_ANKLE_ROLL = 0.1
ARMATURE_SHOULDER = 0.041125
ARMATURE_ELBOW = 0.041125
ARMATURE_WAIST_YAW = 0.048865

# --- effort / velocity limit (tiangong2_plus_urdf.urdf <limit> 原值) ---
EFFORT_LIMIT_HIP_YAW = 235.0
EFFORT_LIMIT_HIP_ROLL = 235.0
EFFORT_LIMIT_HIP_PITCH = 330.0
EFFORT_LIMIT_KNEE_PITCH = 330.0
EFFORT_LIMIT_ANKLE_PITCH = 55.0
EFFORT_LIMIT_ANKLE_ROLL = 55.0
EFFORT_LIMIT_SHOULDER = 55.0
EFFORT_LIMIT_ELBOW = 55.0
EFFORT_LIMIT_WAIST_YAW = 91.0

VELOCITY_LIMIT_HIP_YAW = 16.755160819145562
VELOCITY_LIMIT_HIP_ROLL = 16.755160819145562
VELOCITY_LIMIT_HIP_PITCH = 15.707963267948966
VELOCITY_LIMIT_KNEE_PITCH = 15.707963267948966
VELOCITY_LIMIT_ANKLE_PITCH = 14.137166941154069
VELOCITY_LIMIT_ANKLE_ROLL = 14.137166941154069
VELOCITY_LIMIT_SHOULDER = 14.137166941154069
VELOCITY_LIMIT_ELBOW = 14.137166941154069
VELOCITY_LIMIT_WAIST_YAW = 9.215338450530059

# --- stiffness / damping (tunable baseline, 沿用 yrp03_v2) ---
STIFFNESS_HIP_YAW = 500.0
STIFFNESS_HIP_ROLL = 700.0
STIFFNESS_HIP_PITCH = 700.0
STIFFNESS_KNEE_PITCH = 700.0
STIFFNESS_ANKLE_PITCH = 100.0
STIFFNESS_ANKLE_ROLL = 40.0
STIFFNESS_SHOULDER = 50.0
STIFFNESS_ELBOW = 50.0
STIFFNESS_WAIST_YAW = 400.0

DAMPING_HIP_YAW = 10.0
DAMPING_HIP_ROLL = 20.0
DAMPING_HIP_PITCH = 10.0
DAMPING_KNEE_PITCH = 10.0
DAMPING_ANKLE_PITCH = 6.0
DAMPING_ANKLE_ROLL = 2.0
DAMPING_SHOULDER = 3.0
DAMPING_ELBOW = 3.0
DAMPING_WAIST_YAW = 8.0

# --- 力矩曲线 (tunable baseline, 沿用 yrp03_v2) ---
TORQUE_CURVE_X1_HIP_YAW = 10.0
TORQUE_CURVE_X1_HIP_ROLL = 12.0
TORQUE_CURVE_X1_HIP_PITCH = 17.0
TORQUE_CURVE_X1_KNEE_PITCH = 17.0
TORQUE_CURVE_X1_ANKLE_PITCH = 17.0
TORQUE_CURVE_X1_ANKLE_ROLL = 17.0
TORQUE_CURVE_X1_SHOULDER = 13.0
TORQUE_CURVE_X1_ELBOW = 13.0
TORQUE_CURVE_X1_WAIST_YAW = 10.0

TORQUE_CURVE_Y1_HIP_YAW = 150.0
TORQUE_CURVE_Y1_HIP_ROLL = 150.0
TORQUE_CURVE_Y1_HIP_PITCH = 600.0
TORQUE_CURVE_Y1_KNEE_PITCH = 600.0
TORQUE_CURVE_Y1_ANKLE_PITCH = 55.0
TORQUE_CURVE_Y1_ANKLE_ROLL = 55.0
TORQUE_CURVE_Y1_SHOULDER = 55.0
TORQUE_CURVE_Y1_ELBOW = 55.0
TORQUE_CURVE_Y1_WAIST_YAW = 91.0

TIANGONG2PLUS_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/tiangong2plus/usd/tiangong2_plus_urdf.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.96),  # 站立角腿 FK + 脚掌网格偏置 (见模块 docstring)
        joint_pos={
            "knee_pitch_.*": math.radians(30),
            "hip_pitch_.*": math.radians(-15),
            "ankle_pitch_.*": math.radians(-15),
            "hip_yaw_.*": 0.0,
            "hip_roll_.*": 0.0,
            "ankle_roll_.*": 0.0,
            "waist_yaw_joint": 0.0,
            "shoulder_pitch_.*": 0.5,
            "shoulder_roll_.*": 0.0,
            "shoulder_yaw_.*": 0.0,
            "elbow_pitch_.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": CustomDelayedPDActuatorCfg(
            joint_names_expr=[
                "hip_yaw_.*",
                "hip_roll_.*",
                "hip_pitch_.*",
                "knee_pitch_.*",
            ],
            effort_limit_sim={
                "hip_yaw_.*": EFFORT_LIMIT_HIP_YAW,
                "hip_roll_.*": EFFORT_LIMIT_HIP_ROLL,
                "hip_pitch_.*": EFFORT_LIMIT_HIP_PITCH,
                "knee_pitch_.*": EFFORT_LIMIT_KNEE_PITCH,
            },
            velocity_limit_sim={
                "hip_yaw_.*": VELOCITY_LIMIT_HIP_YAW,
                "hip_roll_.*": VELOCITY_LIMIT_HIP_ROLL,
                "hip_pitch_.*": VELOCITY_LIMIT_HIP_PITCH,
                "knee_pitch_.*": VELOCITY_LIMIT_KNEE_PITCH,
            },
            stiffness={
                "hip_yaw_.*": STIFFNESS_HIP_YAW,
                "hip_roll_.*": STIFFNESS_HIP_ROLL,
                "hip_pitch_.*": STIFFNESS_HIP_PITCH,
                "knee_pitch_.*": STIFFNESS_KNEE_PITCH,
            },
            damping={
                "hip_yaw_.*": DAMPING_HIP_YAW,
                "hip_roll_.*": DAMPING_HIP_ROLL,
                "hip_pitch_.*": DAMPING_HIP_PITCH,
                "knee_pitch_.*": DAMPING_KNEE_PITCH,
            },
            armature={
                "hip_yaw_.*": ARMATURE_HIP_YAW,
                "hip_roll_.*": ARMATURE_HIP_ROLL,
                "hip_pitch_.*": ARMATURE_HIP_PITCH,
                "knee_pitch_.*": ARMATURE_KNEE_PITCH,
            },
            min_delay=MIN_ACTUATOR_DELAY,
            max_delay=MAX_ACTUATOR_DELAY,
            X1={
                "hip_yaw_.*": TORQUE_CURVE_X1_HIP_YAW,
                "hip_roll_.*": TORQUE_CURVE_X1_HIP_ROLL,
                "hip_pitch_.*": TORQUE_CURVE_X1_HIP_PITCH,
                "knee_pitch_.*": TORQUE_CURVE_X1_KNEE_PITCH,
            },
            X2={
                "hip_yaw_.*": VELOCITY_LIMIT_HIP_YAW,
                "hip_roll_.*": VELOCITY_LIMIT_HIP_ROLL,
                "hip_pitch_.*": VELOCITY_LIMIT_HIP_PITCH,
                "knee_pitch_.*": VELOCITY_LIMIT_KNEE_PITCH,
            },
            Y1={
                "hip_yaw_.*": TORQUE_CURVE_Y1_HIP_YAW,
                "hip_roll_.*": TORQUE_CURVE_Y1_HIP_ROLL,
                "hip_pitch_.*": TORQUE_CURVE_Y1_HIP_PITCH,
                "knee_pitch_.*": TORQUE_CURVE_Y1_KNEE_PITCH,
            },
            Y2={
                "hip_yaw_.*": EFFORT_LIMIT_HIP_YAW,
                "hip_roll_.*": EFFORT_LIMIT_HIP_ROLL,
                "hip_pitch_.*": EFFORT_LIMIT_HIP_PITCH,
                "knee_pitch_.*": EFFORT_LIMIT_KNEE_PITCH,
            },
        ),
        "ankle_pitch": CustomDelayedPDActuatorCfg(
            joint_names_expr=["ankle_pitch_.*"],
            effort_limit_sim={"ankle_pitch_.*": EFFORT_LIMIT_ANKLE_PITCH},
            velocity_limit_sim={"ankle_pitch_.*": VELOCITY_LIMIT_ANKLE_PITCH},
            stiffness={"ankle_pitch_.*": STIFFNESS_ANKLE_PITCH},
            damping={"ankle_pitch_.*": DAMPING_ANKLE_PITCH},
            armature={"ankle_pitch_.*": ARMATURE_ANKLE_PITCH},
            min_delay=MIN_ACTUATOR_DELAY,
            max_delay=MAX_ACTUATOR_DELAY,
            X1={"ankle_pitch_.*": TORQUE_CURVE_X1_ANKLE_PITCH},
            X2={"ankle_pitch_.*": VELOCITY_LIMIT_ANKLE_PITCH},
            Y1={"ankle_pitch_.*": TORQUE_CURVE_Y1_ANKLE_PITCH},
            Y2={"ankle_pitch_.*": EFFORT_LIMIT_ANKLE_PITCH},
        ),
        "ankle_roll": CustomDelayedPDActuatorCfg(
            joint_names_expr=["ankle_roll_.*"],
            effort_limit_sim={"ankle_roll_.*": EFFORT_LIMIT_ANKLE_ROLL},
            velocity_limit_sim={"ankle_roll_.*": VELOCITY_LIMIT_ANKLE_ROLL},
            stiffness={"ankle_roll_.*": STIFFNESS_ANKLE_ROLL},
            damping={"ankle_roll_.*": DAMPING_ANKLE_ROLL},
            armature={"ankle_roll_.*": ARMATURE_ANKLE_ROLL},
            min_delay=MIN_ACTUATOR_DELAY,
            max_delay=MAX_ACTUATOR_DELAY,
            X1={"ankle_roll_.*": TORQUE_CURVE_X1_ANKLE_ROLL},
            X2={"ankle_roll_.*": VELOCITY_LIMIT_ANKLE_ROLL},
            Y1={"ankle_roll_.*": TORQUE_CURVE_Y1_ANKLE_ROLL},
            Y2={"ankle_roll_.*": EFFORT_LIMIT_ANKLE_ROLL},
        ),
        "arms": CustomDelayedPDActuatorCfg(
            joint_names_expr=["shoulder_pitch_.*", "shoulder_roll_.*", "shoulder_yaw_.*", "elbow_pitch_.*"],
            effort_limit_sim={
                "shoulder_pitch_.*": EFFORT_LIMIT_SHOULDER,
                "shoulder_roll_.*": EFFORT_LIMIT_SHOULDER,
                "shoulder_yaw_.*": EFFORT_LIMIT_SHOULDER,
                "elbow_pitch_.*": EFFORT_LIMIT_ELBOW,
            },
            velocity_limit_sim={
                "shoulder_pitch_.*": VELOCITY_LIMIT_SHOULDER,
                "shoulder_roll_.*": VELOCITY_LIMIT_SHOULDER,
                "shoulder_yaw_.*": VELOCITY_LIMIT_SHOULDER,
                "elbow_pitch_.*": VELOCITY_LIMIT_ELBOW,
            },
            stiffness={
                "shoulder_pitch_.*": STIFFNESS_SHOULDER,
                "shoulder_roll_.*": STIFFNESS_SHOULDER,
                "shoulder_yaw_.*": STIFFNESS_SHOULDER,
                "elbow_pitch_.*": STIFFNESS_ELBOW,
            },
            damping={
                "shoulder_pitch_.*": DAMPING_SHOULDER,
                "shoulder_roll_.*": DAMPING_SHOULDER,
                "shoulder_yaw_.*": DAMPING_SHOULDER,
                "elbow_pitch_.*": DAMPING_ELBOW,
            },
            armature={
                "shoulder_pitch_.*": ARMATURE_SHOULDER,
                "shoulder_roll_.*": ARMATURE_SHOULDER,
                "shoulder_yaw_.*": ARMATURE_SHOULDER,
                "elbow_pitch_.*": ARMATURE_ELBOW,
            },
            min_delay=MIN_ACTUATOR_DELAY,
            max_delay=MAX_ACTUATOR_DELAY,
            X1={
                "shoulder_pitch_.*": TORQUE_CURVE_X1_SHOULDER,
                "shoulder_roll_.*": TORQUE_CURVE_X1_SHOULDER,
                "shoulder_yaw_.*": TORQUE_CURVE_X1_SHOULDER,
                "elbow_pitch_.*": TORQUE_CURVE_X1_ELBOW,
            },
            X2={
                "shoulder_pitch_.*": VELOCITY_LIMIT_SHOULDER,
                "shoulder_roll_.*": VELOCITY_LIMIT_SHOULDER,
                "shoulder_yaw_.*": VELOCITY_LIMIT_SHOULDER,
                "elbow_pitch_.*": VELOCITY_LIMIT_ELBOW,
            },
            Y1={
                "shoulder_pitch_.*": TORQUE_CURVE_Y1_SHOULDER,
                "shoulder_roll_.*": TORQUE_CURVE_Y1_SHOULDER,
                "shoulder_yaw_.*": TORQUE_CURVE_Y1_SHOULDER,
                "elbow_pitch_.*": TORQUE_CURVE_Y1_ELBOW,
            },
            Y2={
                "shoulder_pitch_.*": EFFORT_LIMIT_SHOULDER,
                "shoulder_roll_.*": EFFORT_LIMIT_SHOULDER,
                "shoulder_yaw_.*": EFFORT_LIMIT_SHOULDER,
                "elbow_pitch_.*": EFFORT_LIMIT_ELBOW,
            },
        ),
        "waist": CustomDelayedPDActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim={"waist_yaw_joint": EFFORT_LIMIT_WAIST_YAW},
            velocity_limit_sim={"waist_yaw_joint": VELOCITY_LIMIT_WAIST_YAW},
            stiffness={"waist_yaw_joint": STIFFNESS_WAIST_YAW},
            damping={"waist_yaw_joint": DAMPING_WAIST_YAW},
            armature={"waist_yaw_joint": ARMATURE_WAIST_YAW},
            min_delay=MIN_ACTUATOR_DELAY,
            max_delay=MAX_ACTUATOR_DELAY,
            X1={"waist_yaw_joint": TORQUE_CURVE_X1_WAIST_YAW},
            X2={"waist_yaw_joint": VELOCITY_LIMIT_WAIST_YAW},
            Y1={"waist_yaw_joint": TORQUE_CURVE_Y1_WAIST_YAW},
            Y2={"waist_yaw_joint": EFFORT_LIMIT_WAIST_YAW},
        ),
    },
)


# AMP expert feature layout (与 yrp03_v2/g1 相同的 15 特征关节布局):
# right_arm(1) + left_arm(1) + right_leg(6) + left_leg(6) + waist(1) = 15 -> 42 维。
# leg 段语义序 [hip_yaw, hip_roll, hip_pitch, knee_pitch, ankle_pitch, ankle_roll]
# (find_joints name_keys 保证, 与 URDF 链序 roll->pitch->yaw 无关);
# leg_mirror_flip_indices: hip_yaw(0), hip_roll(1), ankle_roll(5) 镜像变号。
AMP_LOADER_KWARGS = {
    "leg_joint_size": 6,
    "leg_mirror_flip_indices": (0, 1, 5),
}
