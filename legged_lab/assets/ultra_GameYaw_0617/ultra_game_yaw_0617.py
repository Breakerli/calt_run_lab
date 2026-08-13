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

"""Ultra GameYaw humanoid articulation configuration."""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab.actuators import CustomDelayedPDActuatorCfg
from legged_lab.assets import ISAAC_ASSET_DIR

ARMATURE_HIP_YAW = 0.048865
ARMATURE_HIP_ROLL = 0.40318
ARMATURE_HIP_PITCH = 0.40318
ARMATURE_KNEE_PITCH = 0.40318
ARMATURE_ANKLE_PITCH = 0.21448
ARMATURE_SHOULDER_PITCH = 0.041125
ARMATURE_WAIST_YAW = 0.048865

EFFORT_LIMIT_HIP_YAW = 250
EFFORT_LIMIT_HIP_ROLL = 657
EFFORT_LIMIT_HIP_PITCH = 657
EFFORT_LIMIT_KNEE_PITCH = 657
EFFORT_LIMIT_ANKLE_PITCH = 300
EFFORT_LIMIT_SHOULDER_PITCH = 140
EFFORT_LIMIT_WAIST_YAW = 250

VELOCITY_LIMIT_HIP_YAW = 20.9
VELOCITY_LIMIT_HIP_ROLL = 18.8
VELOCITY_LIMIT_HIP_PITCH = 26.1
VELOCITY_LIMIT_KNEE_PITCH = 26.1
VELOCITY_LIMIT_ANKLE_PITCH = 30
VELOCITY_LIMIT_SHOULDER_PITCH = 26.1
VELOCITY_LIMIT_WAIST_YAW = 20.9

def _natural_freq(freq_hz: float) -> float:
    return freq_hz * 2.0 * math.pi


NATURAL_FREQ_HZ_HIP_YAW = 14.5
NATURAL_FREQ_HZ_HIP_ROLL = 6.5
NATURAL_FREQ_HZ_HIP_PITCH = 6.5
NATURAL_FREQ_HZ_KNEE_PITCH = 6.5
NATURAL_FREQ_HZ_ANKLE_PITCH = 3.5
NATURAL_FREQ_HZ_SHOULDER_PITCH = 5.0
NATURAL_FREQ_HZ_WAIST_YAW = 10.0

DAMPING_RATIO_HIP_YAW = 1.5
DAMPING_RATIO_HIP_ROLL = 1.2
DAMPING_RATIO_HIP_PITCH = 1.2
DAMPING_RATIO_KNEE_PITCH = 1.2
DAMPING_RATIO_ANKLE_PITCH = 1.3
DAMPING_RATIO_SHOULDER_PITCH = 1.5
DAMPING_RATIO_WAIST_YAW = 1.5

NATURAL_FREQ_HIP_YAW = _natural_freq(NATURAL_FREQ_HZ_HIP_YAW)
NATURAL_FREQ_HIP_ROLL = _natural_freq(NATURAL_FREQ_HZ_HIP_ROLL)
NATURAL_FREQ_HIP_PITCH = _natural_freq(NATURAL_FREQ_HZ_HIP_PITCH)
NATURAL_FREQ_KNEE_PITCH = _natural_freq(NATURAL_FREQ_HZ_KNEE_PITCH)
NATURAL_FREQ_ANKLE_PITCH = _natural_freq(NATURAL_FREQ_HZ_ANKLE_PITCH)
NATURAL_FREQ_SHOULDER_PITCH = _natural_freq(NATURAL_FREQ_HZ_SHOULDER_PITCH)
NATURAL_FREQ_WAIST_YAW = _natural_freq(NATURAL_FREQ_HZ_WAIST_YAW)


STIFFNESS_HIP_YAW = ARMATURE_HIP_YAW * NATURAL_FREQ_HIP_YAW**2
STIFFNESS_HIP_ROLL = ARMATURE_HIP_ROLL * NATURAL_FREQ_HIP_ROLL**2
STIFFNESS_HIP_PITCH = ARMATURE_HIP_PITCH * NATURAL_FREQ_HIP_PITCH**2
STIFFNESS_KNEE_PITCH = ARMATURE_KNEE_PITCH * NATURAL_FREQ_KNEE_PITCH**2
STIFFNESS_ANKLE_PITCH = ARMATURE_ANKLE_PITCH * NATURAL_FREQ_ANKLE_PITCH**2
STIFFNESS_SHOULDER_PITCH = ARMATURE_SHOULDER_PITCH * NATURAL_FREQ_SHOULDER_PITCH**2
STIFFNESS_WAIST_YAW = ARMATURE_WAIST_YAW * NATURAL_FREQ_WAIST_YAW**2
# Original stiffness settings:
STIFFNESS_HIP_YAW = 500
STIFFNESS_HIP_ROLL = 700
STIFFNESS_HIP_PITCH = 700
STIFFNESS_KNEE_PITCH = 700
STIFFNESS_ANKLE_PITCH = 100.0
STIFFNESS_SHOULDER_PITCH = 50.0
STIFFNESS_WAIST_YAW = 400.0


DAMPING_HIP_YAW = 2.0 * DAMPING_RATIO_HIP_YAW * ARMATURE_HIP_YAW * NATURAL_FREQ_HIP_YAW
DAMPING_HIP_ROLL = 2.0 * DAMPING_RATIO_HIP_ROLL * ARMATURE_HIP_ROLL * NATURAL_FREQ_HIP_ROLL
DAMPING_HIP_PITCH = 2.0 * DAMPING_RATIO_HIP_PITCH * ARMATURE_HIP_PITCH * NATURAL_FREQ_HIP_PITCH
DAMPING_KNEE_PITCH = 2.0 * DAMPING_RATIO_KNEE_PITCH * ARMATURE_KNEE_PITCH * NATURAL_FREQ_KNEE_PITCH
DAMPING_ANKLE_PITCH = 2.0 * DAMPING_RATIO_ANKLE_PITCH * ARMATURE_ANKLE_PITCH * NATURAL_FREQ_ANKLE_PITCH
DAMPING_SHOULDER_PITCH = (
    2.0 * DAMPING_RATIO_SHOULDER_PITCH * ARMATURE_SHOULDER_PITCH * NATURAL_FREQ_SHOULDER_PITCH
)
DAMPING_WAIST_YAW = 2.0 * DAMPING_RATIO_WAIST_YAW * ARMATURE_WAIST_YAW * NATURAL_FREQ_WAIST_YAW

# Original damping settings:
DAMPING_HIP_YAW = 10
DAMPING_HIP_ROLL = 20
DAMPING_HIP_PITCH = 10
DAMPING_KNEE_PITCH = 10
DAMPING_ANKLE_PITCH = 6.0
DAMPING_SHOULDER_PITCH = 3.0
DAMPING_WAIST_YAW = 8.0


MIN_ACTUATOR_DELAY = 0
MAX_ACTUATOR_DELAY = 4

TORQUE_CURVE_X1_HIP_YAW = 10.0
TORQUE_CURVE_X1_HIP_ROLL = 12.0
TORQUE_CURVE_X1_HIP_PITCH = 17.0
TORQUE_CURVE_X1_KNEE_PITCH = 17.0
TORQUE_CURVE_X1_ANKLE_PITCH = 17.0
TORQUE_CURVE_X1_SHOULDER_PITCH = 13.0
TORQUE_CURVE_X1_WAIST_YAW = 10.0

TORQUE_CURVE_Y1_HIP_YAW = 150
TORQUE_CURVE_Y1_HIP_ROLL = 600
TORQUE_CURVE_Y1_HIP_PITCH = 600
TORQUE_CURVE_Y1_KNEE_PITCH = 600
TORQUE_CURVE_Y1_ANKLE_PITCH = 300
TORQUE_CURVE_Y1_SHOULDER_PITCH = 100
TORQUE_CURVE_Y1_WAIST_YAW = 150

ULTRA_GAME_YAW_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/ultra_GameYaw_0617/usd/ultra_game_yaw_0617.usd",
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
        pos=(0.0, 0.0, 1.23),
        joint_pos={
            "knee_pitch_l_joint": math.radians(30),
            "knee_pitch_r_joint": math.radians(30),
            "hip_pitch_l_joint": math.radians(-15),
            "hip_pitch_r_joint": math.radians(-15),
            "ankle_pitch_l_joint": math.radians(-15),
            "ankle_pitch_r_joint": math.radians(-15),
            "hip_roll_l_joint": 0.0,
            "hip_roll_r_joint": 0.0,
            "hip_yaw_l_joint": 0.0,
            "hip_yaw_r_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "shoulder_pitch_l_joint": 0.5,
            "shoulder_pitch_r_joint": 0.5,
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
        "feet": CustomDelayedPDActuatorCfg(
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
        "arms": CustomDelayedPDActuatorCfg(
            joint_names_expr=["shoulder_pitch_.*"],
            effort_limit_sim={"shoulder_pitch_.*": EFFORT_LIMIT_SHOULDER_PITCH},
            velocity_limit_sim={"shoulder_pitch_.*": VELOCITY_LIMIT_SHOULDER_PITCH},
            stiffness={"shoulder_pitch_.*": STIFFNESS_SHOULDER_PITCH},
            damping={"shoulder_pitch_.*": DAMPING_SHOULDER_PITCH},
            armature={"shoulder_pitch_.*": ARMATURE_SHOULDER_PITCH},
            min_delay=MIN_ACTUATOR_DELAY,
            max_delay=MAX_ACTUATOR_DELAY,
            X1={"shoulder_pitch_.*": TORQUE_CURVE_X1_SHOULDER_PITCH},
            X2={"shoulder_pitch_.*": VELOCITY_LIMIT_SHOULDER_PITCH},
            Y1={"shoulder_pitch_.*": TORQUE_CURVE_Y1_SHOULDER_PITCH},
            Y2={"shoulder_pitch_.*": EFFORT_LIMIT_SHOULDER_PITCH},
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
