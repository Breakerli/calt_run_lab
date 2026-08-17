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

"""Isaac Lab articulation configuration for the 13-DoF 01RYP-v2 robot."""

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab.assets import ISAAC_ASSET_DIR

__all__ = [
    "ROBOT_01RYP_V2_CFG",
    "ROBOT_01RYP_V2_DEFAULT_BASE_HEIGHT",
    "ROBOT_01RYP_V2_DEFAULT_JOINT_POS",
    "ROBOT_01RYP_V2_JOINT_NAMES",
]

# This is the policy/control interface order.  The USD articulation may expose
# joints in topology order, so callers must resolve these names explicitly.
ROBOT_01RYP_V2_JOINT_NAMES = (
    "hip_roll_l_joint",
    "hip_yaw_l_joint",
    "hip_pitch_l_joint",
    "knee_pitch_l_joint",
    "ankle_pitch_l_joint",
    "hip_roll_r_joint",
    "hip_yaw_r_joint",
    "hip_pitch_r_joint",
    "knee_pitch_r_joint",
    "ankle_pitch_r_joint",
    "shoulder_pitch_l_joint",
    "shoulder_pitch_r_joint",
    "waist_yaw_joint",
)

# URDF nominal pose FK: pelvis-to-ankle origin is about 1.019 m and the
# collision mesh extends another 0.070 m below the ankle origin.
ROBOT_01RYP_V2_DEFAULT_BASE_HEIGHT = 1.09

ROBOT_01RYP_V2_DEFAULT_JOINT_POS = {
    "hip_roll_l_joint": 0.0,
    "hip_yaw_l_joint": 0.0,
    "hip_pitch_l_joint": math.radians(-15),
    "knee_pitch_l_joint": math.radians(30),
    "ankle_pitch_l_joint": math.radians(-15),
    "hip_roll_r_joint": 0.0,
    "hip_yaw_r_joint": 0.0,
    "hip_pitch_r_joint": math.radians(-15),
    "knee_pitch_r_joint": math.radians(30),
    "ankle_pitch_r_joint": math.radians(-15),
    "shoulder_pitch_l_joint": 0.0,
    "shoulder_pitch_r_joint": 0.0,
    "waist_yaw_joint": 0.0,
}

ROBOT_01RYP_V2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # Export this file from urdf/01RYP-v2.urdf before using the Isaac asset.
        usd_path=f"{ISAAC_ASSET_DIR}/ryp01-v2/usd/01RYP-v2.usd",
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
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, ROBOT_01RYP_V2_DEFAULT_BASE_HEIGHT),
        joint_pos=ROBOT_01RYP_V2_DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "hip_roll_.*_joint",
                "hip_yaw_.*_joint",
                "hip_pitch_.*_joint",
                "knee_pitch_.*_joint",
            ],
            effort_limit_sim={
                "hip_roll_.*_joint": 657,
                "hip_pitch_.*_joint": 657,
                "hip_yaw_.*_joint": 190,
                "knee_pitch_.*_joint": 657,
            },
            velocity_limit_sim={
                "hip_roll_.*_joint": 25.6,
                "hip_pitch_.*_joint": 25.6,
                "hip_yaw_.*_joint": 13.0,
                "knee_pitch_.*_joint": 25.6,
            },
            stiffness={
                "hip_roll_.*_joint": 700,
                "hip_pitch_.*_joint": 700,
                "hip_yaw_.*_joint": 500,
                "knee_pitch_.*_joint": 700,
            },
            damping={
                "hip_roll_.*_joint": 10,
                "hip_pitch_.*_joint": 10,
                "hip_yaw_.*_joint": 5,
                "knee_pitch_.*_joint": 10,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                "ankle_pitch_.*_joint",
            ],
            effort_limit_sim={
                "ankle_pitch_.*_joint": 180,
            },
            velocity_limit_sim={
                "ankle_pitch_.*_joint": 26.0,
            },
            stiffness={
                "ankle_pitch_.*_joint": 30,
            },
            damping={
                "ankle_pitch_.*_joint": 2.5,
            },
        ),
        "shoulders": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pitch_.*_joint",
            ],
            effort_limit_sim={
                "shoulder_pitch_.*_joint": 52.5,
            },
            velocity_limit_sim={
                "shoulder_pitch_.*_joint": 14.1,
            },
            stiffness={
                "shoulder_pitch_.*_joint": 60,
            },
            damping={
                "shoulder_pitch_.*_joint": 3,
            },
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim={"waist_yaw_joint": 250},
            velocity_limit_sim={"waist_yaw_joint": 20.9},
            stiffness={"waist_yaw_joint": 400},
            damping={"waist_yaw_joint": 8},
        ),
    },
)
