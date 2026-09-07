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

"""Unitree G1 29 DoF (rev 1.0) humanoid articulation configuration.

Adapted from yrp03_v2.py (same 6 DoF legs: hip_yaw / hip_roll / hip_pitch /
knee / ankle_pitch / ankle_roll). G1 29dof rev 1_0 has 3 waist joints
(waist_yaw / waist_roll / waist_pitch) and 7 arm joints per side
(shoulder_pitch / shoulder_roll / shoulder_yaw / elbow / wrist_roll /
wrist_pitch / wrist_yaw); joint names carry the ``left_`` / ``right_``
prefix and the root body is ``pelvis``. Effort / velocity limits are taken
from g1_custom_collision_29dof.urdf; stiffness / damping / armature are
tunable sim baselines (MJCF only defines normalized -1..1 motors).
"""

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab.actuators import CustomDelayedPDActuatorCfg
from legged_lab.assets import ISAAC_ASSET_DIR

ARMATURE_HIP_YAW = 0.05
ARMATURE_HIP_ROLL = 0.1
ARMATURE_HIP_PITCH = 0.1
ARMATURE_KNEE = 0.1
ARMATURE_ANKLE_PITCH = 0.05
ARMATURE_ANKLE_ROLL = 0.02
ARMATURE_WAIST_YAW = 0.05
ARMATURE_WAIST_ROLL = 0.02
ARMATURE_WAIST_PITCH = 0.02
ARMATURE_SHOULDER = 0.01
ARMATURE_ELBOW = 0.01
ARMATURE_WRIST_ROLL = 0.005
ARMATURE_WRIST_PITCH = 0.002
ARMATURE_WRIST_YAW = 0.002

# Effort / velocity limits from g1_custom_collision_29dof.urdf:
EFFORT_LIMIT_HIP_YAW = 88
EFFORT_LIMIT_HIP_ROLL = 139
EFFORT_LIMIT_HIP_PITCH = 88
EFFORT_LIMIT_KNEE = 139
EFFORT_LIMIT_ANKLE_PITCH = 50
EFFORT_LIMIT_ANKLE_ROLL = 50
EFFORT_LIMIT_WAIST_YAW = 88
EFFORT_LIMIT_WAIST_ROLL = 50
EFFORT_LIMIT_WAIST_PITCH = 50
EFFORT_LIMIT_SHOULDER = 25
EFFORT_LIMIT_ELBOW = 25
EFFORT_LIMIT_WRIST_ROLL = 25
EFFORT_LIMIT_WRIST_PITCH = 5
EFFORT_LIMIT_WRIST_YAW = 5

VELOCITY_LIMIT_HIP_YAW = 32
VELOCITY_LIMIT_HIP_ROLL = 20
VELOCITY_LIMIT_HIP_PITCH = 32
VELOCITY_LIMIT_KNEE = 20
VELOCITY_LIMIT_ANKLE_PITCH = 37
VELOCITY_LIMIT_ANKLE_ROLL = 37
VELOCITY_LIMIT_WAIST_YAW = 32
VELOCITY_LIMIT_WAIST_ROLL = 37
VELOCITY_LIMIT_WAIST_PITCH = 37
VELOCITY_LIMIT_SHOULDER = 37
VELOCITY_LIMIT_ELBOW = 37
VELOCITY_LIMIT_WRIST_ROLL = 37
VELOCITY_LIMIT_WRIST_PITCH = 22
VELOCITY_LIMIT_WRIST_YAW = 22

# Tunable sim baselines (MJCF has no PD gains):
STIFFNESS_HIP_YAW = 150.0
STIFFNESS_HIP_ROLL = 300.0
STIFFNESS_HIP_PITCH = 300.0
STIFFNESS_KNEE = 300.0
STIFFNESS_ANKLE_PITCH = 80.0
STIFFNESS_ANKLE_ROLL = 80.0
STIFFNESS_WAIST_YAW = 150.0
STIFFNESS_WAIST_ROLL = 80.0
STIFFNESS_WAIST_PITCH = 80.0
STIFFNESS_SHOULDER = 40.0
STIFFNESS_ELBOW = 40.0
STIFFNESS_WRIST_ROLL = 10.0
STIFFNESS_WRIST_PITCH = 5.0
STIFFNESS_WRIST_YAW = 5.0

DAMPING_HIP_YAW = 5.0
DAMPING_HIP_ROLL = 8.0
DAMPING_HIP_PITCH = 8.0
DAMPING_KNEE = 8.0
DAMPING_ANKLE_PITCH = 3.0
DAMPING_ANKLE_ROLL = 3.0
DAMPING_WAIST_YAW = 5.0
DAMPING_WAIST_ROLL = 3.0
DAMPING_WAIST_PITCH = 3.0
DAMPING_SHOULDER = 2.0
DAMPING_ELBOW = 2.0
DAMPING_WRIST_ROLL = 1.0
DAMPING_WRIST_PITCH = 0.5
DAMPING_WRIST_YAW = 0.5

MIN_ACTUATOR_DELAY = 0
MAX_ACTUATOR_DELAY = 4

# Torque curve knee points: X1 at half the velocity limit, Y1 at 90% effort.
TORQUE_CURVE_X1_HIP_YAW = 16.0
TORQUE_CURVE_X1_HIP_ROLL = 10.0
TORQUE_CURVE_X1_HIP_PITCH = 16.0
TORQUE_CURVE_X1_KNEE = 10.0
TORQUE_CURVE_X1_ANKLE_PITCH = 18.5
TORQUE_CURVE_X1_ANKLE_ROLL = 18.5
TORQUE_CURVE_X1_WAIST_YAW = 16.0
TORQUE_CURVE_X1_WAIST_ROLL = 18.5
TORQUE_CURVE_X1_WAIST_PITCH = 18.5
TORQUE_CURVE_X1_SHOULDER = 18.5
TORQUE_CURVE_X1_ELBOW = 18.5
TORQUE_CURVE_X1_WRIST_ROLL = 18.5
TORQUE_CURVE_X1_WRIST_PITCH = 11.0
TORQUE_CURVE_X1_WRIST_YAW = 11.0

TORQUE_CURVE_Y1_HIP_YAW = 79
TORQUE_CURVE_Y1_HIP_ROLL = 125
TORQUE_CURVE_Y1_HIP_PITCH = 79
TORQUE_CURVE_Y1_KNEE = 125
TORQUE_CURVE_Y1_ANKLE_PITCH = 45
TORQUE_CURVE_Y1_ANKLE_ROLL = 45
TORQUE_CURVE_Y1_WAIST_YAW = 79
TORQUE_CURVE_Y1_WAIST_ROLL = 45
TORQUE_CURVE_Y1_WAIST_PITCH = 45
TORQUE_CURVE_Y1_SHOULDER = 22
TORQUE_CURVE_Y1_ELBOW = 22
TORQUE_CURVE_Y1_WRIST_ROLL = 22
TORQUE_CURVE_Y1_WRIST_PITCH = 4.5
TORQUE_CURVE_Y1_WRIST_YAW = 4.5

UNITREE_G1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/unitree_g1/usd/g1_29dof_rev_1_0.usd",
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
        pos=(0.0, 0.0, 0.793),  # g1_mocap_29dof.xml pelvis standing height
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": CustomDelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": EFFORT_LIMIT_HIP_YAW,
                ".*_hip_roll_joint": EFFORT_LIMIT_HIP_ROLL,
                ".*_hip_pitch_joint": EFFORT_LIMIT_HIP_PITCH,
                ".*_knee_joint": EFFORT_LIMIT_KNEE,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": VELOCITY_LIMIT_HIP_YAW,
                ".*_hip_roll_joint": VELOCITY_LIMIT_HIP_ROLL,
                ".*_hip_pitch_joint": VELOCITY_LIMIT_HIP_PITCH,
                ".*_knee_joint": VELOCITY_LIMIT_KNEE,
            },
            stiffness={
                ".*_hip_yaw_joint": STIFFNESS_HIP_YAW,
                ".*_hip_roll_joint": STIFFNESS_HIP_ROLL,
                ".*_hip_pitch_joint": STIFFNESS_HIP_PITCH,
                ".*_knee_joint": STIFFNESS_KNEE,
            },
            damping={
                ".*_hip_yaw_joint": DAMPING_HIP_YAW,
                ".*_hip_roll_joint": DAMPING_HIP_ROLL,
                ".*_hip_pitch_joint": DAMPING_HIP_PITCH,
                ".*_knee_joint": DAMPING_KNEE,
            },
            armature={
                ".*_hip_yaw_joint": ARMATURE_HIP_YAW,
                ".*_hip_roll_joint": ARMATURE_HIP_ROLL,
                ".*_hip_pitch_joint": ARMATURE_HIP_PITCH,
                ".*_knee_joint": ARMATURE_KNEE,
            },
            min_delay=MIN_ACTUATOR_DELAY,
            max_delay=MAX_ACTUATOR_DELAY,
            X1={
                ".*_hip_yaw_joint": TORQUE_CURVE_X1_HIP_YAW,
                ".*_hip_roll_joint": TORQUE_CURVE_X1_HIP_ROLL,
                ".*_hip_pitch_joint": TORQUE_CURVE_X1_HIP_PITCH,
                ".*_knee_joint": TORQUE_CURVE_X1_KNEE,
            },
            X2={
                ".*_hip_yaw_joint": VELOCITY_LIMIT_HIP_YAW,
                ".*_hip_roll_joint": VELOCITY_LIMIT_HIP_ROLL,
                ".*_hip_pitch_joint": VELOCITY_LIMIT_HIP_PITCH,
                ".*_knee_joint": VELOCITY_LIMIT_KNEE,
            },
            Y1={
                ".*_hip_yaw_joint": TORQUE_CURVE_Y1_HIP_YAW,
                ".*_hip_roll_joint": TORQUE_CURVE_Y1_HIP_ROLL,
                ".*_hip_pitch_joint": TORQUE_CURVE_Y1_HIP_PITCH,
                ".*_knee_joint": TORQUE_CURVE_Y1_KNEE,
            },
            Y2={
                ".*_hip_yaw_joint": EFFORT_LIMIT_HIP_YAW,
                ".*_hip_roll_joint": EFFORT_LIMIT_HIP_ROLL,
                ".*_hip_pitch_joint": EFFORT_LIMIT_HIP_PITCH,
                ".*_knee_joint": EFFORT_LIMIT_KNEE,
            },
        ),
        "ankle_pitch": CustomDelayedPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint"],
            effort_limit_sim={".*_ankle_pitch_joint": EFFORT_LIMIT_ANKLE_PITCH},
            velocity_limit_sim={".*_ankle_pitch_joint": VELOCITY_LIMIT_ANKLE_PITCH},
            stiffness={".*_ankle_pitch_joint": STIFFNESS_ANKLE_PITCH},
            damping={".*_ankle_pitch_joint": DAMPING_ANKLE_PITCH},
            armature={".*_ankle_pitch_joint": ARMATURE_ANKLE_PITCH},
            min_delay=MIN_ACTUATOR_DELAY,
            max_delay=MAX_ACTUATOR_DELAY,
            X1={".*_ankle_pitch_joint": TORQUE_CURVE_X1_ANKLE_PITCH},
            X2={".*_ankle_pitch_joint": VELOCITY_LIMIT_ANKLE_PITCH},
            Y1={".*_ankle_pitch_joint": TORQUE_CURVE_Y1_ANKLE_PITCH},
            Y2={".*_ankle_pitch_joint": EFFORT_LIMIT_ANKLE_PITCH},
        ),
        "ankle_roll": CustomDelayedPDActuatorCfg(
            joint_names_expr=[".*_ankle_roll_joint"],
            effort_limit_sim={".*_ankle_roll_joint": EFFORT_LIMIT_ANKLE_ROLL},
            velocity_limit_sim={".*_ankle_roll_joint": VELOCITY_LIMIT_ANKLE_ROLL},
            stiffness={".*_ankle_roll_joint": STIFFNESS_ANKLE_ROLL},
            damping={".*_ankle_roll_joint": DAMPING_ANKLE_ROLL},
            armature={".*_ankle_roll_joint": ARMATURE_ANKLE_ROLL},
            min_delay=MIN_ACTUATOR_DELAY,
            max_delay=MAX_ACTUATOR_DELAY,
            X1={".*_ankle_roll_joint": TORQUE_CURVE_X1_ANKLE_ROLL},
            X2={".*_ankle_roll_joint": VELOCITY_LIMIT_ANKLE_ROLL},
            Y1={".*_ankle_roll_joint": TORQUE_CURVE_Y1_ANKLE_ROLL},
            Y2={".*_ankle_roll_joint": EFFORT_LIMIT_ANKLE_ROLL},
        ),
        "waist": CustomDelayedPDActuatorCfg(
            joint_names_expr=[
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            effort_limit_sim={
                "waist_yaw_joint": EFFORT_LIMIT_WAIST_YAW,
                "waist_roll_joint": EFFORT_LIMIT_WAIST_ROLL,
                "waist_pitch_joint": EFFORT_LIMIT_WAIST_PITCH,
            },
            velocity_limit_sim={
                "waist_yaw_joint": VELOCITY_LIMIT_WAIST_YAW,
                "waist_roll_joint": VELOCITY_LIMIT_WAIST_ROLL,
                "waist_pitch_joint": VELOCITY_LIMIT_WAIST_PITCH,
            },
            stiffness={
                "waist_yaw_joint": STIFFNESS_WAIST_YAW,
                "waist_roll_joint": STIFFNESS_WAIST_ROLL,
                "waist_pitch_joint": STIFFNESS_WAIST_PITCH,
            },
            damping={
                "waist_yaw_joint": DAMPING_WAIST_YAW,
                "waist_roll_joint": DAMPING_WAIST_ROLL,
                "waist_pitch_joint": DAMPING_WAIST_PITCH,
            },
            armature={
                "waist_yaw_joint": ARMATURE_WAIST_YAW,
                "waist_roll_joint": ARMATURE_WAIST_ROLL,
                "waist_pitch_joint": ARMATURE_WAIST_PITCH,
            },
            min_delay=MIN_ACTUATOR_DELAY,
            max_delay=MAX_ACTUATOR_DELAY,
            X1={
                "waist_yaw_joint": TORQUE_CURVE_X1_WAIST_YAW,
                "waist_roll_joint": TORQUE_CURVE_X1_WAIST_ROLL,
                "waist_pitch_joint": TORQUE_CURVE_X1_WAIST_PITCH,
            },
            X2={
                "waist_yaw_joint": VELOCITY_LIMIT_WAIST_YAW,
                "waist_roll_joint": VELOCITY_LIMIT_WAIST_ROLL,
                "waist_pitch_joint": VELOCITY_LIMIT_WAIST_PITCH,
            },
            Y1={
                "waist_yaw_joint": TORQUE_CURVE_Y1_WAIST_YAW,
                "waist_roll_joint": TORQUE_CURVE_Y1_WAIST_ROLL,
                "waist_pitch_joint": TORQUE_CURVE_Y1_WAIST_PITCH,
            },
            Y2={
                "waist_yaw_joint": EFFORT_LIMIT_WAIST_YAW,
                "waist_roll_joint": EFFORT_LIMIT_WAIST_ROLL,
                "waist_pitch_joint": EFFORT_LIMIT_WAIST_PITCH,
            },
        ),
        "arms": CustomDelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": EFFORT_LIMIT_SHOULDER,
                ".*_shoulder_roll_joint": EFFORT_LIMIT_SHOULDER,
                ".*_shoulder_yaw_joint": EFFORT_LIMIT_SHOULDER,
                ".*_elbow_joint": EFFORT_LIMIT_ELBOW,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": VELOCITY_LIMIT_SHOULDER,
                ".*_shoulder_roll_joint": VELOCITY_LIMIT_SHOULDER,
                ".*_shoulder_yaw_joint": VELOCITY_LIMIT_SHOULDER,
                ".*_elbow_joint": VELOCITY_LIMIT_ELBOW,
            },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_SHOULDER,
                ".*_shoulder_roll_joint": STIFFNESS_SHOULDER,
                ".*_shoulder_yaw_joint": STIFFNESS_SHOULDER,
                ".*_elbow_joint": STIFFNESS_ELBOW,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_SHOULDER,
                ".*_shoulder_roll_joint": DAMPING_SHOULDER,
                ".*_shoulder_yaw_joint": DAMPING_SHOULDER,
                ".*_elbow_joint": DAMPING_ELBOW,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_SHOULDER,
                ".*_shoulder_roll_joint": ARMATURE_SHOULDER,
                ".*_shoulder_yaw_joint": ARMATURE_SHOULDER,
                ".*_elbow_joint": ARMATURE_ELBOW,
            },
            min_delay=MIN_ACTUATOR_DELAY,
            max_delay=MAX_ACTUATOR_DELAY,
            X1={
                ".*_shoulder_pitch_joint": TORQUE_CURVE_X1_SHOULDER,
                ".*_shoulder_roll_joint": TORQUE_CURVE_X1_SHOULDER,
                ".*_shoulder_yaw_joint": TORQUE_CURVE_X1_SHOULDER,
                ".*_elbow_joint": TORQUE_CURVE_X1_ELBOW,
            },
            X2={
                ".*_shoulder_pitch_joint": VELOCITY_LIMIT_SHOULDER,
                ".*_shoulder_roll_joint": VELOCITY_LIMIT_SHOULDER,
                ".*_shoulder_yaw_joint": VELOCITY_LIMIT_SHOULDER,
                ".*_elbow_joint": VELOCITY_LIMIT_ELBOW,
            },
            Y1={
                ".*_shoulder_pitch_joint": TORQUE_CURVE_Y1_SHOULDER,
                ".*_shoulder_roll_joint": TORQUE_CURVE_Y1_SHOULDER,
                ".*_shoulder_yaw_joint": TORQUE_CURVE_Y1_SHOULDER,
                ".*_elbow_joint": TORQUE_CURVE_Y1_ELBOW,
            },
            Y2={
                ".*_shoulder_pitch_joint": EFFORT_LIMIT_SHOULDER,
                ".*_shoulder_roll_joint": EFFORT_LIMIT_SHOULDER,
                ".*_shoulder_yaw_joint": EFFORT_LIMIT_SHOULDER,
                ".*_elbow_joint": EFFORT_LIMIT_ELBOW,
            },
        ),
        "wrists": CustomDelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_wrist_roll_joint": EFFORT_LIMIT_WRIST_ROLL,
                ".*_wrist_pitch_joint": EFFORT_LIMIT_WRIST_PITCH,
                ".*_wrist_yaw_joint": EFFORT_LIMIT_WRIST_YAW,
            },
            velocity_limit_sim={
                ".*_wrist_roll_joint": VELOCITY_LIMIT_WRIST_ROLL,
                ".*_wrist_pitch_joint": VELOCITY_LIMIT_WRIST_PITCH,
                ".*_wrist_yaw_joint": VELOCITY_LIMIT_WRIST_YAW,
            },
            stiffness={
                ".*_wrist_roll_joint": STIFFNESS_WRIST_ROLL,
                ".*_wrist_pitch_joint": STIFFNESS_WRIST_PITCH,
                ".*_wrist_yaw_joint": STIFFNESS_WRIST_YAW,
            },
            damping={
                ".*_wrist_roll_joint": DAMPING_WRIST_ROLL,
                ".*_wrist_pitch_joint": DAMPING_WRIST_PITCH,
                ".*_wrist_yaw_joint": DAMPING_WRIST_YAW,
            },
            armature={
                ".*_wrist_roll_joint": ARMATURE_WRIST_ROLL,
                ".*_wrist_pitch_joint": ARMATURE_WRIST_PITCH,
                ".*_wrist_yaw_joint": ARMATURE_WRIST_YAW,
            },
            min_delay=MIN_ACTUATOR_DELAY,
            max_delay=MAX_ACTUATOR_DELAY,
            X1={
                ".*_wrist_roll_joint": TORQUE_CURVE_X1_WRIST_ROLL,
                ".*_wrist_pitch_joint": TORQUE_CURVE_X1_WRIST_PITCH,
                ".*_wrist_yaw_joint": TORQUE_CURVE_X1_WRIST_YAW,
            },
            X2={
                ".*_wrist_roll_joint": VELOCITY_LIMIT_WRIST_ROLL,
                ".*_wrist_pitch_joint": VELOCITY_LIMIT_WRIST_PITCH,
                ".*_wrist_yaw_joint": VELOCITY_LIMIT_WRIST_YAW,
            },
            Y1={
                ".*_wrist_roll_joint": TORQUE_CURVE_Y1_WRIST_ROLL,
                ".*_wrist_pitch_joint": TORQUE_CURVE_Y1_WRIST_PITCH,
                ".*_wrist_yaw_joint": TORQUE_CURVE_Y1_WRIST_YAW,
            },
            Y2={
                ".*_wrist_roll_joint": EFFORT_LIMIT_WRIST_ROLL,
                ".*_wrist_pitch_joint": EFFORT_LIMIT_WRIST_PITCH,
                ".*_wrist_yaw_joint": EFFORT_LIMIT_WRIST_YAW,
            },
        ),
    },
)


# AMP expert feature layout for this articulation. Only depends on the joint
# structure (mirroring metadata), so it lives here next to UNITREE_G1_CFG.
# 29 DoF: right_arm(7) + left_arm(7) + right_leg(6) + left_leg(6) + waist(3).
# leg_mirror_flip_indices: leg-local indices whose sign flips under left/right
# mirror — hip_yaw(0), hip_roll(1), ankle_roll(5).
AMP_LOADER_KWARGS = {
    "leg_joint_size": 6,
    "leg_mirror_flip_indices": (0, 1, 5),
}
