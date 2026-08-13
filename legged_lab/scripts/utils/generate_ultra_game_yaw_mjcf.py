#!/usr/bin/env python3
"""Generate MuJoCo MJCF for ultra_GameYaw from ultra0523Yaw.urdf.

Output: legged_lab/assets/ultra_GameYaw/mjcf/ultra_game_yaw.xml

Requires: mujoco (same env as sim2sim).

MuJoCo URDF loader resolves mesh filenames relative to the URDF directory
(basename only). This script symlinks STL files into urdf/ temporarily if needed.
"""

from __future__ import annotations

import os
import textwrap

import mujoco

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
ASSET_DIR = os.path.join(REPO_ROOT, "legged_lab/assets/ultra_GameYaw")
URDF_DIR = os.path.join(ASSET_DIR, "urdf")
MESH_DIR = os.path.join(ASSET_DIR, "meshes_simplified")
MJCF_DIR = os.path.join(ASSET_DIR, "mjcf")
URDF_FILE = os.path.join(URDF_DIR, "ultra0523Yaw.urdf")
OUT_XML = os.path.join(MJCF_DIR, "ultra_game_yaw.xml")

# MJCF actuator / sensor order (URDF chain order, matches sim2sim_ultraYaw.py)
JOINT_NAMES = [
    "hip_yaw_l_joint",
    "hip_roll_l_joint",
    "hip_pitch_l_joint",
    "knee_pitch_l_joint",
    "ankle_pitch_l_joint",
    "hip_yaw_r_joint",
    "hip_roll_r_joint",
    "hip_pitch_r_joint",
    "knee_pitch_r_joint",
    "ankle_pitch_r_joint",
    "waist_yaw_joint",
    "shoulder_pitch_l_joint",
    "shoulder_pitch_r_joint",
]

# PD + armature + effort from ultra_game_yaw.py (Isaac training cfg)
JOINT_ACTUATOR = {
    "hip_yaw_l_joint": dict(kp=500, kv=10, armature=0.048865, effort=250),
    "hip_yaw_r_joint": dict(kp=500, kv=10, armature=0.048865, effort=250),
    "hip_roll_l_joint": dict(kp=700, kv=20, armature=0.40318, effort=657),
    "hip_roll_r_joint": dict(kp=700, kv=20, armature=0.40318, effort=657),
    "hip_pitch_l_joint": dict(kp=700, kv=10, armature=0.40318, effort=657),
    "hip_pitch_r_joint": dict(kp=700, kv=10, armature=0.40318, effort=657),
    "knee_pitch_l_joint": dict(kp=700, kv=10, armature=0.40318, effort=657),
    "knee_pitch_r_joint": dict(kp=700, kv=10, armature=0.40318, effort=657),
    "ankle_pitch_l_joint": dict(kp=100, kv=6, armature=0.21448, effort=400),
    "ankle_pitch_r_joint": dict(kp=100, kv=6, armature=0.21448, effort=400),
    "waist_yaw_joint": dict(kp=400, kv=8, armature=0.048865, effort=250),
    "shoulder_pitch_l_joint": dict(kp=50, kv=3, armature=0.041125, effort=140),
    "shoulder_pitch_r_joint": dict(kp=50, kv=3, armature=0.041125, effort=140),
}

BASE_HEIGHT = 1.23  # ultra_game_yaw.py init_state pos z

# Foot-ground contact (sim2sim): stiff ground only; small foot margin (large margin -> unstable stand).
_GROUND_CONTACT_SOL = 'solref="0.01 1" solimp="0.9 0.95 0.001 0.5 2"'
_GROUND_GEOM_EXTRA = f'condim="3" friction="1 0.5 0.005" {_GROUND_CONTACT_SOL}'
_FOOT_COLLISION_GEOM = 'group="3" friction="1 0.5 0.005" margin="0.001" material="body"'

# Visual-only mesh (group 1). Collision primitives use group 3 (hidden in MuJoCo viewer).
_VIS = (
    '<geom type="mesh" mesh="{mesh}" contype="0" conaffinity="0" '
    'group="1" density="0" material="body"/>'
)


def _mesh_assets_block() -> str:
    lines: list[str] = []
    for fn in sorted(os.listdir(MESH_DIR)):
        if not fn.lower().endswith(".stl"):
            continue
        name = os.path.splitext(fn)[0]
        lines.append(f'                <mesh name="{name}" file="{fn}"/>')
    return "\n".join(lines)


def _ensure_mesh_symlinks() -> list[str]:
    """MuJoCo URDF loader looks up mesh basenames next to the URDF file."""
    created: list[str] = []
    for fn in os.listdir(MESH_DIR):
        if not fn.lower().endswith(".stl"):
            continue
        dst = os.path.join(URDF_DIR, fn)
        if os.path.lexists(dst):
            continue
        os.symlink(os.path.join(MESH_DIR, fn), dst)
        created.append(dst)
    return created


def _remove_paths(paths: list[str]) -> None:
    for p in paths:
        if os.path.islink(p):
            os.unlink(p)


def _load_urdf_model() -> mujoco.MjModel:
    cwd = os.getcwd()
    try:
        os.chdir(URDF_DIR)
        return mujoco.MjModel.from_xml_path(os.path.basename(URDF_FILE))
    finally:
        os.chdir(cwd)


def _actuator_block() -> str:
    lines = []
    for name in JOINT_NAMES:
        cfg = JOINT_ACTUATOR[name]
        lines.append(
            f'        <position name="{name}" joint="{name}" '
            f'kp="{cfg["kp"]:.1f}" kv="{cfg["kv"]:.1f}"/>'
        )
    return "\n".join(lines)


def _sensor_block() -> str:
    lines = []
    for name in JOINT_NAMES:
        lines.append(f'        <jointpos joint="{name}"/>')
    for name in JOINT_NAMES:
        lines.append(f'        <jointvel joint="{name}"/>')
    lines.extend(
        [
            '        <framequat name="orientation" objtype="site" noise="0.001" objname="imu"/>',
            '        <framepos name="position" objtype="site" noise="0.001" objname="imu"/>',
            '        <gyro name="angular-velocity" site="imu" noise="0.005" cutoff="34.9"/>',
            '        <velocimeter name="linear-velocity" site="imu" noise="0.001" cutoff="30"/>',
            '        <accelerometer name="linear-acceleration" site="imu" noise="0.005" cutoff="157"/>',
        ]
    )
    return "\n".join(lines)


def build_mjcf_xml() -> str:
    """Hand-crafted MJCF: kinematics from ultra0523Yaw.urdf, tuned for sim2sim."""
    arm = {n: JOINT_ACTUATOR[n] for n in JOINT_NAMES}
    return textwrap.dedent(
        f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <!--
          ultra_GameYaw MuJoCo model for sim2sim (ultra0523Yaw.urdf).
          Generated by legged_lab/scripts/generate_ultra_game_yaw_mjcf.py

          - Kinematics / inertial / collision primitives: from URDF
          - Mesh visual (group 1): all links via meshes_simplified/
          - Collision (group 3, viewer-hidden): URDF cylinders + ankle mesh feet
          - PD kp/kv/armature: ultra_game_yaw.py Isaac training cfg
          - Joint order in actuators/sensors: URDF chain order (see sim2sim_ultraYaw.py)
        -->
        <mujoco model="ultra_game_yaw">
            <compiler angle="radian" meshdir="../meshes_simplified/" eulerseq="zyx" autolimits="true"/>
            <option gravity="0 0 -9.81" integrator="implicitfast"/>
            <size njmax="500" nconmax="200"/>

            <visual>
                <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
                <rgba force="1 0 0 1"/>
                <global azimuth="135" elevation="-25"/>
            </visual>

            <asset>
                <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
                <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2"
                         width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>
                <material name="matplane" reflectance="0." texture="texplane" texrepeat="1 1" texuniform="true"/>
                <material name="body" rgba="0.752941 0.752941 0.752941 1"/>

        {_mesh_assets_block()}
            </asset>

            <worldbody>
                <light directional="true" diffuse=".4 .4 .4" specular="0.1 0.1 0.1" pos="0 0 5" dir="0 0 -1" castshadow="false"/>
                <geom name="ground" type="plane" size="0 0 1" pos="0 0 0" material="matplane" {_GROUND_GEOM_EXTRA}/>

                <body name="base_link" pos="0 0 {BASE_HEIGHT}">
                    <freejoint name="root"/>
                    <site name="imu" size="0.01" pos="0 0 0"/>
                    <inertial pos="-0.00105 0  -0.13823" mass="6.30030"
                              fullinertia="0.05024 0.03127 0.03303 0  -0.00023 0"/>
                    {_VIS.format(mesh="base_link")}
                    <geom size="0.05 0.01" pos="0 0 -0.08" type="cylinder" group="3" material="body"/>

                    <!-- Left leg -->
                    <body name="hip_yaw_l_link" pos="0 0.07 -0.2135">
                        <inertial pos="-0.10593 0.01716 -0.0943" quat="0.931073 0.062054 0.301882 -0.195243"
                                  mass="4.17605" diaginertia="0.0172795 0.0167561 0.00993442"/>
                        <joint name="hip_yaw_l_joint" axis="0 0 1" range="-0.436332 0.436332"
                               actuatorfrcrange="-{arm['hip_yaw_l_joint']['effort']} {arm['hip_yaw_l_joint']['effort']}"
                               armature="{arm['hip_yaw_l_joint']['armature']}"/>
                        {_VIS.format(mesh="hip_yaw_l_link")}
                        <body name="hip_roll_l_link" pos="-0.0815 0.024 -0.115">
                            <inertial pos="0.07397 -0.01706 0.00017" quat="0.580819 0.575652 0.401057 0.412829"
                                      mass="4.09037" diaginertia="0.0131236 0.0118403 0.0101961"/>
                            <joint name="hip_roll_l_joint" axis="1 0 0" range="-0.349066 0.523599"
                                   actuatorfrcrange="-{arm['hip_roll_l_joint']['effort']} {arm['hip_roll_l_joint']['effort']}"
                                   armature="{arm['hip_roll_l_joint']['armature']}"/>
                            {_VIS.format(mesh="hip_roll_l_link")}
                            <body name="hip_pitch_l_link" pos="0.0815 0.024 0">
                                <inertial pos="0.00427 0.08854 -0.07756" quat="0.972821 -0.0351176 -0.0279091 0.227174"
                                          mass="8.24544" diaginertia="0.170371 0.168966 0.0227434"/>
                                <joint name="hip_pitch_l_joint" axis="0 1 0" range="-2.79253 2.0944"
                                       actuatorfrcrange="-{arm['hip_pitch_l_joint']['effort']} {arm['hip_pitch_l_joint']['effort']}"
                                       armature="{arm['hip_pitch_l_joint']['armature']}"/>
                                {_VIS.format(mesh="hip_pitch_l_link")}
                                <geom size="0.06 0.1" pos="0 0.029 -0.225" type="cylinder" group="3" material="body"/>
                                <body name="knee_pitch_l_link" pos="0 0.0705 -0.45">
                                    <inertial pos="0.010531 0.010036 -0.160398" quat="0.466522 -0.00304073 0.0430458 0.883456"
                                              mass="5.23089" diaginertia="0.0802343 0.0773968 0.0115898"/>
                                    <joint name="knee_pitch_l_joint" axis="0 1 0" range="-0.0872665 2.44346"
                                           actuatorfrcrange="-{arm['knee_pitch_l_joint']['effort']} {arm['knee_pitch_l_joint']['effort']}"
                                           armature="{arm['knee_pitch_l_joint']['armature']}"/>
                                    {_VIS.format(mesh="knee_pitch_l_link")}
                                    <geom size="0.05 0.1" pos="0 0 -0.225" type="cylinder" group="3" material="body"/>
                                    <body name="ankle_pitch_l_link" pos="0 0 -0.45">
                                        <inertial pos="0.007487 -0.003298 -0.043773" quat="0.317134 0.632029 0.370816 0.602047"
                                                  mass="0.934211" diaginertia="0.0046141 0.0044328 0.000604104"/>
                                        <joint name="ankle_pitch_l_joint" axis="0 1 0" range="-1.22173 1.0472"
                                               actuatorfrcrange="-{arm['ankle_pitch_l_joint']['effort']} {arm['ankle_pitch_l_joint']['effort']}"
                                               armature="{arm['ankle_pitch_l_joint']['armature']}"/>
                                        {_VIS.format(mesh="ankle_pitch_l_link")}
                                        <geom type="mesh" mesh="ankle_pitch_l_link" {_FOOT_COLLISION_GEOM}/>
                                    </body>
                                </body>
                            </body>
                        </body>
                    </body>

                    <!-- Right leg -->
                    <body name="hip_yaw_r_link" pos="0 -0.07 -0.2135">
                        <inertial pos="-0.10593 -0.01716 -0.0943" quat="0.931073 -0.062054 0.301882 0.195243"
                                  mass="4.17605" diaginertia="0.0172795 0.0167561 0.00993442"/>
                        <joint name="hip_yaw_r_joint" axis="0 0 1" range="-0.436332 0.436332"
                               actuatorfrcrange="-{arm['hip_yaw_r_joint']['effort']} {arm['hip_yaw_r_joint']['effort']}"
                               armature="{arm['hip_yaw_r_joint']['armature']}"/>
                        {_VIS.format(mesh="hip_yaw_r_link")}
                        <body name="hip_roll_r_link" pos="-0.0815 -0.024 -0.115">
                            <inertial pos="0.07397 0.01706 0.00017" quat="0.412829 0.401057 0.575652 0.580819"
                                      mass="4.09037" diaginertia="0.0131236 0.0118403 0.0101961"/>
                            <joint name="hip_roll_r_joint" axis="1 0 0" range="-0.523599 0.349066"
                                   actuatorfrcrange="-{arm['hip_roll_r_joint']['effort']} {arm['hip_roll_r_joint']['effort']}"
                                   armature="{arm['hip_roll_r_joint']['armature']}"/>
                            {_VIS.format(mesh="hip_roll_r_link")}
                            <body name="hip_pitch_r_link" pos="0.0815 -0.024 0">
                                <inertial pos="0.00427 -0.08854 -0.07756" quat="0.972821 0.0351176 -0.0279091 -0.227174"
                                          mass="8.24544" diaginertia="0.170371 0.168966 0.0227434"/>
                                <joint name="hip_pitch_r_joint" axis="0 1 0" range="-2.79253 2.0944"
                                       actuatorfrcrange="-{arm['hip_pitch_r_joint']['effort']} {arm['hip_pitch_r_joint']['effort']}"
                                       armature="{arm['hip_pitch_r_joint']['armature']}"/>
                                {_VIS.format(mesh="hip_pitch_r_link")}
                                <geom size="0.06 0.1" pos="0 -0.029 -0.225" type="cylinder" group="3" material="body"/>
                                <body name="knee_pitch_r_link" pos="0 -0.0705 -0.45">
                                    <inertial pos="0.010531 -0.010036 -0.160398" quat="0.883456 0.0430458 -0.00304073 0.466522"
                                              mass="5.23089" diaginertia="0.0802343 0.0773968 0.0115898"/>
                                    <joint name="knee_pitch_r_joint" axis="0 1 0" range="-0.0872665 2.44346"
                                           actuatorfrcrange="-{arm['knee_pitch_r_joint']['effort']} {arm['knee_pitch_r_joint']['effort']}"
                                           armature="{arm['knee_pitch_r_joint']['armature']}"/>
                                    {_VIS.format(mesh="knee_pitch_r_link")}
                                    <geom size="0.05 0.1" pos="0 0 -0.225" type="cylinder" group="3" material="body"/>
                                    <body name="ankle_pitch_r_link" pos="0 0 -0.45">
                                        <inertial pos="0.007487 0.003298 -0.043773" quat="0.602047 0.370816 0.632029 0.317134"
                                                  mass="0.934211" diaginertia="0.0046141 0.0044328 0.000604104"/>
                                        <joint name="ankle_pitch_r_joint" axis="0 1 0" range="-1.22173 1.0472"
                                               actuatorfrcrange="-{arm['ankle_pitch_r_joint']['effort']} {arm['ankle_pitch_r_joint']['effort']}"
                                               armature="{arm['ankle_pitch_r_joint']['armature']}"/>
                                        {_VIS.format(mesh="ankle_pitch_r_link")}
                                        <geom type="mesh" mesh="ankle_pitch_r_link" {_FOOT_COLLISION_GEOM}/>
                                    </body>
                                </body>
                            </body>
                        </body>
                    </body>

                    <!-- Waist + arms -->
                    <body name="waist_yaw_link">
                        <inertial pos="0.00156 -0.001173 0.242753" quat="0.999785 -0.000564997 -0.0202735 0.00438609"
                                  mass="15.6961" diaginertia="0.390472 0.291932 0.181409"/>
                        <joint name="waist_yaw_joint" axis="0 0 1" range="-1.5708 1.5708"
                               actuatorfrcrange="-{arm['waist_yaw_joint']['effort']} {arm['waist_yaw_joint']['effort']}"
                               armature="{arm['waist_yaw_joint']['armature']}"/>
                        {_VIS.format(mesh="waist_yaw_link")}
                        <geom size="0.1 0.025" pos="0 0 0.16" type="cylinder" group="3" material="body"/>
                        <body name="shoulder_pitch_l_link" pos="0.00052293 0.1866 0.26789" quat="0.995247 0.0870717 0.00380173 -0.0434533">
                            <inertial pos="0.070697 0.049271 -0.277004" mass="1.895842"
                                      fullinertia="0.066849 0.072121 0.007156 -0.001084 0.018180 0.004462"/>
                            <joint name="shoulder_pitch_l_joint" axis="0 1 0" range="-2.96706 2.96706"
                                   actuatorfrcrange="-{arm['shoulder_pitch_l_joint']['effort']} {arm['shoulder_pitch_l_joint']['effort']}"
                                   armature="{arm['shoulder_pitch_l_joint']['armature']}"/>
                            {_VIS.format(mesh="shoulder_pitch_l_link")}
                        </body>
                        <body name="shoulder_pitch_r_link" pos="0.00052293 -0.1866 0.26789" quat="0.995247 -0.0870717 0.00380173 0.0434533">
                            <inertial pos="0.070697 -0.049271 -0.277004" mass="1.895842"
                                      fullinertia="0.066849 0.072121 0.007156 0.001084 0.018180 -0.004462"/>
                            <joint name="shoulder_pitch_r_joint" axis="0 1 0" range="-2.96706 2.96706"
                                   actuatorfrcrange="-{arm['shoulder_pitch_r_joint']['effort']} {arm['shoulder_pitch_r_joint']['effort']}"
                                   armature="{arm['shoulder_pitch_r_joint']['armature']}"/>
                            {_VIS.format(mesh="shoulder_pitch_r_link")}
                        </body>
                    </body>
                </body>
            </worldbody>

            <actuator>
        {_actuator_block()}
            </actuator>

            <sensor>
        {_sensor_block()}
            </sensor>
        </mujoco>
        """
    )


def main() -> None:
    os.makedirs(MJCF_DIR, exist_ok=True)
    symlinks = _ensure_mesh_symlinks()
    try:
        xml = build_mjcf_xml()
        with open(OUT_XML, "w", encoding="utf-8") as f:
            f.write(xml)
        model = mujoco.MjModel.from_xml_path(OUT_XML)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        print(f"[OK] Wrote {OUT_XML}")
        print(f"     nu={model.nu} nv={model.nv} nbody={model.nbody}")
        actuators = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
        print(f"     actuators: {actuators}")
    finally:
        _remove_paths(symlinks)


if __name__ == "__main__":
    main()
