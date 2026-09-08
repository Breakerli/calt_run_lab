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

import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils  # type: ignore
import numpy as np
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg
from isaaclab.managers import EventManager, RewardManager
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.sensors.camera import TiledCamera
from isaaclab.sim import PhysxCfg, SimulationContext
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import quat_apply, quat_conjugate
from scipy.spatial.transform import Rotation

from legged_lab.envs.unitree_g1.motion_run_cfg import UnitreeG1MotionRunEnvCfg
import legged_lab.mdp.observations as mdp_obs
from legged_lab.mdp.commands import CustomVelocityCommand
from legged_lab.utils.env_utils.scene import SceneCfg
from rsl_rl.env import VecEnv
from rsl_rl.utils import AMPLoaderDisplay, AMPLoader


class UnitreeG1MotionRunEnv(VecEnv):
    def __init__(
        self,
        cfg: UnitreeG1MotionRunEnvCfg,
        headless,
    ):
        self.cfg: UnitreeG1MotionRunEnvCfg

        self.cfg = cfg
        self.headless = headless
        self.device = self.cfg.device
        self.physics_dt = self.cfg.sim.dt
        self.step_dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.num_envs = self.cfg.scene.num_envs
        self.seed(cfg.scene.seed)

        sim_cfg = sim_utils.SimulationCfg(
            device=cfg.device,
            dt=cfg.sim.dt,
            render_interval=cfg.sim.decimation,
            physx=PhysxCfg(gpu_max_rigid_patch_count=cfg.sim.physx.gpu_max_rigid_patch_count),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )
        self.sim = SimulationContext(sim_cfg)

        scene_cfg = SceneCfg(config=cfg.scene, physics_dt=self.physics_dt, step_dt=self.step_dt)
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()

        self.robot: Articulation = self.scene["robot"]
        self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]

        hs_cfg = self.cfg.scene.height_scanner
        if hs_cfg.enable_height_scan:
            self.height_scanner: RayCaster = self.scene.sensors["height_scanner"]
        self.feet_height_scanners: list[RayCaster] = []
        if hs_cfg.enable_feet_height_scan:
            for foot_name in hs_cfg.feet_body_names:
                sensor_name = f"feet_height_scanner_{foot_name}"
                self.feet_height_scanners.append(self.scene.sensors[sensor_name])

        # Instantiate LiDAR and Depth Camera Sensors if enabled
        if self.cfg.scene.lidar.enable_lidar:
            self.lidar: RayCaster = self.scene.sensors["lidar"]
        if self.cfg.scene.depth_camera.enable_depth_camera:
            self.depth_camera: TiledCamera = self.scene.sensors["depth_camera"]

        command_cfg = UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=self.cfg.commands.resampling_time_range,
            rel_standing_envs=self.cfg.commands.rel_standing_envs,
            rel_heading_envs=self.cfg.commands.rel_heading_envs,
            heading_command=self.cfg.commands.heading_command,
            heading_control_stiffness=self.cfg.commands.heading_control_stiffness,
            debug_vis=self.cfg.commands.debug_vis,
            ranges=self.cfg.commands.ranges,
        )
        # self.command_generator = UniformVelocityCommand(cfg=command_cfg, env=self)
        self.command_generator = CustomVelocityCommand(cfg=command_cfg, env=self)
        self.reward_manager = RewardManager(self.cfg.reward, self)

        self.init_buffers()

        env_ids = torch.arange(self.num_envs, device=self.device)
        self.event_manager = EventManager(self.cfg.domain_rand.events, self)
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        self._cache_physics_material_coeffs()
        self.reset(env_ids)
        self._validate_height_scanners()

        self.amp_loader_display = AMPLoaderDisplay(
            motion_files=self.cfg.amp_motion_files_display, device=self.device, time_between_frames=self.physics_dt
        )
        self.amp_loader_dict = {}
        for style_id in range(self.num_styles):
            style_key = f"style_{style_id}"
            self.amp_loader_dict[style_id] = AMPLoader(
                motion_files=self.cfg.amp_motion_files_dict[style_key],
                device=self.device,
                time_between_frames=self.physics_dt,
                mirror_motion=getattr(self.cfg, "amp_mirror_expert_motion", False),
                **self.cfg.amp_loader_kwargs,
            )
        self.motion_len = self.amp_loader_display.trajectory_num_frames[0]

    def init_buffers(self):
        self.extras = {}

        self.max_episode_length_s = self.cfg.scene.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.step_dt)
        self.num_actions = self.robot.data.default_joint_pos.shape[1]
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations

        # expose env handle for actuator-side access
        actuators = getattr(self.robot, "actuators")
        for actuator in actuators.values():
            actuator._env = self
        self.motor_strength = torch.ones(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.robot.motor_strength = self.motor_strength
        self.joint_static_friction_scale = torch.ones(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.joint_dynamic_friction_scale = torch.ones(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.joint_pos_bias = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )

        self.action_scale = self.cfg.robot.action_scale
        action_buffer_max_delay = max(self.cfg.domain_rand.action_delay.params["max_delay"], 2)
        self.action_buffer = DelayBuffer(
            action_buffer_max_delay, self.num_envs, device=self.device
        )
        self.action_buffer.compute(
            torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        )
        if self.cfg.domain_rand.action_delay.enable:
            time_lags = torch.randint(
                low=self.cfg.domain_rand.action_delay.params["min_delay"],
                high=self.cfg.domain_rand.action_delay.params["max_delay"] + 1,
                size=(self.num_envs,),
                dtype=torch.int,
                device=self.device,
            )
            self.action_buffer.set_time_lag(time_lags, torch.arange(self.num_envs, device=self.device))

        self.robot_cfg = SceneEntityCfg(name="robot")
        self.robot_cfg.resolve(self.scene)
        self.termination_contact_cfg = SceneEntityCfg(
            name="contact_sensor", body_names=self.cfg.robot.terminate_contacts_body_names
        )
        self.termination_contact_cfg.resolve(self.scene)
        self.feet_cfg = SceneEntityCfg(name="contact_sensor", body_names=self.cfg.robot.feet_body_names)
        self.feet_cfg.resolve(self.scene)

        self.feet_body_ids, _ = self.robot.find_bodies(
            name_keys=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
        )
        self.shoulder_body_ids, _ = self.robot.find_bodies(
            name_keys=["left_shoulder_pitch_link", "right_shoulder_pitch_link"], preserve_order=True
        )
        self.left_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                "left_hip_yaw_joint",
                "left_hip_roll_joint",
                "left_hip_pitch_joint",
                "left_knee_joint",
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
            ],
            preserve_order=True,
        )
        self.right_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                "right_hip_yaw_joint",
                "right_hip_roll_joint",
                "right_hip_pitch_joint",
                "right_knee_joint",
                "right_ankle_pitch_joint",
                "right_ankle_roll_joint",
            ],
            preserve_order=True,
        )
        # AMP 特征关节: arm 只取 shoulder_pitch（AMP obs 42 维布局要求每侧 1 个手臂关节）。
        self.left_arm_ids, _ = self.robot.find_joints(
            name_keys=[
                "left_shoulder_pitch_joint",
            ],
            preserve_order=True,
        )
        self.right_arm_ids, _ = self.robot.find_joints(
            name_keys=[
                "right_shoulder_pitch_joint",
            ],
            preserve_order=True,
        )
        self.ankle_joint_ids, _ = self.robot.find_joints(
            name_keys=["left_ankle_pitch_joint", "right_ankle_pitch_joint"],
            preserve_order=True,
        )

        self.waist_yaw_ids, _ = self.robot.find_joints(
            name_keys=["waist_yaw_joint"], preserve_order=True,
        )

        # 显式按名字索引常用关节, 让 reward 函数不依赖 left_leg_ids 的内部位置.
        # 长度均为 2, 顺序固定为 [left, right] (由 name_keys 保证).
        self.hip_roll_ids, _ = self.robot.find_joints(
            name_keys=["left_hip_roll_joint", "right_hip_roll_joint"], preserve_order=True
        )
        self.hip_yaw_ids, _ = self.robot.find_joints(
            name_keys=["left_hip_yaw_joint", "right_hip_yaw_joint"], preserve_order=True
        )
        # G1 ankle_roll joints (order [left, right]); used by symmetry
        # (roll-axis negation) and ankle_roll_action reward.
        self.ankle_roll_ids, _ = self.robot.find_joints(
            name_keys=["left_ankle_roll_joint", "right_ankle_roll_joint"], preserve_order=True
        )
        # G1 额外的手臂/腰 roll/yaw 关节 (镜像变号用; 全 pitch 关节如 elbow/waist_pitch 不变号).
        # 减臂 USD (19DoF 肩 pitch+roll / 17DoF 仅肩 pitch) 下 shoulder_yaw/wrist_* 等已锁为
        # fixed, find_joints 查不到会 raise, 故按 robot 实际关节过滤, 缺失时置空表 (symmetry.py 容忍空表).
        _joint_names = set(self.robot.data.joint_names)

        def _find_optional(name_keys):
            avail = [k for k in name_keys if k in _joint_names]
            if not avail:
                return []
            ids, _ = self.robot.find_joints(name_keys=avail, preserve_order=True)
            return ids

        self.shoulder_roll_ids = _find_optional(["left_shoulder_roll_joint", "right_shoulder_roll_joint"])
        self.shoulder_yaw_ids = _find_optional(["left_shoulder_yaw_joint", "right_shoulder_yaw_joint"])
        self.wrist_roll_ids = _find_optional(["left_wrist_roll_joint", "right_wrist_roll_joint"])
        self.wrist_yaw_ids = _find_optional(["left_wrist_yaw_joint", "right_wrist_yaw_joint"])
        self.waist_roll_ids, _ = self.robot.find_joints(
            name_keys=["waist_roll_joint"], preserve_order=True
        )
        # 整机镜像左右交换列表 (leg 6 + arm 全部左右关节; waist 中线不交换).
        # symmetry.py 优先用 left/right_mirror_ids；AMP 布局的 left_arm_ids 仍只有 shoulder_pitch。
        _LEFT_ARM_ALL = [
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
        ]
        _RIGHT_ARM_ALL = [n.replace("left_", "right_") for n in _LEFT_ARM_ALL]
        _LEFT_LEG_NAMES = [
            "left_hip_yaw_joint",
            "left_hip_roll_joint",
            "left_hip_pitch_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
        ]
        _RIGHT_LEG_NAMES = [n.replace("left_", "right_") for n in _LEFT_LEG_NAMES]
        self.left_mirror_ids, _ = self.robot.find_joints(
            name_keys=_LEFT_LEG_NAMES + [n for n in _LEFT_ARM_ALL if n in _joint_names], preserve_order=True
        )
        self.right_mirror_ids, _ = self.robot.find_joints(
            name_keys=_RIGHT_LEG_NAMES + [n for n in _RIGHT_ARM_ALL if n in _joint_names], preserve_order=True
        )

        self.obs_scales = self.cfg.normalization.obs_scales
        cmd_scale = self.obs_scales.commands
        if isinstance(cmd_scale, (int, float)):
            cmd_scale = (cmd_scale, cmd_scale, cmd_scale)
        self.command_obs_scale = torch.tensor(cmd_scale, device=self.device, dtype=torch.float32)
        self.add_noise = self.cfg.noise.add_noise

        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.last_episode_length = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_episode_track_lin_vel_x_exp = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        # 距上次 episode 结束的 common step 数；0 表示刚结束一局
        self.last_episode_age = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # ── 步态指标（run style 步频/步幅/腾空占比）逐步累计 + last-episode 缓存 ──
        # 落脚判定的接触力阈值（N）：远低于站立支撑力、远高于传感器噪声。
        self._gait_contact_threshold = 10.0
        self._gait_foot_contact_prev = torch.zeros(
            self.num_envs, 2, dtype=torch.bool, device=self.device, requires_grad=False
        )
        self._run_step_count = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self._run_forward_dist = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self._run_flight_steps = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self._run_style_steps = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_episode_cadence = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_episode_stride_length = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_episode_flight_ratio = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        # 最近一次落脚的落脚点相对质心的前向位移（yaw 系，前向为正），供 foot_strike_com_offset 奖励读取
        self._foot_strike_offset = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        # 本局落脚点前向位移累计（用于监测点 Gait/foot_strike_com_offset 的局内均值）
        self._foot_strike_offset_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_episode_foot_strike_offset = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.common_step_counter = 0
        self._curriculum_log_cache: dict[str, float] = {}
        if self.cfg.commands.use_curriculum:
            self._curriculum_log_cache = {
                "Curriculum/lin_vel_x_max": float(self.cfg.commands.ranges.lin_vel_x[1]),
            }
        self.sim_step_counter = 0
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # G1 零位手臂朝前下方 (URDF 链: shoulder->wrist 前向 ~0.20 m, 下降 ~0.11 m);
        # 手部位置特征的近似杆长, tunable。
        self.left_arm_local_vec = torch.tensor([0.24, 0.0, -0.10], device=self.device).repeat((self.num_envs, 1))
        self.right_arm_local_vec = torch.tensor([0.24, 0.0, -0.10], device=self.device).repeat((self.num_envs, 1))

        self.action = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.avg_feet_force_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.avg_feet_speed_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.track_avg_vel_window_s = float(getattr(self.cfg, "track_avg_vel_window_s", 0.5))
        self.avg_lin_vel_x_yaw = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.avg_lin_vel_y_yaw = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.avg_ang_vel_z = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        self.num_styles = self.cfg.num_styles
        self.style_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        priv = self.cfg.critic_privilege
        self.critic_base_height_target = priv.base_height_target
        self.critic_foot_height_offset = priv.foot_height_offset
        self.critic_height_scale = priv.height_scale
        self.critic_foot_force_scale = priv.foot_force_scale
        self.refresh_material_on_reset = (
            priv.refresh_material_on_reset and self._domain_rand_material_on_reset()
        )
        # base_h(1)+vel_yaw(3)+feet_vz(2)+fz(2)+contact(2)+h_err(1)+clearance(2)+domain(2+3*n_dof)
        self.critic_extra_dim = 15 + 3 * self.num_actions

        hs_cfg = self.cfg.scene.height_scanner
        self.num_height_scan = 0
        self.num_feet_scan_points = 0
        self.height_scan_center_idx = 0
        self.height_scan_mirror_indices = None
        self._cached_height_scan = None
        if hs_cfg.enable_height_scan:
            self.num_height_scan = mdp_obs.height_scan_num_points(hs_cfg.size, hs_cfg.resolution)
            self.height_scan_center_idx = mdp_obs.height_scan_center_index(hs_cfg.size, hs_cfg.resolution)
            self.height_scan_mirror_indices = mdp_obs.height_scan_y_mirror_permutation(
                hs_cfg.size, hs_cfg.resolution, self.device
            )
        if hs_cfg.enable_feet_height_scan:
            self.num_feet_scan_points = mdp_obs.height_scan_num_points(
                hs_cfg.feet_scan_size, hs_cfg.feet_scan_resolution
            )

        self.init_obs_buffer()

    def visualize_motion(self, time: float | None = None, frame_idx: int | None = None):
        """
        Update the robot simulation state based on the AMP motion capture data.

        Args:
            time: Time in seconds (uses interpolation). Ignored if frame_idx is set.
            frame_idx: Direct frame index (recommended for export).

        Returns:
            AMP expert feature tensor.
        """
        if frame_idx is not None:
            visual_motion_frame = self.amp_loader_display.trajectories_full[0][frame_idx].clone()
        else:
            if time is None:
                time = 0.0
            visual_motion_frame = self.amp_loader_display.get_full_frame_at_time(0, time)
        device = self.device

        dof_pos = torch.zeros((self.num_envs, self.robot.num_joints), device=device)
        dof_vel = torch.zeros((self.num_envs, self.robot.num_joints), device=device)

        root_pos = visual_motion_frame[:3].clone()
        
        # root_pos[0] += 76.0
        # root_pos[1] += 64.0
        
        root_pos[2] += 1.5
        
        euler = visual_motion_frame[3:6].cpu().numpy()
        quat_xyzw = Rotation.from_euler("XYZ", euler, degrees=False).as_quat()  # [x, y, z, w]
        quat_wxyz = torch.tensor(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=torch.float32, device=device
        )

        # Motion files come in two schemas (legacy artifact from different retargeting runs):
        #   - 44-dim: 2-dof arm (shoulder_pitch + elbow_pitch)        e.g. stand.txt, run.txt
        #   - 52-dim: 4-dof arm (shoulder pitch/roll/yaw + elbow)     e.g. walk007.txt
        # ultra88p_0508 only has 2-dof arms, so for the 52-dim case we pick the
        # shoulder_pitch and elbow_pitch indices and discard the unused arm dofs.
        frame_dim = visual_motion_frame.shape[0]
        
        # dof_pos[:, self.left_leg_ids] = visual_motion_frame[[7, 6, 8, 9, 10]]
        # dof_pos[:, self.right_leg_ids] = visual_motion_frame[[13, 12, 14, 15, 16]]
        
        # # 44-dim data
        # dof_pos[:, self.left_arm_ids] = visual_motion_frame[18]
        # dof_pos[:, self.right_arm_ids] = visual_motion_frame[20]
        
        # lin_vel = visual_motion_frame[22:25].clone()
        
        # dof_vel[:, self.left_leg_ids]  = visual_motion_frame[[29, 28, 30, 31, 32]]
        # dof_vel[:, self.right_leg_ids] = visual_motion_frame[[35, 34, 36, 37, 38]]
        # dof_vel[:, self.left_arm_ids]  = visual_motion_frame[40]
        # dof_vel[:, self.right_arm_ids] = visual_motion_frame[42]
        
        # # 52-dim data
        # dof_pos[:, self.left_arm_ids] = visual_motion_frame[18]
        # dof_pos[:, self.right_arm_ids] = visual_motion_frame[22]
        
        # lin_vel = visual_motion_frame[26:29].clone()
        
        # dof_vel[:, self.left_leg_ids]  = visual_motion_frame[[33, 32, 34, 35, 36]]
        # dof_vel[:, self.right_leg_ids] = visual_motion_frame[[39, 38, 40, 41, 42]]
        # dof_vel[:, self.left_arm_ids]  = visual_motion_frame[44]
        # dof_vel[:, self.right_arm_ids] = visual_motion_frame[48]
        
        # # 260515 from eric zhao
        dof_pos[:, self.left_leg_ids[:5]] = visual_motion_frame[6:11]  # ankle_roll(6th) stays 0
        dof_pos[:, self.right_leg_ids[:5]] = visual_motion_frame[11:16]  # ankle_roll(6th) stays 0
        
        dof_pos[:, self.waist_yaw_ids] = visual_motion_frame[16]
        
        dof_pos[:, self.left_arm_ids] = visual_motion_frame[17]
        dof_pos[:, self.right_arm_ids] = visual_motion_frame[18]
        
        lin_vel = visual_motion_frame[19:22].clone()
        ang_vel = visual_motion_frame[22:25].clone()
        
        dof_vel[:, self.left_leg_ids[:5]]  = visual_motion_frame[25:30]
        dof_vel[:, self.right_leg_ids[:5]] = visual_motion_frame[30:35]
        
        dof_vel[:, self.waist_yaw_ids] = visual_motion_frame[35]
        
        dof_vel[:, self.left_arm_ids]  = visual_motion_frame[36]
        dof_vel[:, self.right_arm_ids] = visual_motion_frame[37]
        

        # dof_pos[:, self.left_leg_ids] = visual_motion_frame[6:12]
        # dof_pos[:, self.right_leg_ids] = visual_motion_frame[12:18]

        # if frame_dim == 44:
        #     dof_pos[:, self.left_arm_ids]  = visual_motion_frame[18:20]
        #     dof_pos[:, self.right_arm_ids] = visual_motion_frame[20:22]

        #     lin_vel = visual_motion_frame[22:25].clone()

        #     dof_vel[:, self.left_leg_ids]  = visual_motion_frame[28:34]
        #     dof_vel[:, self.right_leg_ids] = visual_motion_frame[34:40]
        #     dof_vel[:, self.left_arm_ids]  = visual_motion_frame[40:42]
        #     dof_vel[:, self.right_arm_ids] = visual_motion_frame[42:44]
        # elif frame_dim == 52:
        #     dof_pos[:, self.left_arm_ids]  = visual_motion_frame[[18, 21]]
        #     dof_pos[:, self.right_arm_ids] = visual_motion_frame[[22, 25]]

        #     lin_vel = visual_motion_frame[26:29].clone()

        #     dof_vel[:, self.left_leg_ids]  = visual_motion_frame[32:38]
        #     dof_vel[:, self.right_leg_ids] = visual_motion_frame[38:44]
        #     dof_vel[:, self.left_arm_ids]  = visual_motion_frame[[44, 47]]
        #     dof_vel[:, self.right_arm_ids] = visual_motion_frame[[48, 51]]
        # else:
        #     raise RuntimeError(
        #         f"Unsupported motion frame dimension: {frame_dim}. Expected 44 (2-dof arm) "
        #         f"or 52 (4-dof arm). Check your motion_visualization/*.txt files."
        #     )

        # ang_vel = torch.zeros_like(lin_vel)

        self.robot.write_joint_position_to_sim(dof_pos)
        self.robot.write_joint_velocity_to_sim(dof_vel)

        env_ids = torch.arange(self.num_envs, device=device)

        # root state: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        root_state = torch.zeros((self.num_envs, 13), device=device)
        root_state[:, 0:3] = torch.tile(root_pos.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 3:7] = torch.tile(quat_wxyz.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 7:10] = torch.tile(lin_vel.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 10:13] = torch.tile(ang_vel.unsqueeze(0), (self.num_envs, 1))

        self.robot.write_root_state_to_sim(root_state, env_ids)
        self.sim.render()
        self.scene.update(dt=self.step_dt)

        left_hand_pos = (
            self.robot.data.body_state_w[:, self.shoulder_body_ids[0], :3]
            - self.robot.data.root_state_w[:, 0:3]
            + quat_apply(self.robot.data.body_state_w[:, self.shoulder_body_ids[0], 3:7], self.left_arm_local_vec)
        )
        right_hand_pos = (
            self.robot.data.body_state_w[:, self.shoulder_body_ids[1], :3]
            - self.robot.data.root_state_w[:, 0:3]
            + quat_apply(self.robot.data.body_state_w[:, self.shoulder_body_ids[1], 3:7], self.right_arm_local_vec)
        )
        left_hand_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_hand_pos)
        right_hand_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_hand_pos)
        left_foot_pos = (
            self.robot.data.body_state_w[:, self.feet_body_ids[0], :3] - self.robot.data.root_state_w[:, 0:3]
        )
        right_foot_pos = (
            self.robot.data.body_state_w[:, self.feet_body_ids[1], :3] - self.robot.data.root_state_w[:, 0:3]
        )
        left_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_foot_pos)
        right_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_foot_pos)


        self.left_leg_dof_pos = dof_pos[:, self.left_leg_ids]
        self.right_leg_dof_pos = dof_pos[:, self.right_leg_ids]
        self.left_leg_dof_vel = dof_vel[:, self.left_leg_ids]
        self.right_leg_dof_vel = dof_vel[:, self.right_leg_ids]
        self.left_arm_dof_pos = dof_pos[:, self.left_arm_ids]
        self.right_arm_dof_pos = dof_pos[:, self.right_arm_ids]
        self.left_arm_dof_vel = dof_vel[:, self.left_arm_ids]
        self.right_arm_dof_vel = dof_vel[:, self.right_arm_ids]
        self.waist_yaw_dof_pos = dof_pos[:, self.waist_yaw_ids]
        self.waist_yaw_dof_vel = dof_vel[:, self.waist_yaw_ids]
        return torch.cat(
            (
                self.right_arm_dof_pos,
                self.left_arm_dof_pos,
                self.right_leg_dof_pos,
                self.left_leg_dof_pos,
                self.waist_yaw_dof_pos,
                self.right_arm_dof_vel,
                self.left_arm_dof_vel,
                self.right_leg_dof_vel,
                self.left_leg_dof_vel,
                self.waist_yaw_dof_vel,
                left_hand_pos,
                right_hand_pos,
                left_foot_pos,
                right_foot_pos,
            ),
            dim=-1,
        )

    def _domain_rand_material_on_reset(self) -> bool:
        """True if physics_material domain rand re-samples on env reset."""
        events = self.cfg.domain_rand.events
        mat_event = getattr(events, "physics_material", None)
        return mat_event is not None and getattr(mat_event, "mode", None) == "reset"

    def _validate_height_scanners(self) -> None:
        """Assert configured grid sizes match Isaac RayCaster num_rays."""
        hs_cfg = self.cfg.scene.height_scanner
        if hs_cfg.enable_height_scan:
            actual = int(self.height_scanner.num_rays)
            if actual != self.num_height_scan:
                raise ValueError(
                    f"height_scanner num_rays={actual} != configured {self.num_height_scan} "
                    f"(size={hs_cfg.size}, res={hs_cfg.resolution})"
                )
        if hs_cfg.enable_feet_height_scan:
            expected = self.num_feet_scan_points
            for scanner in self.feet_height_scanners:
                actual = int(scanner.num_rays)
                if actual != expected:
                    raise ValueError(
                        f"feet height scanner num_rays={actual} != configured {expected} "
                        f"(size={hs_cfg.feet_scan_size}, res={hs_cfg.feet_scan_resolution})"
                    )

    def _cache_physics_material_coeffs(self) -> None:
        """Cache per-env friction/restitution after startup domain rand (critic-only)."""
        mats = self.robot.root_physx_view.get_material_properties()
        if mats is None:
            self.friction_coeffs = torch.ones(self.num_envs, 1, device=self.device)
            self.restitution_coeffs = torch.ones(self.num_envs, 1, device=self.device)
            return
        mats = mats.to(self.device)
        self.friction_coeffs = mats[..., 0].mean(dim=-1, keepdim=True)
        self.restitution_coeffs = mats[..., 2].mean(dim=-1, keepdim=True)

    def _refresh_domain_rand_obs_cache(self, env_ids: torch.Tensor | None = None) -> None:
        """Refresh cached domain-rand features when material is re-randomized on reset."""
        del env_ids  # full refresh; per-env slice not needed for current PhysX API
        if not self.refresh_material_on_reset:
            return
        self._cache_physics_material_coeffs()

    def _update_style_from_command(self, env_ids: torch.Tensor | None = None) -> None:
        """style：vx/vy/vyaw 均 < stand_enter 为 stand；否则按 |vx| 滞回划分 walk/run。"""
        thr = self.cfg.style_from_command
        cmd = self.command_generator.command
        speed_x = torch.abs(cmd[:, 0])
        cmd_active_enter = (
            (torch.abs(cmd[:, 0]) >= thr.stand_enter)
            | (torch.abs(cmd[:, 1]) >= thr.stand_enter)
            | (torch.abs(cmd[:, 2]) >= thr.stand_enter)
        )
        cmd_active_exit = (
            (torch.abs(cmd[:, 0]) >= thr.stand_exit)
            | (torch.abs(cmd[:, 1]) >= thr.stand_exit)
            | (torch.abs(cmd[:, 2]) >= thr.stand_exit)
        )
        is_stand = (
            (torch.abs(cmd[:, 0]) < thr.stand_enter)
            & (torch.abs(cmd[:, 1]) < thr.stand_enter)
            & (torch.abs(cmd[:, 2]) < thr.stand_enter)
        )
        prev = self.style_ids
        new = prev.clone()

        from_stand = prev == 0
        new[from_stand & (speed_x >= thr.run_enter)] = 2
        new[from_stand & cmd_active_exit & (speed_x < thr.run_enter)] = 1

        from_run = prev == 2
        new[from_run & (speed_x < thr.run_exit)] = 1
        new[from_run & ~cmd_active_enter] = 0

        from_walk = prev == 1
        new[from_walk & (speed_x >= thr.run_enter)] = 2
        new[from_walk & ~cmd_active_enter] = 0

        new = torch.where(is_stand, torch.zeros_like(new), new)

        if env_ids is not None and len(env_ids) > 0:
            self.style_ids[env_ids] = new[env_ids]
        else:
            self.style_ids = new

    def _update_height_scan_cache(self) -> None:
        if self.num_height_scan > 0:
            self._cached_height_scan = mdp_obs.compute_height_scan(
                self.height_scanner,
                self.cfg.normalization.height_scan_offset,
                self.obs_scales.height_scan,
            )
        else:
            self._cached_height_scan = None

    def _get_heights_below_base(self) -> torch.Tensor:
        """Ground height [m] at base scan grid center (nogait heights_below_base)."""
        if self.num_height_scan > 0:
            hit_z = mdp_obs.sanitize_hit_heights_z(
                self.height_scanner.data.ray_hits_w[..., 2], self.height_scanner.data.pos_w[:, 2]
            )
            return hit_z[:, self.height_scan_center_idx].unsqueeze(1)
        return torch.zeros(self.num_envs, 1, device=self.device)

    def _get_measured_feet_ground_heights(self) -> torch.Tensor:
        """Mean ground z under each foot scan grid, shape (N, 2)."""
        if self.feet_height_scanners:
            return mdp_obs.feet_ground_height_mean(self.feet_height_scanners)
        return torch.zeros(self.num_envs, len(self.feet_body_ids), device=self.device)

    def _compute_critic_extra_obs(self) -> torch.Tensor:
        """Privileged critic features (nogait n_critic_extra, without avg_xy EMA)."""
        robot = self.robot
        h_scale = self.critic_height_scale
        target_h = self.critic_base_height_target
        root_z = robot.data.root_pos_w[:, 2]
        heights_below_base = self._get_heights_below_base()

        if self.num_height_scan > 0:
            height_rel = self._cached_height_scan
            if height_rel is None:
                height_rel = mdp_obs.compute_height_scan(
                    self.height_scanner,
                    self.cfg.normalization.height_scan_offset,
                    self.obs_scales.height_scan,
                )
            base_height = (
                torch.clip(height_rel[:, self.height_scan_center_idx].unsqueeze(1), -1.0, 1.0) * h_scale
            )
        else:
            base_height = torch.clip(root_z.unsqueeze(1) - target_h, -1.0, 1.0) * h_scale

        root_lin_vel_yaw = mdp_obs.root_lin_vel_yaw_frame(robot, self.obs_scales.lin_vel)

        feet_vel_z = robot.data.body_lin_vel_w[:, self.feet_body_ids, 2]
        net_contact_forces = self.contact_sensor.data.net_forces_w_history
        foot_force_z = (
            net_contact_forces[:, -1, self.feet_cfg.body_ids, 2] * self.critic_foot_force_scale
        )
        feet_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 1.0
        ).float()

        base_h_error = torch.clip(
            root_z.unsqueeze(1) - heights_below_base - target_h, -0.5, 0.5
        ) * h_scale
        foot_z = robot.data.body_pos_w[:, self.feet_body_ids, 2]
        if self.feet_height_scanners:
            ground_h = self._get_measured_feet_ground_heights()
            foot_clearance = torch.clip(
                foot_z - ground_h - self.critic_foot_height_offset, -1.0, 1.0
            ) * h_scale
        else:
            foot_clearance = torch.clip(foot_z - self.critic_foot_height_offset, -1.0, 1.0) * h_scale

        domain_rand_feats = mdp_obs.domain_rand_features(self)
        return torch.cat(
            [
                base_height,
                root_lin_vel_yaw,
                feet_vel_z,
                foot_force_z,
                feet_contact,
                base_h_error,
                foot_clearance,
                domain_rand_feats,
            ],
            dim=-1,
        )

    def compute_current_observations(self):
        robot = self.robot

        ang_vel = robot.data.root_ang_vel_b
        projected_gravity = robot.data.projected_gravity_b
        command = self.command_generator.command
        joint_pos = robot.data.joint_pos - robot.data.default_joint_pos
        joint_vel = robot.data.joint_vel - robot.data.default_joint_vel
        action = self.action_buffer._circular_buffer.buffer[:, -1, :]

        # no-gait actor obs: 无步态相位、无 style one-hot（style 由 |cmd_x| 推断；AMP 仍用 style_ids）
        current_actor_obs = torch.cat(
            [
                ang_vel * self.obs_scales.ang_vel,  # 3
                projected_gravity * self.obs_scales.projected_gravity,  # 3
                command * self.command_obs_scale,  # 3 (lin_vel_x, lin_vel_y, ang_vel_z)
                joint_pos * self.obs_scales.joint_pos,  # num_actions
                joint_vel * self.obs_scales.joint_vel,  # num_actions
                action * self.obs_scales.actions,  # num_actions
            ],
            dim=-1,
        )
        current_critic_obs = torch.cat([current_actor_obs, self._compute_critic_extra_obs()], dim=-1)

        return current_actor_obs, current_critic_obs

    def compute_observations(self):
        self._update_height_scan_cache()
        current_actor_obs, current_critic_obs = self.compute_current_observations()
        self.num_one_step_obs = int(current_actor_obs.shape[1])
        self.num_one_step_critic_obs = int(current_critic_obs.shape[1])
        # root_lin_vel_yaw in critic extra: actor(48) + base_height(1)
        self.him_vel_index_in_critic = self.num_one_step_obs + 1
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec

        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)

        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)
        if self._cached_height_scan is not None:
            # nogait n_scan=221：仅进 critic，不进 actor
            critic_obs = torch.cat([critic_obs, self._cached_height_scan], dim=-1)

        if self.cfg.scene.depth_camera.enable_depth_camera:
            depth_image = self.depth_camera.data.output["distance_to_image_plane"]

            # (num_envs, height, width, 1) --> (num_envs, height * width)
            flattened_depth = depth_image.view(self.num_envs, -1)

            # Append the flattened depth data to the end of the actor and critic observation vectors.
            actor_obs = torch.cat([actor_obs, flattened_depth], dim=-1)
            critic_obs = torch.cat([critic_obs, flattened_depth], dim=-1)

        actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)
        critic_obs = torch.clip(critic_obs, -self.clip_obs, self.clip_obs)

        return actor_obs, critic_obs

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return

        # Reset buffer
        self.avg_feet_force_per_step[env_ids] = 0.0
        self.avg_feet_speed_per_step[env_ids] = 0.0

        # if getattr(self, "num_styles", None) is not None:
        #     self.style_ids[env_ids] = torch.randint(
        #         0, self.num_styles, (len(env_ids),), device=self.device
        #     )
        #     self.style_one_hot[env_ids] = torch.nn.functional.one_hot(
        #         self.style_ids[env_ids], num_classes=self.num_styles
        #     ).float()

        self.extras["log"] = dict()
        if self.cfg.scene.terrain_generator is not None:
            if self.cfg.scene.terrain_generator.curriculum:
                terrain_levels = self.update_terrain_levels(env_ids)
                self.extras["log"].update(terrain_levels)
        self._cache_episode_curriculum_metrics(env_ids)
        self.extras["log"].update(self._cache_episode_gait_metrics(env_ids))

        self.scene.reset(env_ids)
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="reset",
                env_ids=env_ids,
                dt=self.step_dt,
                global_env_step_count=self.sim_step_counter // self.cfg.sim.decimation,
            )
        self._refresh_domain_rand_obs_cache(env_ids)

        reward_extras = self.reward_manager.reset(env_ids)
        self.extras["log"].update(reward_extras)
        self.extras["time_outs"] = self.time_out_buf

        self.command_generator.reset(env_ids)
        self._update_style_from_command(env_ids)
        self.actor_obs_buffer.reset(env_ids)
        self.critic_obs_buffer.reset(env_ids)
        self.action_buffer.reset(env_ids)
        self.episode_length_buf[env_ids] = 0
        self._reset_reward_lazy_buffers(env_ids)
        # 清零步态累计并假设落地（初始站姿双脚着地），避免下一局首步误判落脚沿
        self._run_step_count[env_ids] = 0.0
        self._run_forward_dist[env_ids] = 0.0
        self._run_flight_steps[env_ids] = 0.0
        self._run_style_steps[env_ids] = 0.0
        self._gait_foot_contact_prev[env_ids] = True
        self._foot_strike_offset[env_ids] = 0.0
        self._foot_strike_offset_sum[env_ids] = 0.0

        self.scene.write_data_to_sim()
        self.sim.forward()
        vel_yaw = math_utils.quat_apply_inverse(
            math_utils.yaw_quat(self.robot.data.root_quat_w[env_ids]), self.robot.data.root_lin_vel_w[env_ids, :3]
        )
        self.avg_lin_vel_x_yaw[env_ids] = vel_yaw[:, 0]
        self.avg_lin_vel_y_yaw[env_ids] = vel_yaw[:, 1]
        self.avg_ang_vel_z[env_ids] = self.robot.data.root_ang_vel_w[env_ids, 2]
        if hasattr(self, "track_lin_vel_x_yaw_avg_buffer"):
            self.track_lin_vel_x_yaw_avg_buffer[env_ids] = vel_yaw[:, 0]

    def step(self, actions: torch.Tensor):
        delayed_actions = self.action_buffer.compute(actions)
        self.action = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)

        processed_actions = (
            self.action * self.action_scale + self.robot.data.default_joint_pos + self.joint_pos_bias
        )

        self.avg_feet_force_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.avg_feet_speed_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            self.robot.set_joint_position_target(processed_actions)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

            self.avg_feet_force_per_step += torch.norm(
                self.contact_sensor.data.net_forces_w[:, self.feet_cfg.body_ids, :3], dim=-1
            )
            self.avg_feet_speed_per_step += torch.norm(self.robot.data.body_lin_vel_w[:, self.feet_body_ids, :], dim=-1)

        self.avg_feet_force_per_step /= self.cfg.sim.decimation
        self.avg_feet_speed_per_step /= self.cfg.sim.decimation

        if not self.headless:
            self.sim.render()

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.last_episode_age += 1

        self.command_generator.compute(self.step_dt)
        self._update_style_from_command()
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self._update_track_avg_velocities()
        self._update_gait_metrics()

        self.reset_buf, self.time_out_buf = self.check_reset()
        reward_buf = self.reward_manager.compute(self.step_dt)
        self.reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()

        if len(self.reset_env_ids) > 0:
            _, termination_critic_obs = self.compute_current_observations()
            termination_critic_obs = termination_critic_obs[self.reset_env_ids]
        else:
            termination_critic_obs = torch.zeros(0, self.num_one_step_critic_obs, device=self.device)

        self.reset(self.reset_env_ids)

        if self.cfg.commands.use_curriculum:
            if self._should_update_command_curriculum():
                self._curriculum_log_cache = self.update_command_curriculum()
            self.extras.setdefault("log", {})
            self.extras["log"].update(self._curriculum_log_cache)

        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {
            "critic": critic_obs,
            "termination_critic_obs": termination_critic_obs,
        }

        return actor_obs, reward_buf, self.reset_buf, self.extras

    def check_reset(self):
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        reset_buf = torch.any(
            torch.max(
                torch.norm(
                    net_contact_forces[:, :, self.termination_contact_cfg.body_ids],
                    dim=-1,
                ),
                dim=1,
            )[0]
            > 1.0,
            dim=1,
        )
        time_out_buf = self.episode_length_buf >= self.max_episode_length
        reset_buf |= time_out_buf
        return reset_buf, time_out_buf

    def init_obs_buffer(self):
        if self.add_noise:
            actor_obs, _ = self.compute_current_observations()
            noise_vec = torch.zeros_like(actor_obs[0])
            noise_scales = self.cfg.noise.noise_scales
            noise_vec[:3] = noise_scales.ang_vel * self.obs_scales.ang_vel
            noise_vec[3:6] = noise_scales.projected_gravity * self.obs_scales.projected_gravity
            noise_vec[6:9] = 0
            noise_vec[9 : 9 + self.num_actions] = noise_scales.joint_pos * self.obs_scales.joint_pos
            noise_vec[9 + self.num_actions : 9 + self.num_actions * 2] = (
                noise_scales.joint_vel * self.obs_scales.joint_vel
            )
            noise_vec[9 + self.num_actions * 2 : 9 + self.num_actions * 3] = 0.0
            self.noise_scale_vec = noise_vec

            if self.num_height_scan > 0:
                height_scan = mdp_obs.compute_height_scan(
                    self.height_scanner,
                    self.cfg.normalization.height_scan_offset,
                    self.obs_scales.height_scan,
                )
                height_scan_noise_vec = torch.zeros_like(height_scan[0])
                height_scan_noise_vec[:] = noise_scales.height_scan * self.obs_scales.height_scan
                self.height_scan_noise_vec = height_scan_noise_vec

        self.actor_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.actor_obs_history_length, batch_size=self.num_envs, device=self.device
        )
        self.critic_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.critic_obs_history_length, batch_size=self.num_envs, device=self.device
        )
        _actor_step, _critic_step = self.compute_current_observations()
        self.num_one_step_obs = int(_actor_step.shape[1])
        self.num_one_step_critic_obs = int(_critic_step.shape[1])
        self.him_vel_index_in_critic = self.num_one_step_obs + 1

    @property
    def actor_obs_dim(self) -> int:
        """Flattened actor observation dim (deploy / sim2sim / export)."""
        return int(self.cfg.robot.actor_obs_history_length) * (9 + self.num_actions * 3)

    @property
    def critic_obs_dim(self) -> int:
        """Flattened critic observation dim (training only)."""
        actor_step = 9 + self.num_actions * 3
        critic_step = actor_step + self.critic_extra_dim
        return int(self.cfg.robot.critic_obs_history_length) * critic_step + self.num_height_scan

    def update_terrain_levels(self, env_ids):
        distance = torch.norm(self.robot.data.root_pos_w[env_ids, :2] - self.scene.env_origins[env_ids, :2], dim=1)
        move_up = distance > self.scene.terrain.cfg.terrain_generator.size[0] / 2
        move_down = (
            distance < torch.norm(self.command_generator.command[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        )
        move_down *= ~move_up
        self.scene.terrain.update_env_origins(env_ids, move_up, move_down)
        extras = {}
        extras["Curriculum/terrain_levels"] = torch.mean(self.scene.terrain.terrain_levels.float())
        return extras

    def _update_track_avg_velocities(self) -> None:
        """EMA of yaw-frame lin_vel_x/y and world ang_vel_z over track_avg_vel_window_s."""
        alpha = min(1.0, self.step_dt / max(self.track_avg_vel_window_s, 1e-6))
        vel_yaw = math_utils.quat_apply_inverse(
            math_utils.yaw_quat(self.robot.data.root_quat_w), self.robot.data.root_lin_vel_w[:, :3]
        )
        lin_vel_x = vel_yaw[:, 0]
        lin_vel_y = vel_yaw[:, 1]
        ang_vel_z = self.robot.data.root_ang_vel_w[:, 2]
        self.avg_lin_vel_x_yaw.mul_(1.0 - alpha).add_(alpha * lin_vel_x)
        self.avg_lin_vel_y_yaw.mul_(1.0 - alpha).add_(alpha * lin_vel_y)
        self.avg_ang_vel_z.mul_(1.0 - alpha).add_(alpha * ang_vel_z)

    def _update_gait_metrics(self) -> None:
        """逐步累计 run style 的步频/步幅/腾空统计（episode 结束时缓存为均值）。"""
        net_forces = self.contact_sensor.data.net_forces_w_history[:, -1, self.feet_cfg.body_ids, :3]
        contact = torch.norm(net_forces, dim=-1) > self._gait_contact_threshold  # (num_envs, 2)
        rising = contact & ~self._gait_foot_contact_prev
        run_mask = (self.style_ids == 2).float()  # 仅统计 run style，避免 stand/walk 污染
        self._run_step_count += rising.any(dim=-1).float() * run_mask
        self._run_flight_steps += (~contact.any(dim=-1)).float() * run_mask
        self._run_style_steps += run_mask
        vel_yaw = math_utils.quat_apply_inverse(
            math_utils.yaw_quat(self.robot.data.root_quat_w), self.robot.data.root_lin_vel_w[:, :3]
        )
        self._run_forward_dist += (torch.clamp(vel_yaw[:, 0], min=0.0) * self.step_dt) * run_mask

        # 落脚点相对质心的前向位移（yaw 系，前向为正），在落脚沿采样缓存
        yaw_q = math_utils.yaw_quat(self.robot.data.root_quat_w)
        rel = self.robot.data.body_pos_w[:, self.feet_body_ids, :] - self.robot.data.root_pos_w[:, :3][:, None, :]
        rel_yaw = math_utils.quat_apply_inverse(
            yaw_q[:, None, :].expand(-1, 2, -1).reshape(-1, 4), rel.reshape(-1, 3)
        ).reshape(self.num_envs, 2, 3)
        strike_x = rel_yaw[..., 0]  # (N,2) 前向为正
        landed_x = torch.where(rising, strike_x, torch.full_like(strike_x, -1e9))
        strike_offset = landed_x.max(dim=-1).values  # (N,)
        did_land = rising.any(dim=-1) & (self.style_ids == 2)
        self._foot_strike_offset = torch.where(did_land, strike_offset, self._foot_strike_offset)
        self._foot_strike_offset_sum += strike_offset * did_land.float()

        self._gait_foot_contact_prev = contact

    def _reset_reward_lazy_buffers(self, env_ids: torch.Tensor) -> None:
        """reward 里 lazy-init 的滤波 buffer 在 reset 时清零，避免跨 episode 污染。"""
        if hasattr(self, "last_contacts_buffer"):
            self.last_contacts_buffer[env_ids] = False
        if hasattr(self, "feet_distance_buffer"):
            self.feet_distance_buffer[env_ids] = 0.0
        if hasattr(self, "last_feet_distance_buffer"):
            self.last_feet_distance_buffer[env_ids] = 0.0

    def _should_update_command_curriculum(self) -> bool:
        """对齐 2026gym：每 max_episode_length 个 common step 判定一次（边界检测替代 %）。"""
        if self.common_step_counter <= 0:
            return False
        ep_len = int(self.max_episode_length)
        prev_c = self.common_step_counter - 1
        return (prev_c // ep_len) < (self.common_step_counter // ep_len)

    def _cache_episode_curriculum_metrics(self, env_ids: torch.Tensor) -> None:
        """在 reward_manager.reset 清空前缓存上局指标（对齐 legged_gym last_episode_*）。"""
        if len(env_ids) == 0:
            return
        valid = self.episode_length_buf[env_ids] > 0
        if not torch.any(valid):
            return
        env_ids = env_ids[valid]

        reward: RewardManager = self.reward_manager
        ep_len = self.episode_length_buf[env_ids].float().clamp(min=1.0)
        episode_time_s = (ep_len * self.step_dt).clamp(min=self.step_dt)
        self.last_episode_track_lin_vel_x_exp[env_ids] = (
            reward._episode_sums["track_lin_vel_x_run_exp"][env_ids] / episode_time_s
        )
        self.last_episode_length[env_ids] = ep_len
        self.last_episode_age[env_ids] = 0

    def _cache_episode_gait_metrics(self, env_ids: torch.Tensor) -> dict[str, float]:
        """缓存并返回本局 run style 的步频/步幅/腾空占比（在 reward_manager.reset 清空前调用）。"""
        if len(env_ids) == 0:
            return {}
        valid = (self.episode_length_buf[env_ids] > 0) & (self._run_style_steps[env_ids] > 0)
        if not torch.any(valid):
            return {}
        vids = env_ids[valid]

        run_time = self._run_style_steps[vids] * self.step_dt  # 秒
        cadence = self._run_step_count[vids] / run_time.clamp(min=self.step_dt)  # steps/s
        # 步幅 = 完整步态周期（左右各一步，共 2 步）内的前向位移
        stride = 2.0 * self._run_forward_dist[vids] / self._run_step_count[vids].clamp(min=1.0)  # m
        flight = self._run_flight_steps[vids] / self._run_style_steps[vids].clamp(min=1.0)

        strike_offset = self._foot_strike_offset_sum[vids] / self._run_step_count[vids].clamp(min=1.0)  # m

        self.last_episode_cadence[vids] = cadence
        self.last_episode_stride_length[vids] = stride
        self.last_episode_flight_ratio[vids] = flight
        self.last_episode_foot_strike_offset[vids] = strike_offset
        return {
            "Gait/cadence": float(cadence.mean().item()),
            "Gait/stride_length": float(stride.mean().item()),
            "Gait/flight_ratio": float(flight.mean().item()),
            "Gait/foot_strike_com_offset": float(strike_offset.mean().item()),
            "Gait/run_step_count": float(self._run_step_count[vids].mean().item()),
        }

    def _curriculum_rwd_ratio(self, lin_vel_x_max: float) -> float:
        """随 lin_vel_x 上限在 [v_lo, v_hi] 上线性插值跟踪 reward 门槛。"""
        cmd_cfg = self.cfg.commands
        rwd_high = float(getattr(cmd_cfg, "curriculum_rwd_threshold", 0.55))
        rwd_low = getattr(cmd_cfg, "curriculum_rwd_threshold_min", None)
        if rwd_low is None:
            return rwd_high

        v_lo = float(getattr(cmd_cfg, "curriculum_rwd_threshold_v_lo", 5.0))
        v_hi = float(getattr(cmd_cfg, "curriculum_rwd_threshold_v_hi", 15.0))
        if v_hi <= v_lo + 1e-6:
            return rwd_high

        t = max(0.0, min(1.0, (lin_vel_x_max - v_lo) / (v_hi - v_lo)))
        return rwd_high + (float(rwd_low) - rwd_high) * t

    def update_command_curriculum(self):
        """速度课程：两条件升速（对齐 2026gym 逻辑）。

        2026gym 为纯 run 任务故对全 env 取均值；本 env style 随 |cmd_x| 变化，
        仅对 **当前 style_id=2 (run)** 且已有上局统计的 env 取均值，避免
        stand/walk 指令把门槛拉低。

        1. mean(last_episode_length) > ratio * max_episode_length
        2. mean(last_episode_track_lin_vel_x_exp) > rwd_ratio(vmax) * track_lin_vel_x_run_exp.weight
           rwd_ratio 随 lin_vel_x 上限在 [v_lo, v_hi] 从 curriculum_rwd_threshold 线性降至 min
        """
        command_cfg = self.command_generator.cfg
        lin_vel_x_max = float(command_cfg.ranges.lin_vel_x[1])

        reward: RewardManager = self.reward_manager
        lin_track_x_idx = reward._term_names.index("track_lin_vel_x_run_exp")
        lin_track_x_weight = float(reward._term_cfgs[lin_track_x_idx].weight)

        max_ep_len = float(self.max_episode_length)
        run_ids = (self.style_ids == 2).nonzero(as_tuple=False).flatten()
        valid = self.last_episode_length[run_ids] > 0.0
        active_run_ids = run_ids[valid]

        len_ratio = float(getattr(self.cfg.commands, "curriculum_min_mean_episode_ratio", 0.8))
        rwd_ratio = self._curriculum_rwd_ratio(lin_vel_x_max)
        update_step = float(getattr(self.cfg.commands, "curriculum_update_step", 0.2))
        max_lin_vel_x = getattr(self.cfg.commands, "curriculum_max_lin_vel_x")

        mean_ep_len = 0.0
        mean_track = 0.0
        length_ok = False
        track_ok = False
        active_frac = 0.0

        if len(active_run_ids) > 0:
            mean_ep_len = float(self.last_episode_length[active_run_ids].mean().item())
            mean_track = float(self.last_episode_track_lin_vel_x_exp[active_run_ids].mean().item())
            active_frac = float(len(active_run_ids) / max(len(run_ids), 1))
            length_ok = mean_ep_len > len_ratio * max_ep_len
            track_ok = mean_track > rwd_ratio * lin_track_x_weight

        advanced = False
        if length_ok and track_ok and max_lin_vel_x is not None:
            old_max = command_cfg.ranges.lin_vel_x[1]
            new_max = min(old_max + update_step, max_lin_vel_x[1])
            new_range = (command_cfg.ranges.lin_vel_x[0], new_max)
            command_cfg.ranges.lin_vel_x = new_range
            self.cfg.commands.ranges.lin_vel_x = new_range
            self.command_generator.cfg = command_cfg
            advanced = new_max > old_max + 1e-6
            lin_vel_x_max = float(new_max)

        at_cap = float(max_lin_vel_x is not None and lin_vel_x_max >= max_lin_vel_x[1] - 1e-6)
        return {
            "Curriculum/lin_vel_x_max": lin_vel_x_max,
            "Curriculum/run_active_env_frac": active_frac,
            "Curriculum/run_mean_episode_length": mean_ep_len,
            "Curriculum/run_mean_episode_length_ratio": mean_ep_len / max_ep_len if max_ep_len > 0 else 0.0,
            "Curriculum/run_mean_episode_length_gate": len_ratio,
            "Curriculum/run_length_ok": float(length_ok),
            "Curriculum/run_track_lin_vel_x_exp": mean_track,
            "Curriculum/run_track_gate": rwd_ratio * lin_track_x_weight,
            "Curriculum/curriculum_rwd_threshold": rwd_ratio,
            "Curriculum/run_track_ok": float(track_ok),
            "Curriculum/run_advanced": float(advanced),
            "Curriculum/run_at_cap": at_cap,
        }

    def get_observations(self):
        actor_obs, critic_obs = self.compute_observations()
        self.extras.setdefault("observations", {})
        self.extras["observations"]["critic"] = critic_obs
        return actor_obs, self.extras

    def get_amp_obs_for_expert_trans(self, env_ids: torch.Tensor | None = None):
        """Gets AMP obs from policy. Pass env_ids to compute only resetting envs."""
        if env_ids is None or len(env_ids) >= self.num_envs:
            idx = slice(None)
        else:
            idx = env_ids

        root_pos = self.robot.data.root_state_w[idx, 0:3]
        root_quat = self.robot.data.root_state_w[idx, 3:7]
        body_state = self.robot.data.body_state_w[idx]
        joint_pos = self.robot.data.joint_pos[idx]
        joint_vel = self.robot.data.joint_vel[idx]

        left_hand_pos = (
            body_state[:, self.shoulder_body_ids[0], :3]
            - root_pos
            + quat_apply(body_state[:, self.shoulder_body_ids[0], 3:7], self.left_arm_local_vec)
        )
        right_hand_pos = (
            body_state[:, self.shoulder_body_ids[1], :3]
            - root_pos
            + quat_apply(body_state[:, self.shoulder_body_ids[1], 3:7], self.right_arm_local_vec)
        )
        left_hand_pos = quat_apply(quat_conjugate(root_quat), left_hand_pos)
        right_hand_pos = quat_apply(quat_conjugate(root_quat), right_hand_pos)
        left_foot_pos = body_state[:, self.feet_body_ids[0], :3] - root_pos
        right_foot_pos = body_state[:, self.feet_body_ids[1], :3] - root_pos
        left_foot_pos = quat_apply(quat_conjugate(root_quat), left_foot_pos)
        right_foot_pos = quat_apply(quat_conjugate(root_quat), right_foot_pos)
        return torch.cat(
            (
                joint_pos[:, self.right_arm_ids],
                joint_pos[:, self.left_arm_ids],
                joint_pos[:, self.right_leg_ids],
                joint_pos[:, self.left_leg_ids],
                joint_pos[:, self.waist_yaw_ids],
                joint_vel[:, self.right_arm_ids],
                joint_vel[:, self.left_arm_ids],
                joint_vel[:, self.right_leg_ids],
                joint_vel[:, self.left_leg_ids],
                joint_vel[:, self.waist_yaw_ids],
                left_hand_pos,
                right_hand_pos,
                left_foot_pos,
                right_foot_pos,
            ),
            dim=-1,
        )

    def get_amp_observations_pos(self):
        """WGAN 正样本：按当前 AMP style 从对应专家库采样（支持 style_2 多段 run 随机）。"""
        times = self.episode_length_buf * self.step_dt
        amp_dim = int(next(iter(self.amp_loader_dict.values())).trajectories[0].shape[1])
        expert_data = torch.zeros(self.num_envs, amp_dim, device=self.device, dtype=torch.float32)
        for style_id in range(self.num_styles):
            env_ids = (self.style_ids == style_id).nonzero(as_tuple=True)[0]
            n = int(env_ids.numel())
            if n == 0:
                continue
            loader = self.amp_loader_dict[style_id]
            traj_idxs = loader.weighted_traj_idx_sample_batch(n)
            frames = loader.get_frame_at_time_batch(traj_idxs, times[env_ids].detach().cpu().numpy())
            expert_data.index_copy_(0, env_ids, frames.to(dtype=torch.float32, device=self.device))
        return expert_data

    @staticmethod
    def seed(seed: int = -1) -> int:
        try:
            import omni.replicator.core as rep  # type: ignore

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        return torch_utils.set_seed(seed)

