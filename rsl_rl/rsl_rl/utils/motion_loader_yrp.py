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
# This file is a copy of motion_loader.AMPLoader adapted for the yrp03-v2 robot
# (15 DoF: the ultra_GameYaw 13 joints + ankle_roll_l/r). Only the joint sizes
# and the left/right mirror layout differ from the original; keep both in sync
# for any future AMP-loader behavior changes.

import glob
import json

import numpy as np
import torch


class AMPLoaderYrp:
    # yrp03-v2: right_arm(1) + left_arm(1) + right_leg(6) + left_leg(6) + waist(1) = 15
    JOINT_POS_SIZE = 15
    JOINT_VEL_SIZE = 15

    END_EFFECTOR_POS_SIZE = 12

    JOINT_POSE_START_IDX = 0
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE

    JOINT_VEL_START_IDX = JOINT_POSE_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE

    END_POS_START_IDX = JOINT_VEL_END_IDX
    END_POS_END_IDX = END_POS_START_IDX + END_EFFECTOR_POS_SIZE

    def __init__(
        self,
        device,
        time_between_frames,
        data_dir="",
        preload_transitions=False,
        num_preload_transitions=1000000,
        motion_files=glob.glob("datasets/motion_amp_expert/*"),
        mirror_motion=False,
    ):
        """Expert dataset providing yrp03-v2 (15-DoF) AMP observations.

        time_between_frames: Amount of time in seconds between transition.
        """
        self.device = device
        self.time_between_frames = time_between_frames

        # Values to store for each trajectory.
        self.trajectories = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # Traj length in seconds.
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []

        for motion_file in motion_files:
            with open(motion_file) as f:
                motion_json = json.load(f)
                motion_data = np.array(motion_json["Frames"])
                motion_tensor = torch.tensor(
                    motion_data[:, : AMPLoaderYrp.END_POS_END_IDX], dtype=torch.float32, device=device
                )
                frame_duration = float(motion_json["FrameDuration"])
                traj_len = (motion_data.shape[0] - 1) * frame_duration

                self._append_trajectory(
                    motion_file,
                    motion_tensor,
                    float(motion_json["MotionWeight"]),
                    frame_duration,
                    traj_len,
                    motion_data.shape[0],
                )
                if mirror_motion:
                    self._append_trajectory(
                        f"{motion_file}::mirror",
                        self.mirror_amp_observations(motion_tensor),
                        float(motion_json["MotionWeight"]),
                        frame_duration,
                        traj_len,
                        motion_data.shape[0],
                    )
                print(f"traj_len:{traj_len}")

            print(f"Loaded {traj_len}s. motion from {motion_file}.")

        # Trajectory weights are used to sample some trajectories more than others.
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # Preload transitions.
        self.preload_transitions = preload_transitions
        if self.preload_transitions:
            print(f"Preloading {num_preload_transitions} transitions")
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            times = self.traj_time_sample_batch(traj_idxs)

            self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times)
            self.preloaded_s_next = self.get_full_frame_at_time_batch(traj_idxs, times + self.time_between_frames)
            print("Finished preloading")

        self.all_trajectories_full = torch.vstack(self.trajectories_full)

    def _append_trajectory(self, name, trajectory, weight, frame_duration, traj_len, num_frames):
        traj_idx = len(self.trajectories)
        self.trajectory_names.append(str(name).split(".")[0])
        self.trajectories.append(trajectory)
        self.trajectories_full.append(trajectory)
        self.trajectory_idxs.append(traj_idx)
        self.trajectory_weights.append(weight)
        self.trajectory_frame_durations.append(frame_duration)
        self.trajectory_lens.append(traj_len)
        self.trajectory_num_frames.append(float(num_frames))

    @staticmethod
    def _swap_segments(tensor, start_a, start_b, length):
        tmp = tensor[..., start_a : start_a + length].clone()
        tensor[..., start_a : start_a + length] = tensor[..., start_b : start_b + length]
        tensor[..., start_b : start_b + length] = tmp

    @staticmethod
    def mirror_amp_observations(frames):
        """Mirror yrp03-v2 AMP observations in the fixed expert feature layout.

        Joint layout (per pos/vel block, 15 dof):
            0 right_arm, 1 left_arm,
            2..7  right_leg [hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll]
            8..13 left_leg  [hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll]
            14 waist_yaw
        Mirroring: swap arms, swap legs, negate roll/yaw axes
        (hip_yaw, hip_roll, ankle_roll for both legs + waist_yaw).
        """
        mirrored = frames.clone()
        if mirrored.shape[-1] < AMPLoaderYrp.END_POS_END_IDX:
            raise ValueError(
                f"AMP frame dim {mirrored.shape[-1]} is smaller than expected {AMPLoaderYrp.END_POS_END_IDX}."
            )

        for offset in (AMPLoaderYrp.JOINT_POSE_START_IDX, AMPLoaderYrp.JOINT_VEL_START_IDX):
            # swap arms (len 1) and legs (len 6)
            AMPLoaderYrp._swap_segments(mirrored, offset + 0, offset + 1, 1)
            AMPLoaderYrp._swap_segments(mirrored, offset + 2, offset + 8, 6)
            # negate roll/yaw axes: right(hip_yaw=2, hip_roll=3, ankle_roll=7),
            #                       left(hip_yaw=8, hip_roll=9, ankle_roll=13), waist=14
            mirrored[..., [offset + 2, offset + 3, offset + 7,
                           offset + 8, offset + 9, offset + 13, offset + 14]] *= -1.0

        # End-effector layout: left_hand, right_hand, left_foot, right_foot in root frame.
        AMPLoaderYrp._swap_segments(mirrored, AMPLoaderYrp.END_POS_START_IDX + 0, AMPLoaderYrp.END_POS_START_IDX + 3, 3)
        AMPLoaderYrp._swap_segments(mirrored, AMPLoaderYrp.END_POS_START_IDX + 6, AMPLoaderYrp.END_POS_START_IDX + 9, 3)
        mirrored[..., [
            AMPLoaderYrp.END_POS_START_IDX + 1,
            AMPLoaderYrp.END_POS_START_IDX + 4,
            AMPLoaderYrp.END_POS_START_IDX + 7,
            AMPLoaderYrp.END_POS_START_IDX + 10,
        ]] *= -1.0
        return mirrored

    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """Batch sample traj idxs."""
        return np.random.choice(self.trajectory_idxs, size=size, p=self.trajectory_weights, replace=True)

    def traj_time_sample(self, traj_idx):
        """Sample random time for traj."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idx]
        return max(0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        """Sample random time for multiple trajectories."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]

        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst

        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, frame1, frame2, blend):
        return (1.0 - blend) * frame1 + blend * frame2

    def get_trajectory(self, traj_idx):
        """Returns trajectory of AMP observations."""
        return self.trajectories_full[traj_idx]

    def get_frame_at_time(self, traj_idx, time):
        """Returns frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low

        return self.slerp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs, times):
        traj_lens = self.trajectory_lens[traj_idxs]
        times = np.mod(times, traj_lens)

        n = self.trajectory_num_frames[traj_idxs]
        p = times / traj_lens

        idx = p * n
        idx_low  = np.floor(idx).astype(np.int64)
        idx_high = idx_low + 1

        idx_low  = np.clip(idx_low,  0, n - 1)
        idx_high = np.clip(idx_high, 0, n - 1)

        all_frame_starts = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        all_frame_ends   = torch.zeros_like(all_frame_starts)

        for traj_idx in np.unique(traj_idxs):
            mask = traj_idxs == traj_idx
            trajectory = self.trajectories[traj_idx]
            all_frame_starts[mask] = trajectory[idx_low[mask]]
            all_frame_ends[mask]   = trajectory[idx_high[mask]]

        blend = torch.tensor(idx - idx_low, device=self.device).clamp(0, 1).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time(self, traj_idx, time):
        """Returns full frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int64), np.ceil(p * n).astype(np.int64)
        all_frame_amp_starts = torch.zeros(
            len(traj_idxs), AMPLoaderYrp.END_POS_END_IDX - AMPLoaderYrp.JOINT_POSE_START_IDX, device=self.device
        )
        all_frame_amp_ends = torch.zeros(
            len(traj_idxs), AMPLoaderYrp.END_POS_END_IDX - AMPLoaderYrp.JOINT_POSE_START_IDX, device=self.device
        )
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_amp_starts[traj_mask] = trajectory[idx_low[traj_mask]][
                :, AMPLoaderYrp.JOINT_POSE_START_IDX : AMPLoaderYrp.END_POS_END_IDX
            ]
            all_frame_amp_ends[traj_mask] = trajectory[idx_high[traj_mask]][
                :, AMPLoaderYrp.JOINT_POSE_START_IDX : AMPLoaderYrp.END_POS_END_IDX
            ]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)

        amp_blend = self.slerp(all_frame_amp_starts, all_frame_amp_ends, blend)
        return torch.cat([amp_blend], dim=-1)

    def get_frame(self):
        """Returns random frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self):
        """Returns random full frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames):
        if self.preload_transitions:
            idxs = np.random.choice(self.preloaded_s.shape[0], size=num_frames)
            return self.preloaded_s[idxs]
        else:
            traj_idxs = self.weighted_traj_idx_sample_batch(num_frames)
            times = self.traj_time_sample_batch(traj_idxs)
            return self.get_full_frame_at_time_batch(traj_idxs, times)

    def blend_frame_pose(self, frame0, frame1, blend):
        """Linearly interpolate between two frames, including orientation."""
        joints0, joints1 = AMPLoaderYrp.get_joint_pose(frame0), AMPLoaderYrp.get_joint_pose(frame1)
        joint_vel_0, joint_vel_1 = AMPLoaderYrp.get_joint_vel(frame0), AMPLoaderYrp.get_joint_vel(frame1)

        blend_joint_q = self.slerp(joints0, joints1, blend)
        blend_joints_vel = self.slerp(joint_vel_0, joint_vel_1, blend)

        return torch.cat([blend_joint_q, blend_joints_vel])

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """Generates a batch of AMP transitions."""
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                idxs = np.random.choice(self.preloaded_s.shape[0], size=mini_batch_size)
                s = self.preloaded_s[idxs, AMPLoaderYrp.JOINT_POSE_START_IDX : AMPLoaderYrp.END_POS_END_IDX]
                s_next = self.preloaded_s_next[idxs, AMPLoaderYrp.JOINT_POSE_START_IDX : AMPLoaderYrp.END_POS_END_IDX]
            else:
                s, s_next = [], []
                traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
                times = self.traj_time_sample_batch(traj_idxs)
                for traj_idx, frame_time in zip(traj_idxs, times):
                    s.append(self.get_frame_at_time(traj_idx, frame_time))
                    s_next.append(self.get_frame_at_time(traj_idx, frame_time + self.time_between_frames))

                s = torch.vstack(s)
                s_next = torch.vstack(s_next)
            yield s, s_next

    @property
    def observation_dim(self):
        """Size of AMP observations."""
        return self.trajectories[0].shape[1]

    @property
    def num_motions(self):
        return len(self.trajectory_names)

    def get_joint_pose(pose):
        return pose[AMPLoaderYrp.JOINT_POSE_START_IDX : AMPLoaderYrp.JOINT_POSE_END_IDX]

    def get_joint_pose_batch(poses):
        return poses[:, AMPLoaderYrp.JOINT_POSE_START_IDX : AMPLoaderYrp.JOINT_POSE_END_IDX]

    def get_joint_vel(pose):
        return pose[AMPLoaderYrp.JOINT_VEL_START_IDX : AMPLoaderYrp.JOINT_VEL_END_IDX]

    def get_joint_vel_batch(poses):
        return poses[:, AMPLoaderYrp.JOINT_VEL_START_IDX : AMPLoaderYrp.JOINT_VEL_END_IDX]

    def get_end_pos(pose):
        return pose[AMPLoaderYrp.END_POS_START_IDX : AMPLoaderYrp.END_POS_END_IDX]

    def get_end_pos_batch(poses):
        return poses[:, AMPLoaderYrp.END_POS_START_IDX : AMPLoaderYrp.END_POS_END_IDX]
