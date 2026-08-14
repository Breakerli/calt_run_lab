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

import glob
import json

import numpy as np
import torch


class AMPLoader:
    """Expert dataset providing AMP observations for a configurable joint layout.

    The default layout is the 13-DoF UltraGameYaw robot: right_arm(1) +
    left_arm(1) + right_leg(5) + left_leg(5) + waist(1). For a 6-DoF-leg robot
    (e.g. yrp03-v2), pass ``leg_joint_size=6`` with the matching mirror flip
    indices (see :class:`AMPLoaderYrp`).
    """

    def __init__(
        self,
        device,
        time_between_frames,
        data_dir="",
        preload_transitions=False,
        num_preload_transitions=1000000,
        motion_files=glob.glob("datasets/motion_amp_expert/*"),
        mirror_motion=False,
        arm_joint_size=1,
        leg_joint_size=5,
        waist_joint_size=1,
        leg_mirror_flip_indices=(0, 1),
        mirror_waist=True,
    ):
        """Expert dataset provides AMP observations from Dog mocap dataset.

        time_between_frames: Amount of time in seconds between transition.

        arm_joint_size: Number of joints per arm (default 1).
        leg_joint_size: Number of joints per leg (5 for UltraGameYaw, 6 for
            yrp03-v2 which adds an ankle_roll joint).
        waist_joint_size: Number of waist joints (default 1).
        leg_mirror_flip_indices: Indices, relative to the start of each leg
            segment, whose sign is flipped when mirroring (roll/yaw axes).
        mirror_waist: Whether to flip the sign of the waist joint(s) when
            mirroring.
        """
        self.device = device
        self.time_between_frames = time_between_frames

        self.arm_joint_size = arm_joint_size
        self.leg_joint_size = leg_joint_size
        self.waist_joint_size = waist_joint_size
        self.leg_mirror_flip_indices = tuple(leg_mirror_flip_indices)
        self.mirror_waist = mirror_waist

        self.joint_pos_size = 2 * arm_joint_size + 2 * leg_joint_size + waist_joint_size
        self.joint_vel_size = self.joint_pos_size
        self.end_effector_pos_size = 12

        self.joint_pose_start_idx = 0
        self.joint_pose_end_idx = self.joint_pose_start_idx + self.joint_pos_size

        self.joint_vel_start_idx = self.joint_pose_end_idx
        self.joint_vel_end_idx = self.joint_vel_start_idx + self.joint_vel_size

        self.end_pos_start_idx = self.joint_vel_end_idx
        self.end_pos_end_idx = self.end_pos_start_idx + self.end_effector_pos_size

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
                if motion_data.shape[1] < self.end_pos_end_idx:
                    raise ValueError(
                        f"AMP frame dim {motion_data.shape[1]} is smaller than expected {self.end_pos_end_idx}."
                    )
                motion_tensor = torch.tensor(
                    motion_data[:, : self.end_pos_end_idx], dtype=torch.float32, device=device
                )
                frame_duration = float(motion_json["FrameDuration"])
                traj_len = (motion_data.shape[0] - 1) * frame_duration

                # Remove first 7 observation dimensions (root_pos and root_orn).
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

    def mirror_amp_observations(self, frames):
        """Mirror AMP observations in the configured expert feature layout.

        Joint layout (per pos/vel block): right_arm, left_arm, right_leg,
        left_leg, waist. Mirroring swaps arms and legs, and negates the
        configured roll/yaw axes (``leg_mirror_flip_indices`` per leg + waist).
        """
        mirrored = frames.clone()
        if mirrored.shape[-1] < self.end_pos_end_idx:
            raise ValueError(
                f"AMP frame dim {mirrored.shape[-1]} is smaller than expected {self.end_pos_end_idx}."
            )

        arm, leg = self.arm_joint_size, self.leg_joint_size
        for offset in (self.joint_pose_start_idx, self.joint_vel_start_idx):
            self._swap_segments(mirrored, offset, offset + arm, arm)
            leg_start = offset + 2 * arm
            self._swap_segments(mirrored, leg_start, leg_start + leg, leg)
            flip = [
                side_start + i
                for side_start in (leg_start, leg_start + leg)
                for i in self.leg_mirror_flip_indices
            ]
            if self.mirror_waist:
                flip.append(offset + 2 * arm + 2 * leg)
            mirrored[..., flip] *= -1.0

        # End-effector layout: left_hand, right_hand, left_foot, right_foot in root frame.
        self._swap_segments(mirrored, self.end_pos_start_idx + 0, self.end_pos_start_idx + 3, 3)
        self._swap_segments(mirrored, self.end_pos_start_idx + 6, self.end_pos_start_idx + 9, 3)
        mirrored[..., [
            self.end_pos_start_idx + 1,
            self.end_pos_start_idx + 4,
            self.end_pos_start_idx + 7,
            self.end_pos_start_idx + 10,
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
            len(traj_idxs), self.end_pos_end_idx - self.joint_pose_start_idx, device=self.device
        )
        all_frame_amp_ends = torch.zeros(
            len(traj_idxs), self.end_pos_end_idx - self.joint_pose_start_idx, device=self.device
        )
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_amp_starts[traj_mask] = trajectory[idx_low[traj_mask]][
                :, self.joint_pose_start_idx : self.end_pos_end_idx
            ]
            all_frame_amp_ends[traj_mask] = trajectory[idx_high[traj_mask]][
                :, self.joint_pose_start_idx : self.end_pos_end_idx
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
        """Linearly interpolate between two frames, including orientation.

        Args:
            frame0: First frame to be blended corresponds to (blend = 0).
            frame1: Second frame to be blended corresponds to (blend = 1).
            blend: Float between [0, 1], specifying the interpolation between
            the two frames.
        Returns:
            An interpolation of the two frames.
        """

        joints0, joints1 = self.get_joint_pose(frame0), self.get_joint_pose(frame1)
        joint_vel_0, joint_vel_1 = self.get_joint_vel(frame0), self.get_joint_vel(frame1)

        blend_joint_q = self.slerp(joints0, joints1, blend)
        blend_joints_vel = self.slerp(joint_vel_0, joint_vel_1, blend)

        return torch.cat([blend_joint_q, blend_joints_vel])

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """Generates a batch of AMP transitions."""
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                idxs = np.random.choice(self.preloaded_s.shape[0], size=mini_batch_size)
                s = self.preloaded_s[idxs, self.joint_pose_start_idx : self.end_pos_end_idx]
                s_next = self.preloaded_s_next[idxs, self.joint_pose_start_idx : self.end_pos_end_idx]
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

    def get_joint_pose(self, pose):
        return pose[self.joint_pose_start_idx : self.joint_pose_end_idx]

    def get_joint_pose_batch(self, poses):
        return poses[:, self.joint_pose_start_idx : self.joint_pose_end_idx]

    def get_joint_vel(self, pose):
        return pose[self.joint_vel_start_idx : self.joint_vel_end_idx]

    def get_joint_vel_batch(self, poses):
        return poses[:, self.joint_vel_start_idx : self.joint_vel_end_idx]

    def get_end_pos(self, pose):
        return pose[self.end_pos_start_idx : self.end_pos_end_idx]

    def get_end_pos_batch(self, poses):
        return poses[:, self.end_pos_start_idx : self.end_pos_end_idx]
