# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

# from isaaclab.core.utils.types import ArticulationActions
from isaaclab.utils.types import ArticulationActions

from isaaclab.actuators import DelayedPDActuator
if TYPE_CHECKING:
    from .actuator_cfg import CustomDelayedPDActuatorCfg


class CustomDelayedPDActuator(DelayedPDActuator):
    """Custom actuator class that implements a torque-speed curve for the actuators.

    The torque-speed curve is defined as follows:

            Torque Limit, N·m
                ^
    Y2──────────|
                |──────────────Y1
                |              │\
                |              │ \
                |              │  \
                |              |   \
    ------------+--------------|------> velocity: rad/s
                               X1   X2

    - Y1: Peak Torque Test (Torque and Speed in the Same Direction)
    - Y2: Peak Torque Test (Torque and Speed in the Opposite Direction)
    - X1: Maximum Speed at Full Torque (T-N Curve Knee Point)
    - X2: No-Load Speed Test

    - Fs: Static friction coefficient
    - Fd: Dynamic friction coefficient
    - Va: Velocity at which the friction is fully activated
    """

    cfg: CustomDelayedPDActuatorCfg
    """The configuration for the actuator model."""

    def __init__(self, cfg: CustomDelayedPDActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        # torque-speed curve parameters
        self._joint_vel = torch.zeros_like(self.computed_effort)
        self._effort_y1 = self._parse_joint_parameter(cfg.Y1, 1e9)
        self._effort_y2 = self._parse_joint_parameter(cfg.Y2, cfg.Y1)
        self._velocity_x1 = self._parse_joint_parameter(cfg.X1, 1e9)
        self._velocity_x2 = self._parse_joint_parameter(cfg.X2, 1e9)
        self._friction_static = self._parse_joint_parameter(cfg.Fs, 0.0)
        self._friction_dynamic = self._parse_joint_parameter(cfg.Fd, 0.0)
        self._activation_vel = self._parse_joint_parameter(cfg.Va, 0.01)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # save current joint vel
        self._joint_vel[:] = joint_vel
        # calculate the desired joint torques
        control_action = super().compute(control_action, joint_pos, joint_vel)

        # actuator-side domain randomization is only available inside the training env;
        # standalone tools like mx_tool/show_robot_sim.py don't inject `_env`.
        env = getattr(self, "_env", None)
        friction_static = self._friction_static
        friction_dynamic = self._friction_dynamic
        if env is not None and hasattr(env, "joint_static_friction_scale"):
            friction_static = friction_static * env.joint_static_friction_scale[:, self.joint_indices].to(
                self.applied_effort.device
            )
        if env is not None and hasattr(env, "joint_dynamic_friction_scale"):
            friction_dynamic = friction_dynamic * env.joint_dynamic_friction_scale[:, self.joint_indices].to(
                self.applied_effort.device
            )

        # apply friction model on the torque
        self.applied_effort -= friction_static * torch.tanh(joint_vel / self._activation_vel) + friction_dynamic * joint_vel

        # motor strength randomization
        if env is not None and hasattr(env, "motor_strength"):
            motor_strength = env.motor_strength
            self.applied_effort *= motor_strength[:, self.joint_indices].to(self.applied_effort.device)

        control_action.joint_positions = None
        control_action.joint_velocities = None
        control_action.joint_efforts = self.applied_effort

        return control_action

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        # check if the effort is the same direction as the joint velocity
        same_direction = (self._joint_vel * effort) > 0
        max_effort = torch.where(same_direction, self._effort_y1, self._effort_y2)
        # check if the joint velocity is less than the max speed at full torque
        max_effort = torch.where(
            self._joint_vel.abs() < self._velocity_x1, max_effort, self._compute_effort_limit(max_effort)
        )
        return torch.clip(effort, min=-max_effort, max=max_effort)

    def _compute_effort_limit(self, max_effort):
        k = -max_effort / (self._velocity_x2 - self._velocity_x1)
        limit = k * (self._joint_vel.abs() - self._velocity_x1) + max_effort
        return limit.clip(min=0.0)