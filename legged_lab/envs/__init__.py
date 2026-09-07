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

from legged_lab.envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg
from legged_lab.envs.ryp01_v2.motion_run_cfg import Ryp01V2MotionRunEnvCfg, Ryp01V2MotionRunAgentCfg
from legged_lab.envs.ryp01_v2.ryp01_motion_run_env import Ryp01V2MotionRunEnv
from legged_lab.envs.ultra_game_yaw.motion_run_cfg import UltraGameYawMotionRunEnvCfg, UltraGameYawMotionRunAgentCfg
from legged_lab.envs.ultra_game_yaw.ultra_motion_run_env import UltraGameYawMotionRunEnv
from legged_lab.envs.yrp03_v2.motion_run_cfg import Yrp03V2MotionRunEnvCfg, Yrp03V2MotionRunAgentCfg
from legged_lab.envs.yrp03_v2.yrp03_motion_run_env import Yrp03V2MotionRunEnv

from legged_lab.utils.task_registry import task_registry

task_registry.register("ryp01_v2_run", Ryp01V2MotionRunEnv, Ryp01V2MotionRunEnvCfg(), Ryp01V2MotionRunAgentCfg())
task_registry.register("ultra_game_yaw_run", UltraGameYawMotionRunEnv, UltraGameYawMotionRunEnvCfg(), UltraGameYawMotionRunAgentCfg())
task_registry.register("yrp03_v2_run", Yrp03V2MotionRunEnv, Yrp03V2MotionRunEnvCfg(), Yrp03V2MotionRunAgentCfg())
