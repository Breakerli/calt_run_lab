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

"""Public exports for the 01RYP-v2 robot asset.

The asset directory and implementation filename intentionally use a hyphen to
match the model name.  Importlib is used because a hyphen is not valid in a
Python ``from .module import ...`` statement.
"""

from importlib import import_module

_config = import_module(".ryp01-v2", __name__)

ROBOT_01RYP_V2_CFG = _config.ROBOT_01RYP_V2_CFG
ROBOT_01RYP_V2_DEFAULT_BASE_HEIGHT = _config.ROBOT_01RYP_V2_DEFAULT_BASE_HEIGHT
ROBOT_01RYP_V2_DEFAULT_JOINT_POS = _config.ROBOT_01RYP_V2_DEFAULT_JOINT_POS
ROBOT_01RYP_V2_JOINT_NAMES = _config.ROBOT_01RYP_V2_JOINT_NAMES

__all__ = [
    "ROBOT_01RYP_V2_CFG",
    "ROBOT_01RYP_V2_DEFAULT_BASE_HEIGHT",
    "ROBOT_01RYP_V2_DEFAULT_JOINT_POS",
    "ROBOT_01RYP_V2_JOINT_NAMES",
]
