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
# Thin AMPLoader subclass for the yrp03-v2 robot (15 DoF: the ultra_GameYaw
# 13 joints + ankle_roll_l/r). Only the leg joint size and the mirror flip axes
# differ from the default AMPLoader layout.

from .motion_loader import AMPLoader


class AMPLoaderYrp(AMPLoader):
    """AMPLoader specialized for yrp03-v2 (15 DoF, 6-DoF legs).

    yrp03-v2 adds an ankle_roll joint per leg, so each leg has 6 joints
    [hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll] and mirroring
    additionally flips the ankle_roll axis (leg-local index 5).
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("leg_joint_size", 6)
        kwargs.setdefault("leg_mirror_flip_indices", (0, 1, 5))
        super().__init__(*args, **kwargs)
