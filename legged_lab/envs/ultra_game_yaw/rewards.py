# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

"""Ultra88p-0508 specific reward functions.

This module is reserved for reward functions that are specific to the
Ultra88p-0508 robot (e.g. that depend on its particular joint or link layout
in a way that other robots do not share). It is currently empty: all rewards
used by the Ultra88p-0508 cfg today live in the shared
``legged_lab.mdp.rewards`` module and are accessed via ``mdp.<func>``.

Add new robot-specific rewards here when, and only when, they cannot be made
generic via ``SceneEntityCfg`` / parameterization. Doing so keeps the shared
module portable across robots that differ in DoF count or naming.
"""

from __future__ import annotations

# Placeholder: no Ultra88p-specific reward functions yet.
__all__: list[str] = []
