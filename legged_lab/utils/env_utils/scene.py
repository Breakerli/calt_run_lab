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

from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, patterns
from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from legged_lab.sensors.camera import TiledCameraCfg
from legged_lab.terrains.ray_caster_cfg import RayCasterCfg

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env_config import BaseSceneCfg


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    def __init__(
        self,
        config: "BaseSceneCfg",
        physics_dt,
        step_dt,
        contact_pre_contact_policy_steps: int = 2,
    ):
        super().__init__(num_envs=config.num_envs, env_spacing=config.env_spacing)
        substeps_per_policy = max(1, int(round(step_dt / physics_dt)))
        pre_phys = substeps_per_policy * contact_pre_contact_policy_steps
        # recent window (pre-landing) + older slots to detect prior touchdown in the same swing
        contact_history_length = 2 * pre_phys + substeps_per_policy + 1

        self.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type=config.terrain_type,
            terrain_generator=config.terrain_generator,
            max_init_terrain_level=config.max_init_terrain_level,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=config.terrain_restitution,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=False,
        )

        self.robot: ArticulationCfg = config.robot.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            history_length=contact_history_length,
            track_air_time=True,
            update_period=physics_dt,
        )

        self.light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
        self.sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=750.0,
                texture_file=(
                    f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"
                ),
            ),
        )

        if config.height_scanner.enable_height_scan:
            self.height_scanner = RayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + config.height_scanner.prim_body_name,
                offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
                ray_alignment="yaw",
                pattern_cfg=patterns.GridPatternCfg(
                    resolution=config.height_scanner.resolution, size=config.height_scanner.size
                ),
                debug_vis=config.height_scanner.debug_vis,
                mesh_prim_paths=["/World/ground"],
                update_period=step_dt,
                drift_range=config.height_scanner.drift_range,
            )

        if config.height_scanner.enable_feet_height_scan:
            hs = config.height_scanner
            for foot_name in hs.feet_body_names:
                sensor_name = f"feet_height_scanner_{foot_name}"
                setattr(
                    self,
                    sensor_name,
                    RayCasterCfg(
                        prim_path="{ENV_REGEX_NS}/Robot/" + foot_name,
                        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 5.0)),
                        ray_alignment="yaw",
                        pattern_cfg=patterns.GridPatternCfg(
                            resolution=hs.feet_scan_resolution,
                            size=hs.feet_scan_size,
                        ),
                        debug_vis=False,
                        mesh_prim_paths=["/World/ground"],
                        update_period=step_dt,
                        drift_range=(0.0, 0.0),
                    ),
                )

        if config.lidar.enable_lidar:
            self.lidar = RayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + config.lidar.prim_body_name,
                offset=RayCasterCfg.OffsetCfg(pos=config.lidar.offset, rot=config.lidar.rotation),
                ray_alignment="yaw",
                pattern_cfg=config.lidar.pattern_cfg,
                debug_vis=config.lidar.debug_vis,
                mesh_prim_paths=config.lidar.mesh_prim_paths,
                max_distance=config.lidar.max_distance,
            )

        if config.depth_camera.enable_depth_camera:
            self.depth_camera = TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + config.depth_camera.prim_body_name,
                offset=config.depth_camera.offset,
                height=config.depth_camera.height,
                width=config.depth_camera.width,
                data_types=config.depth_camera.data_types,
                spawn=config.depth_camera.spawn,
                debug_vis=config.depth_camera.debug_vis,
                visualizer_cfg=config.depth_camera.visualizer_cfg,
            )
