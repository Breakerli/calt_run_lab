from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.mdp.events import _randomize_prop_by_op  # noqa: F401, F403
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def randomize_motor_strength(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    motor_strength_range: tuple[float, float],
):
    """Randomize per-action motor strength scales."""
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    else:
        env_ids = env_ids.to(device=env.device)

    env.motor_strength[env_ids] = math_utils.sample_uniform(
        motor_strength_range[0],
        motor_strength_range[1],
        (len(env_ids), env.num_actions),
        env.device,
    )


def randomize_joint_pos_bias(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    bias_range: tuple[float, float],
):
    """Randomize per-joint position bias simulating encoder calibration error (2026gym motor_offset)."""
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    else:
        env_ids = env_ids.to(device=env.device)

    env.joint_pos_bias[env_ids] = math_utils.sample_uniform(
        bias_range[0],
        bias_range[1],
        (len(env_ids), env.num_actions),
        env.device,
    )


def randomize_actuator_friction(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    static_friction_range: tuple[float, float],
    dynamic_friction_range: tuple[float, float],
):
    """Randomize custom actuator static/viscous friction scale per joint."""
    num_actions = getattr(env, "num_actions", env.scene["robot"].data.default_joint_pos.shape[1])
    if not hasattr(env, "joint_static_friction_scale"):
        env.joint_static_friction_scale = torch.ones(
            env.scene.num_envs, num_actions, dtype=torch.float, device=env.device, requires_grad=False
        )
    if not hasattr(env, "joint_dynamic_friction_scale"):
        env.joint_dynamic_friction_scale = torch.ones(
            env.scene.num_envs, num_actions, dtype=torch.float, device=env.device, requires_grad=False
        )
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    else:
        env_ids = env_ids.to(device=env.device)

    env.joint_static_friction_scale[env_ids] = math_utils.sample_uniform(
        static_friction_range[0],
        static_friction_range[1],
        (len(env_ids), num_actions),
        env.device,
    )
    env.joint_dynamic_friction_scale[env_ids] = math_utils.sample_uniform(
        dynamic_friction_range[0],
        dynamic_friction_range[1],
        (len(env_ids), num_actions),
        env.device,
    )


def randomize_joint_armature(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    armature_distribution_params: tuple[float, float],
):
    """Randomize PhysX joint armature by scaling the default armature."""
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    else:
        env_ids = env_ids.to(device=env.device)

    joint_ids = asset_cfg.joint_ids
    if joint_ids == slice(None):
        default_armature = asset.data.default_joint_armature[env_ids].clone()
        shape = default_armature.shape
    else:
        default_armature = asset.data.default_joint_armature[env_ids[:, None], joint_ids].clone()
        shape = default_armature.shape
    scale = math_utils.sample_uniform(
        armature_distribution_params[0], armature_distribution_params[1], shape, env.device
    )
    asset.write_joint_armature_to_sim(default_armature * scale, joint_ids=joint_ids, env_ids=env_ids)


def randomize_joint_friction_coefficients(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    static_friction_distribution_params: tuple[float, float] = (1.0, 1.0),
    dynamic_friction_distribution_params: tuple[float, float] = (1.0, 1.0),
    viscous_friction_distribution_params: tuple[float, float] = (1.0, 1.0),
):
    """Randomize PhysX joint friction coefficients by scaling defaults."""
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    else:
        env_ids = env_ids.to(device=env.device)

    joint_ids = asset_cfg.joint_ids
    if joint_ids == slice(None):
        default_static = asset.data.default_joint_friction_coeff[env_ids].clone()
        default_dynamic = asset.data.default_joint_dynamic_friction_coeff[env_ids].clone()
        default_viscous = asset.data.default_joint_viscous_friction_coeff[env_ids].clone()
    else:
        default_static = asset.data.default_joint_friction_coeff[env_ids[:, None], joint_ids].clone()
        default_dynamic = asset.data.default_joint_dynamic_friction_coeff[env_ids[:, None], joint_ids].clone()
        default_viscous = asset.data.default_joint_viscous_friction_coeff[env_ids[:, None], joint_ids].clone()

    static_scale = math_utils.sample_uniform(
        static_friction_distribution_params[0],
        static_friction_distribution_params[1],
        default_static.shape,
        env.device,
    )
    dynamic_scale = math_utils.sample_uniform(
        dynamic_friction_distribution_params[0],
        dynamic_friction_distribution_params[1],
        default_dynamic.shape,
        env.device,
    )
    viscous_scale = math_utils.sample_uniform(
        viscous_friction_distribution_params[0],
        viscous_friction_distribution_params[1],
        default_viscous.shape,
        env.device,
    )
    asset.write_joint_friction_coefficient_to_sim(
        default_static * static_scale,
        default_dynamic * dynamic_scale,
        default_viscous * viscous_scale,
        joint_ids=joint_ids,
        env_ids=env_ids,
    )


def randomize_rigid_body_mass_inertia(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    mass_inertia_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the inertia of the bodies by adding, scaling, or setting random values.

    This function allows randomizing the mass of the bodies of the asset. The function samples random values from the
    given distribution parameters and adds, scales, or sets the values into the physics simulation based on the operation.

    .. tip::
        This function uses CPU tensors to assign the body masses. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current inertias of the bodies (num_assets, num_bodies)
    # ToDo: ???为啥还要取mass
    if not hasattr(env, "cached_masses"):
       env.cached_masses = asset.root_physx_view.get_masses().clone()
    masses = env.cached_masses
    if not hasattr(env, "cached_inertias"):
       env.cached_inertias = asset.root_physx_view.get_inertias().clone()
    inertias = env.cached_inertias
    
    masses = _randomize_prop_by_op(
        masses, mass_inertia_distribution_params, env_ids, body_ids, operation=operation, distribution=distribution
    )
    scale = masses / asset.root_physx_view.get_masses()
    inertias[env_ids[:, None], body_ids, :] *= scale.unsqueeze(-1)
    for idx in (0, 4, 8):
        inertias[env_ids[:, None], body_ids, idx] = torch.clamp(
            inertias[env_ids[:, None], body_ids, idx], min=1.0e-9
        )

    asset.root_physx_view.set_masses(masses, env_ids)
    asset.root_physx_view.set_inertias(inertias, env_ids)


def randomize_rigid_body_inertia(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    inertia_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize body inertia tensors by scaling the full 9-element tensor per link.

    Uses one scale factor per body (from default inertia) so the matrix stays physically valid.
    Diagonal entries are clamped to a small positive minimum before writing to PhysX.

    .. tip::
        This function uses CPU tensors to assign the body inertias. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    default_inertia = asset.data.default_inertia[env_ids[:, None], body_ids, :].clone()

    # 对每个 body 使用单一缩放因子作用到完整 9 维惯性张量（保持 PSD，避免只改对角元导致负惯量）
    template = default_inertia[..., 0]
    scale = _randomize_prop_by_op(
        template,
        inertia_distribution_params,
        env_ids,
        body_ids,
        operation,
        distribution,
    )
    inertias = asset.root_physx_view.get_inertias()
    inertias[env_ids[:, None], body_ids, :] = default_inertia * scale.unsqueeze(-1)

    # PhysX 要求主轴惯量非负
    for idx in (0, 4, 8):
        inertias[env_ids[:, None], body_ids, idx] = torch.clamp(
            inertias[env_ids[:, None], body_ids, idx], min=1.0e-9
        )

    asset.root_physx_view.set_inertias(inertias, env_ids)


def apply_external_force_torque_stochastic(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: dict[str, tuple[float, float]],
    torque_range: dict[str, tuple[float, float]],
    probability: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize the external forces and torques applied to the bodies.

    This function creates a set of random forces and torques sampled from the given ranges. The number of forces
    and torques is equal to the number of bodies times the number of environments. The forces and torques are
    applied to the bodies by calling ``asset.set_external_force_and_torque``. The forces and torques are only
    applied when ``asset.write_data_to_sim()`` is called in the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # clear the existing forces and torques
    asset._external_force_b *= 0
    asset._external_torque_b *= 0

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    random_values = torch.rand(env_ids.shape, device=env_ids.device)
    mask = random_values < probability
    masked_env_ids = env_ids[mask]

    if len(masked_env_ids) == 0:
        return

    # resolve number of bodies
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

    # sample random forces and torques
    size = (len(masked_env_ids), num_bodies, 3)
    force_range_list = [force_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    force_range = torch.tensor(force_range_list, device=asset.device)
    forces = math_utils.sample_uniform(force_range[:, 0], force_range[:, 1], size, asset.device)
    torque_range_list = [torque_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    torque_range = torch.tensor(torque_range_list, device=asset.device)
    torques = math_utils.sample_uniform(torque_range[:, 0], torque_range[:, 1], size, asset.device)
    # set the forces and torques into the buffers
    # note: these are only applied when you call: `asset.write_data_to_sim()`
    asset.set_external_force_and_torque(forces, torques, env_ids=masked_env_ids, body_ids=asset_cfg.body_ids)


def base_stable_assistance(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    kp: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # clear the existing forces and torques
    asset.has_external_wrench = True
    asset._external_torque_b[:, asset_cfg.body_ids, 0] = -asset.data.root_ang_vel_b[:, 0].unsqueeze(1) * kp[0]
    asset._external_torque_b[:, asset_cfg.body_ids, 1] = -asset.data.root_ang_vel_b[:, 1].unsqueeze(1) * kp[1]
