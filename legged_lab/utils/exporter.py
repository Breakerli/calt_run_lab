import copy
import os

import torch
import torch.nn.functional as F


def export_policy_as_jit(policy: object, normalizer: object | None, path: str, filename: str = "policy.pt"):
    if hasattr(policy, "estimator") and policy.estimator is not None:
        policy_exporter = JitPolicyExporterHIM(policy, normalizer)
        policy_exporter.export(path, filename)
        return
    from isaaclab_rl.rsl_rl.exporter import export_policy_as_jit as _export_jit

    _export_jit(policy, normalizer, path, filename)


def export_policy_as_onnx(policy: object, normalizer: object | None, path: str, filename: str = "policy.onnx"):
    if hasattr(policy, "estimator") and policy.estimator is not None:
        OnnxPolicyExporterHIM(policy, normalizer).export(path, filename)
        return
    from isaaclab_rl.rsl_rl.exporter import export_policy_as_onnx as _export_onnx

    _export_onnx(policy, normalizer=normalizer, path=path, filename=filename)


class JitPolicyExporterHIM(torch.nn.Module):
    """Export HIM policy (estimator encoder + actor) as TorchScript."""

    def __init__(self, actor_critic, normalizer: object | None = None):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.estimator = copy.deepcopy(actor_critic.estimator.encoder)
        self.normalizer = copy.deepcopy(normalizer) if normalizer is not None else torch.nn.Identity()
        self.num_one_step_obs = int(actor_critic.num_one_step_obs)
        self.actor_recent_frames = int(getattr(actor_critic, "actor_recent_frames", 1))
        self.num_latent = int(actor_critic.num_latent)
        self.vel_scale_for_actor = float(getattr(actor_critic, "vel_scale_for_actor", 1.0))

    def forward(self, obs_history):
        single_batch = obs_history.dim() == 1
        if single_batch:
            obs_history = obs_history.unsqueeze(0)
        obs_history = self.normalizer(obs_history)
        parts = self.estimator(obs_history)
        vel, z = parts[..., :3], parts[..., 3 : 3 + self.num_latent]
        vel = vel * self.vel_scale_for_actor
        z = F.normalize(z, dim=-1, p=2.0)
        n = int(self.actor_recent_frames)
        if n <= 0:
            recent = obs_history[:, 0:0]
        else:
            recent = obs_history[:, -(n * self.num_one_step_obs) :]
        actions = self.actor(torch.cat((recent, vel, z), dim=1))
        if single_batch:
            actions = actions.squeeze(0)
        return actions

    @torch.jit.export
    def estimate_lin_vel(self, obs_history: torch.Tensor) -> torch.Tensor:
        """Yaw-frame linear velocity [m/s] from HIM encoder (LCM / sim2sim debug)."""
        if obs_history.dim() == 1:
            obs_history = obs_history.unsqueeze(0)
        obs_history = self.normalizer(obs_history)
        return self.estimator(obs_history)[:, :3]

    def export(self, path: str, filename: str = "policy.pt"):
        os.makedirs(path, exist_ok=True)
        out_path = os.path.join(path, filename)
        self.to("cpu")
        traced = torch.jit.script(self)
        traced.save(out_path)
        print(f"[exporter] HIM TorchScript saved: {out_path}")


class OnnxPolicyExporterHIM(JitPolicyExporterHIM):
    """Export HIM policy (encoder + actor) as ONNX with actions and lin_vel_est outputs."""

    def __init__(self, actor_critic, normalizer: object | None = None):
        super().__init__(actor_critic, normalizer)
        self.history_length = int(getattr(actor_critic, "history_length", 10))

    def forward(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        single_batch = obs_history.dim() == 1
        if single_batch:
            obs_history = obs_history.unsqueeze(0)
        obs_history = self.normalizer(obs_history)
        parts = self.estimator(obs_history)
        vel, z = parts[..., :3], parts[..., 3 : 3 + self.num_latent]
        vel_est = vel
        vel = vel * self.vel_scale_for_actor
        z = F.normalize(z, dim=-1, p=2.0)
        n = int(self.actor_recent_frames)
        if n <= 0:
            recent = obs_history[:, 0:0]
        else:
            recent = obs_history[:, -(n * self.num_one_step_obs) :]
        actions = self.actor(torch.cat((recent, vel, z), dim=1))
        if single_batch:
            actions = actions.squeeze(0)
            vel_est = vel_est.squeeze(0)
        return actions, vel_est

    def export(self, path: str, filename: str = "policy.onnx"):
        os.makedirs(path, exist_ok=True)
        out_path = os.path.join(path, filename)
        self.to("cpu")
        self.eval()
        history_dim = int(self.num_one_step_obs * self.history_length)
        obs = torch.zeros(1, history_dim, dtype=torch.float32)
        torch.onnx.export(
            self,
            obs,
            out_path,
            export_params=True,
            opset_version=14,
            input_names=["obs"],
            output_names=["actions", "lin_vel_est"],
            dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}, "lin_vel_est": {0: "batch"}},
        )
        print(f"[exporter] HIM ONNX saved: {out_path}  (outputs: actions, lin_vel_est)")
