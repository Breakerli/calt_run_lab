# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Policy inference for MuJoCo sim2sim (ONNX default, TorchScript fallback)."""

from __future__ import annotations

import os

import numpy as np


class Sim2SimPolicy:
    """Load HIM policy from ONNX (preferred) or TorchScript."""

    def __init__(self, policy_path: str, expected_obs_dim: int | None = None):
        self.policy_path = os.path.abspath(policy_path)
        self.expected_obs_dim = expected_obs_dim
        if policy_path.endswith(".onnx"):
            self._init_onnx(policy_path)
        elif policy_path.endswith(".pt"):
            self._init_jit(policy_path)
        else:
            raise ValueError(f"Unsupported policy extension: {policy_path}  (use .onnx or .pt)")

    def _init_onnx(self, path: str) -> None:
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required for ONNX sim2sim. Install with: pip install onnxruntime"
            ) from e
        self.backend = "onnx"
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.has_est_vel = "lin_vel_est" in self.output_names
        inp_shape = self.session.get_inputs()[0].shape
        onnx_obs_dim = inp_shape[1] if len(inp_shape) >= 2 and isinstance(inp_shape[1], int) else None
        print(f"[INFO] ONNX policy: inputs={[i.name for i in self.session.get_inputs()]}, "
              f"outputs={self.output_names}")
        if onnx_obs_dim is not None and self.expected_obs_dim is not None and onnx_obs_dim != self.expected_obs_dim:
            pt_fallback = os.path.join(os.path.dirname(path), "policy.pt")
            hint = (
                f"policy.onnx expects obs dim {onnx_obs_dim}, "
                f"but sim2sim uses {self.expected_obs_dim} "
                f"({self.expected_obs_dim // 48 if self.expected_obs_dim % 48 == 0 else '?'} x 48 history). "
                "Align SimToSimCfg.sim.actor_obs_history_length with motion_run_cfg / play export, "
                "or re-run play.py to regenerate policy.onnx. "
            )
            if os.path.isfile(pt_fallback):
                hint += f"Alternatively: --policy {pt_fallback}"
            raise ValueError(hint)
        if onnx_obs_dim is not None and not self.has_est_vel:
            print(
                "[WARN] ONNX has no lin_vel_est output; likely an old actor-only export. "
                "Re-export with play.py for HIM (outputs: actions, lin_vel_est)."
            )

    def _init_jit(self, path: str) -> None:
        import torch

        self.backend = "jit"
        self.model = torch.jit.load(path, map_location="cpu")
        self.model.eval()
        self.has_est_vel = hasattr(self.model, "estimate_lin_vel")
        print("[INFO] TorchScript policy loaded (consider policy.onnx from play.py for ONNX sim2sim).")
        if not self.has_est_vel:
            print("[WARN] policy.pt has no estimate_lin_vel(); LCM est_vel will be zero.")

    def infer(self, obs_history: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Run policy on flat obs history. Returns (actions [num_action], est_lin_vel [3] or None)."""
        if self.backend == "onnx":
            obs = np.asarray(obs_history, dtype=np.float32).reshape(1, -1)
            outputs = self.session.run(self.output_names, {self.input_name: obs})
            actions = np.asarray(outputs[0], dtype=np.float64).reshape(-1)
            est_vel = None
            if self.has_est_vel and len(outputs) > 1:
                est_vel = np.asarray(outputs[1], dtype=np.float64).reshape(-1)[:3]
            return actions, est_vel

        import torch

        with torch.inference_mode():
            obs_t = torch.as_tensor(obs_history, dtype=torch.float32)
            out = self.model(obs_t).detach().cpu().numpy().reshape(-1)
        est_vel = None
        if self.has_est_vel:
            with torch.inference_mode():
                vel = self.model.estimate_lin_vel(torch.as_tensor(obs_history, dtype=torch.float32))
            est_vel = vel.reshape(-1)[:3].cpu().numpy()
        return out, est_vel
