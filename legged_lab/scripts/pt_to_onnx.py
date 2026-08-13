from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


LEGGED_LAB_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

DEFAULT_POLICY = os.path.join(LEGGED_LAB_ROOT_DIR, "Exported_policy", "policy.pt")
DEFAULT_ONNX = os.path.join(LEGGED_LAB_ROOT_DIR, "Exported_policy", "policy.onnx")

# ultra_game_yaw deploy obs: ang_vel(3) + gravity(3) + command(3) + q/qd/action(13*3).
DEFAULT_ACTION_DIM = 13
DEFAULT_OBS_PER_STEP = 9 + DEFAULT_ACTION_DIM * 3
DEFAULT_HISTORY_LENGTH = 10
DEFAULT_INPUT_DIM = DEFAULT_OBS_PER_STEP * DEFAULT_HISTORY_LENGTH

# Policy output order follows IsaacLab robot.data.joint_names / exported actor actions.
ISAAC_POLICY_JOINT_NAMES = [
    "hip_yaw_l_joint",
    "hip_yaw_r_joint",
    "waist_yaw_joint",
    "hip_roll_l_joint",
    "hip_roll_r_joint",
    "shoulder_pitch_l_joint",
    "shoulder_pitch_r_joint",
    "hip_pitch_l_joint",
    "hip_pitch_r_joint",
    "knee_pitch_l_joint",
    "knee_pitch_r_joint",
    "ankle_pitch_l_joint",
    "ankle_pitch_r_joint",
]


def resolve_input_dim(input_dim: int | None, obs_per_step: int, history_length: int) -> int:
    computed = int(obs_per_step) * int(history_length)
    if input_dim is None:
        return computed
    if int(input_dim) != computed:
        raise ValueError(
            f"input_dim={input_dim} does not match obs_per_step*history_length="
            f"{obs_per_step}*{history_length}={computed}."
        )
    return int(input_dim)


def check_action_dim(actions: torch.Tensor, expected_action_dim: int) -> None:
    if actions.dim() != 2:
        raise ValueError(f"Expected policy output rank 2 [batch, action], got shape {tuple(actions.shape)}.")
    actual = int(actions.shape[1])
    if actual != int(expected_action_dim):
        raise ValueError(f"Expected action dim {expected_action_dim}, got {actual}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert exported TorchScript policy.pt to ONNX with ultra_game_yaw deploy dimensions."
    )
    parser.add_argument("--policy", type=str, default=DEFAULT_POLICY, help="Path to TorchScript policy.pt.")
    parser.add_argument("--output", type=str, default=DEFAULT_ONNX, help="Output ONNX path.")
    parser.add_argument("--obs-per-step", type=int, default=DEFAULT_OBS_PER_STEP, help="Single-step actor obs dim.")
    parser.add_argument("--history-length", type=int, default=DEFAULT_HISTORY_LENGTH, help="Actor obs history length.")
    parser.add_argument(
        "--input-dim",
        type=int,
        default=DEFAULT_INPUT_DIM,
        help="Flattened policy input dim. Must equal --obs-per-step * --history-length.",
    )
    parser.add_argument("--action-dim", type=int, default=DEFAULT_ACTION_DIM, help="Expected policy action dim.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument(
        "--fixed-batch",
        action="store_true",
        help="Export with fixed batch size 1 instead of a dynamic batch axis.",
    )
    return parser


def export_policy_to_onnx(args: argparse.Namespace) -> Path:
    policy_path = Path(args.policy).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    input_dim = resolve_input_dim(args.input_dim, args.obs_per_step, args.history_length)

    if not policy_path.is_file():
        raise FileNotFoundError(f"policy.pt not found: {policy_path}")
    if int(args.action_dim) != len(ISAAC_POLICY_JOINT_NAMES):
        raise ValueError(
            f"action_dim={args.action_dim} does not match known policy joint count "
            f"{len(ISAAC_POLICY_JOINT_NAMES)}."
        )

    policy = torch.jit.load(str(policy_path), map_location="cpu").eval()
    dummy_obs = torch.zeros(1, input_dim, dtype=torch.float32)

    with torch.no_grad():
        actions = policy(dummy_obs)
    check_action_dim(actions, args.action_dim)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = None
    if not args.fixed_batch:
        dynamic_axes = {"obs_history": {0: "batch"}, "actions": {0: "batch"}}

    torch.onnx.export(
        policy,
        dummy_obs,
        str(output_path),
        input_names=["obs_history"],
        output_names=["actions"],
        dynamic_axes=dynamic_axes,
        opset_version=int(args.opset),
    )

    try:
        import onnx

        model = onnx.load(str(output_path))
        onnx.checker.check_model(model)
    except ImportError:
        print("[WARN] onnx package is not installed; skipped onnx.checker validation.", file=sys.stderr)

    print(f"[pt_to_onnx] policy: {policy_path}")
    print(f"[pt_to_onnx] onnx:   {output_path}")
    print(f"[pt_to_onnx] input:  obs_history [batch, {input_dim}]")
    print(f"[pt_to_onnx] output: actions [batch, {args.action_dim}]")
    print("[pt_to_onnx] action order:")
    for i, name in enumerate(ISAAC_POLICY_JOINT_NAMES):
        print(f"  {i:02d}: {name}")
    return output_path


def main() -> int:
    args = build_parser().parse_args()
    try:
        export_policy_to_onnx(args)
    except Exception as exc:
        print(f"[pt_to_onnx][ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
