#!/usr/bin/env python3
"""Extract time, vx, and left-side joint velocity/torque from play/sim2sim joint-state CSV.

Joint order: left leg → waist → left arm (same as plot_sim2sim_ultra_yaw_csv DEFAULT_PLOT_JOINTS).

Input: CSV from play.py --log-joints or sim2sim_ultraYaw.py --log-torques
Output: slim CSV — col 1–2: time, vx; col 3–9: qd (7 joints); col 10–16: tau (7 joints).

Usage:
  python legged_lab/scripts/extract_left_joint_log_csv.py logs/play_joint_state.csv
  python legged_lab/scripts/extract_left_joint_log_csv.py logs/play_joint_state.csv -o logs/left_joints.csv
  python legged_lab/scripts/extract_left_joint_log_csv.py logs/play_joint_state.csv --vx-col cmd_vx
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Allow running as script without installing package
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent.parent))

from legged_lab.scripts.plot_sim2sim_ultra_yaw_csv import load_sim2sim_csv

# Left leg → waist → left arm (aligned with plot_sim2sim_ultra_yaw_csv.py)
DEFAULT_JOINT_ORDER: tuple[str, ...] = (
    "hip_yaw_l_joint",
    "hip_roll_l_joint",
    "hip_pitch_l_joint",
    "knee_pitch_l_joint",
    "ankle_pitch_l_joint",
    "waist_yaw_joint",
    "shoulder_pitch_l_joint",
)


def resolve_joint_order(
    available: list[str],
    include_waist: bool = True,
    include_arm: bool = True,
) -> list[str]:
    """Joints in order: left leg, waist, left arm."""
    names = list(DEFAULT_JOINT_ORDER)
    if not include_waist:
        names = [j for j in names if j != "waist_yaw_joint"]
    if not include_arm:
        names = [j for j in names if "shoulder" not in j]
    ordered = [j for j in names if j in available]
    if ordered:
        return ordered
    # Fallback: all left joints then waist
    left = sorted(j for j in available if "_l_joint" in j)
    extra = [j for j in ("waist_yaw_joint",) if include_waist and j in available]
    return left + extra


def build_output_header(left_joints: list[str]) -> list[str]:
    """Columns 1–2: time, vx; 3–(2+N): qd; (3+N)–(2+2N): tau."""
    header = ["time", "vx"]
    header.extend(f"qd_{jname}" for jname in left_joints)
    header.extend(f"tau_{jname}" for jname in left_joints)
    return header


def extract_to_csv(
    input_path: Path,
    output_path: Path,
    vx_col: str = "base_vx",
    include_waist: bool = True,
    include_arm: bool = True,
) -> tuple[int, list[str]]:
    log = load_sim2sim_csv(input_path)
    if log.base is None or vx_col not in log.base:
        choices = "base_vx, cmd_vx" if log.base else "(no base columns in file)"
        raise ValueError(f"Column '{vx_col}' not in {input_path}. Available: {choices}")

    joint_names = resolve_joint_order(
        list(log.joints.keys()),
        include_waist=include_waist,
        include_arm=include_arm,
    )
    if not joint_names:
        raise ValueError(f"No matching joint columns in {input_path}")

    vx = log.base[vx_col]
    n = len(log.t)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(build_output_header(joint_names))
        for i in range(n):
            row: list[float] = [float(log.t[i]), float(vx[i])]
            row.extend(float(log.joints[jname]["qd"][i]) for jname in joint_names)
            row.extend(float(log.joints[jname]["tau"][i]) for jname in joint_names)
            writer.writerow(row)

    return n, joint_names


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_csv",
        type=str,
        help="Source CSV (play --log-joints or sim2sim --log-torques)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: <input_stem>_left_joints.csv beside input)",
    )
    parser.add_argument(
        "--vx-col",
        type=str,
        default="base_vx",
        choices=("base_vx", "cmd_vx"),
        help="Forward velocity column (default: base_vx in yaw frame)",
    )
    parser.add_argument(
        "--no-waist",
        action="store_true",
        help="Exclude waist_yaw_joint (left leg + arm only)",
    )
    parser.add_argument(
        "--no-arm",
        action="store_true",
        help="Exclude shoulder_pitch_l_joint (left leg + waist only)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv).expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"[ERROR] Input not found: {input_path}")

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = input_path.with_name(f"{input_path.stem}_left_joints.csv")

    n, joint_names = extract_to_csv(
        input_path,
        output_path,
        vx_col=args.vx_col,
        include_waist=not args.no_waist,
        include_arm=not args.no_arm,
    )
    print(f"[INFO] Read {input_path}")
    print(f"[INFO] Wrote {n} rows, {len(joint_names)} joints, vx={args.vx_col}")
    print(f"       order: {' -> '.join(joint_names)}")
    print(f"       -> {output_path}")


if __name__ == "__main__":
    main()
