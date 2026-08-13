#!/usr/bin/env bash
# Convert Ultra URDF to USD via Isaac Lab (same workflow as ultra_0513Yaw).
#
# Usage:
#   ./legged_lab/scripts/convert_ultra_urdf_to_usd.sh GameYaw
# Optional override:
#   export ISAACLAB_PATH=/path/to/IsaacLab
#
# Requires: Isaac Sim + Isaac Lab 2.1 environment (./isaaclab.sh).

set -euo pipefail

ROBOT_TAG="${1:-GameYaw}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ASSET_DIR="${REPO_ROOT}/legged_lab/assets/ultra_${ROBOT_TAG}"
URDF_PATH="${ASSET_DIR}/urdf/urdf${ROBOT_TAG}.urdf"
if [[ ! -f "${URDF_PATH}" ]]; then
  # GameYaw 等机器人 URDF 可能已改名（如 ultra0523Yaw.urdf）
  URDF_FALLBACK="$(find "${ASSET_DIR}/urdf" -maxdepth 1 -name '*.urdf' -type f 2>/dev/null | sort | head -1)"
  if [[ -n "${URDF_FALLBACK}" ]]; then
    URDF_PATH="${URDF_FALLBACK}"
  fi
fi
# e.g. GameYaw -> ultra_game_yaw.usd, 0513Yaw -> ultra_0513yaw.usd
USD_BASENAME="ultra_$(echo "${ROBOT_TAG}" | sed -E 's/([a-z])([A-Z])/\1_\2/g' | tr '[:upper:]' '[:lower:]')"
USD_PATH="${ASSET_DIR}/usd/${USD_BASENAME}.usd"

_isaaclab_ok() {
  [[ -n "${1:-}" && -f "${1}/isaaclab.sh" ]]
}

if ! _isaaclab_ok "${ISAACLAB_PATH:-}"; then
  ISAACLAB_PATH=""
  for candidate in \
    "/home/hric-pc/Work/train/IsaacLab" \
    "${HOME}/Work/train/IsaacLab" \
    "${HOME}/IsaacLab"; do
    if _isaaclab_ok "${candidate}"; then
      ISAACLAB_PATH="${candidate}"
      break
    fi
  done
fi

if ! _isaaclab_ok "${ISAACLAB_PATH:-}"; then
  echo "ERROR: Isaac Lab not found (need isaaclab.sh in repo root)."
  echo "  unset ISAACLAB_PATH   # if you set a wrong path earlier"
  echo "  export ISAACLAB_PATH=/home/hric-pc/Work/train/IsaacLab"
  exit 1
fi

echo "Using ISAACLAB_PATH=${ISAACLAB_PATH}"

if [[ ! -f "${URDF_PATH}" ]]; then
  echo "ERROR: URDF not found: ${URDF_PATH}"
  exit 1
fi

mkdir -p "$(dirname "${USD_PATH}")"

cd "${ISAACLAB_PATH}"
./isaaclab.sh -p scripts/tools/convert_urdf.py \
  "${URDF_PATH}" \
  "${USD_PATH}" \
  --headless \
  --joint-stiffness 100.0 \
  --joint-damping 1.0

echo "Done. USD: ${USD_PATH}"
echo "config.yaml should be under: $(dirname "${USD_PATH}")"
