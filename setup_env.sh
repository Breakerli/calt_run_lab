#!/usr/bin/env bash
#
# ---------------------------------------------------------------------------
# 【请务必 source】不要用 sh：否则 PYTHONPATH 只在子 shell 里生效，当前终端无效。
# IMPORTANT: Run this file with SOURCE, not sh/bash as a subprocess:
#   source setup_env.sh        # ok — PYTHONPATH applies to your shell
#   . ./setup_env.sh           # same
#
# Do NOT run:
#   sh setup_env.sh            # wrong — runs in subshell; exports are lost
# ---------------------------------------------------------------------------
#
# Typical workflow:
#   conda activate env_isaaclab   # or your Isaac Lab env (do this first)
#   cd /path/to/ultra_run_lab
#   source setup_env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/rsl_rl:${PYTHONPATH:-}"
