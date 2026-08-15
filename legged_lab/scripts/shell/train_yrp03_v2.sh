#!/usr/bin/env bash
# 正式训练 YRP03-v2 跑步任务 (yrp03_v2_run)
#
# 用法:
#   bash legged_lab/scripts/shell/train_yrp03_v2.sh
#   bash legged_lab/scripts/shell/train_yrp03_v2.sh --num_envs 1024 --max_iterations 5000
#   NUM_ENVS=2048 bash legged_lab/scripts/shell/train_yrp03_v2.sh   # 环境变量覆盖亦可
#
# 续训(在 motion_run_cfg.py 里把 resume=True 并填 load_run/load_checkpoint 后):
#   bash legged_lab/scripts/shell/train_yrp03_v2.sh --resume True --load_run <run> --checkpoint <model.pt>
#
# 多卡(2 卡, 需自行确认 GPU 数量):
#   torchrun --nproc_per_node=2 legged_lab/scripts/train.py --task=yrp03_v2_run --headless --distributed
set -euo pipefail

# ── Isaac Lab 环境 ────────────────────────────────────────
CONDA_ENV="${CONDA_ENV:-calt_run}"                                    # Isaac Lab 5.1 / isaaclab 0.54.2
PYTHON="${PYTHON:-/home/robot02/miniconda3/envs/${CONDA_ENV}/bin/python}"

# ── 仓库根目录(相对本脚本位置向上三级) ─────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# ── 本地 rsl_rl 优先于 site-packages 里可能安装的其它 rsl_rl ──
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/rsl_rl:${PYTHONPATH:-}"

# ── 训练参数(可用环境变量或末尾 "$@" 覆盖, argparse 后者覆盖前者) ──
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-20000}"
SEED="${SEED:-42}"

echo "[train_yrp03_v2] task=yrp03_v2_run num_envs=${NUM_ENVS} max_iterations=${MAX_ITERATIONS} seed=${SEED}"
echo "[train_yrp03_v2] python=${PYTHON}"
echo "[train_yrp03_v2] logdir=${REPO_ROOT}/logs/yrp03_v2_run/"

exec "${PYTHON}" legged_lab/scripts/train.py \
  --task=yrp03_v2_run \
  --num_envs="${NUM_ENVS}" \
  --max_iterations="${MAX_ITERATIONS}" \
  --seed="${SEED}" \
  --headless \
  --logger=tensorboard \
  "$@"
