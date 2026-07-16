#!/bin/bash
#SBATCH --job-name=gomarl
#SBATCH --output=ozstar_logs/%x_%j.out
#SBATCH --error=ozstar_logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl}"
PYTHON_BIN="${PYTHON_BIN:-/home/kyang/.conda/envs/marl_cpu/bin/python}"
SC2PATH="${SC2PATH:-/home/kyang/StarCraftII}"

CONFIG="${CONFIG:-clean_hyper}"
ENV_CONFIG="${ENV_CONFIG:-sc2}"
MAP_NAME="${MAP_NAME:-}"
if [[ "$ENV_CONFIG" == "sc2" && -z "$MAP_NAME" ]]; then
  MAP_NAME="3m"
fi
MODEL_TYPE="${MODEL_TYPE:-qmix_minimal}"
SEED="${SEED:-1}"
RUN_KEY="${MAP_NAME:-$ENV_CONFIG}"
RUN_NAME="${RUN_NAME:-ozstar_${RUN_KEY}_${MODEL_TYPE}_s${SEED}}"

USE_CUDA="${USE_CUDA:-False}"
WANDB_MODE="${WANDB_MODE:-offline}"
USE_WANDB="${USE_WANDB:-True}"

BATCH_SIZE_RUN="${BATCH_SIZE_RUN:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
BUFFER_SIZE="${BUFFER_SIZE:-50}"
T_MAX="${T_MAX:-10000}"
TEST_INTERVAL="${TEST_INTERVAL:-5000}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "$REPO_DIR"
mkdir -p ozstar_logs

export SC2PATH
export WANDB_MODE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "== GoMARL OzSTAR offline job =="
echo "host: $(hostname)"
echo "job_id: ${SLURM_JOB_ID:-none}"
echo "repo: $REPO_DIR"
echo "python: $PYTHON_BIN"
echo "sc2path: $SC2PATH"
echo "config: $CONFIG"
echo "env_config: $ENV_CONFIG"
echo "map: ${MAP_NAME:-<env-config default>}"
echo "model: $MODEL_TYPE"
echo "seed: $SEED"
echo "run_name: $RUN_NAME"
echo "wandb_mode: $WANDB_MODE"
echo "use_cuda: $USE_CUDA"
echo "extra_args: $EXTRA_ARGS"

"$PYTHON_BIN" - <<'PY'
import os
import sys
print("python:", sys.executable)
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    print("torch_num_threads_preflight:", torch.get_num_threads())
    print("torch_num_interop_threads_preflight:", torch.get_num_interop_threads())
    print("slurm_cpus_per_task:", os.environ.get("SLURM_CPUS_PER_TASK"))
except Exception as exc:
    print("torch_check_failed:", repr(exc))
try:
    import smac
    print("smac:", smac.__file__)
except Exception as exc:
    print("smac_check_failed:", repr(exc))
try:
    import gfootball
    print("gfootball:", gfootball.__file__)
except Exception as exc:
    print("gfootball_check_failed:", repr(exc))
print("sc2_binary_exists:", os.path.exists(os.path.join(os.environ["SC2PATH"], "Versions", "Base75689", "SC2_x64")))
PY

MAP_ARGS=()
if [[ -n "$MAP_NAME" ]]; then
  MAP_ARGS+=(env_args.map_name="$MAP_NAME")
fi

MODEL_ARGS=()
if [[ "$CONFIG" == clean_hyper* ]]; then
  MODEL_ARGS+=(clean_model_type="$MODEL_TYPE")
fi

"$PYTHON_BIN" src/main.py \
  --config="$CONFIG" \
  --env-config="$ENV_CONFIG" \
  with \
  "${MAP_ARGS[@]}" \
  "${MODEL_ARGS[@]}" \
  seed="$SEED" \
  use_cuda="$USE_CUDA" \
  use_wandb="$USE_WANDB" \
  wandb_mode="$WANDB_MODE" \
  batch_size_run="$BATCH_SIZE_RUN" \
  batch_size="$BATCH_SIZE" \
  buffer_size="$BUFFER_SIZE" \
  t_max="$T_MAX" \
  test_interval="$TEST_INTERVAL" \
  save_battle_trace=False \
  name="${GROUP_NAME:-$RUN_NAME}" \
  wandb_run_name="$RUN_NAME" \
  $EXTRA_ARGS
