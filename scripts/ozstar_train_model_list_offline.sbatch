#!/bin/bash
#SBATCH --job-name=gomarl_list
#SBATCH --output=ozstar_logs/%x_%j.out
#SBATCH --error=ozstar_logs/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl}"
PYTHON_BIN="${PYTHON_BIN:-/home/kyang/.conda/envs/marl_cpu/bin/python}"
SC2PATH="${SC2PATH:-/home/kyang/StarCraftII}"

CONFIG="${CONFIG:-clean_hyper}"
ENV_CONFIG="${ENV_CONFIG:-sc2}"
MAP_NAME="${MAP_NAME:-5m_vs_6m}"
MODELS="${MODELS:-rpg_action_edge_public_pred_relation_private_single_head rpg_action_edge_public_pred_relation_private_decision_maker rpg_action_edge_graphormer_relation_private_single_head rpg_action_edge_graphit_relation_private_single_head rpg_action_edge_edgeset_relation_private_single_head rpg_action_edge_motif_transformer_relation_private_single_head}"
SEEDS="${SEEDS:-1}"
RUN_PREFIX="${RUN_PREFIX:-ozstar}"

USE_CUDA="${USE_CUDA:-False}"
WANDB_MODE="${WANDB_MODE:-offline}"
USE_WANDB="${USE_WANDB:-True}"

BATCH_SIZE_RUN="${BATCH_SIZE_RUN:-8}"
BATCH_SIZE="${BATCH_SIZE:-32}"
BUFFER_SIZE="${BUFFER_SIZE:-500}"
T_MAX="${T_MAX:-10050000}"
TEST_INTERVAL="${TEST_INTERVAL:-50000}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "$REPO_DIR"
mkdir -p ozstar_logs

export SC2PATH
export WANDB_MODE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-32}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-32}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "== GoMARL OzSTAR offline model-list job =="
echo "host: $(hostname)"
echo "job_id: ${SLURM_JOB_ID:-none}"
echo "repo: $REPO_DIR"
echo "python: $PYTHON_BIN"
echo "sc2path: $SC2PATH"
echo "config: $CONFIG"
echo "env_config: $ENV_CONFIG"
echo "map: $MAP_NAME"
echo "models: $MODELS"
echo "seeds: $SEEDS"
echo "run_prefix: $RUN_PREFIX"
echo "wandb_mode: $WANDB_MODE"
echo "use_cuda: $USE_CUDA"
echo "batch_size_run: $BATCH_SIZE_RUN"
echo "batch_size: $BATCH_SIZE"
echo "buffer_size: $BUFFER_SIZE"
echo "t_max: $T_MAX"
echo "test_interval: $TEST_INTERVAL"
echo "extra_args: $EXTRA_ARGS"

"$PYTHON_BIN" - <<'PY'
import os
import sys
print("python:", sys.executable)
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
except Exception as exc:
    print("torch_check_failed:", repr(exc))
try:
    import smac
    print("smac:", smac.__file__)
except Exception as exc:
    print("smac_check_failed:", repr(exc))
print("sc2_binary_exists:", os.path.exists(os.path.join(os.environ["SC2PATH"], "Versions", "Base75689", "SC2_x64")))
PY

for model_type in $MODELS; do
  for seed in $SEEDS; do
    run_name="${MAP_NAME}_${model_type}_${RUN_PREFIX}_s${seed}"
    echo "== Run start: map=$MAP_NAME model=$model_type seed=$seed run=$run_name =="
    "$PYTHON_BIN" src/main.py \
      --config="$CONFIG" \
      --env-config="$ENV_CONFIG" \
      with \
      env_args.map_name="$MAP_NAME" \
      clean_model_type="$model_type" \
      seed="$seed" \
      use_cuda="$USE_CUDA" \
      use_wandb="$USE_WANDB" \
      wandb_mode="$WANDB_MODE" \
      batch_size_run="$BATCH_SIZE_RUN" \
      batch_size="$BATCH_SIZE" \
      buffer_size="$BUFFER_SIZE" \
      t_max="$T_MAX" \
      test_interval="$TEST_INTERVAL" \
      save_battle_trace=False \
      name="$run_name" \
      wandb_run_name="$run_name" \
      $EXTRA_ARGS
    echo "== Run done: $run_name =="
  done
done
