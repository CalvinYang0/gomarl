#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl}"
MAP_NAME="${MAP_NAME:-3s5z_vs_3s6z}"
SEEDS="${SEEDS:-1}"
MODELS="${MODELS:-rpg_public_private_full_token_transformer_relation_token_head_hypercond rpg_full_obs_transformer_relation_token_head_hypercond}"
GROUP_NAME="${GROUP_NAME:-3s5z_private_raw_transformer_s1}"

CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
MEM="${MEM:-64G}"
TIME="${TIME:-24:00:00}"
BATCH_SIZE_RUN="${BATCH_SIZE_RUN:-8}"
BATCH_SIZE="${BATCH_SIZE:-128}"
BUFFER_SIZE="${BUFFER_SIZE:-5000}"
T_MAX="${T_MAX:-10050000}"
TEST_INTERVAL="${TEST_INTERVAL:-10000}"
USE_WANDB="${USE_WANDB:-True}"
WANDB_MODE="${WANDB_MODE:-offline}"
USE_CUDA="${USE_CUDA:-False}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "$REPO_DIR"
mkdir -p ozstar_logs

echo "== Submit 3s5z private/raw Transformer controls =="
echo "repo: $REPO_DIR"
echo "map: $MAP_NAME"
echo "models: $MODELS"
echo "seeds: $SEEDS"
echo "group_name: $GROUP_NAME"

for model_type in $MODELS; do
  for seed in $SEEDS; do
    run_name="${MAP_NAME}_${model_type}_ozstar_s${seed}"
    job_name="${MAP_NAME}_${model_type}_s${seed}"
    job_name="${job_name:0:60}"
    echo "submit: model=$model_type seed=$seed run=$run_name"
    sbatch \
      --cpus-per-task="$CPUS_PER_TASK" \
      --mem="$MEM" \
      --time="$TIME" \
      --job-name="$job_name" \
      --output=ozstar_logs/%x_%j.out \
      --error=ozstar_logs/%x_%j.err \
      --export=ALL,MAP_NAME="$MAP_NAME",MODEL_TYPE="$model_type",SEED="$seed",T_MAX="$T_MAX",TEST_INTERVAL="$TEST_INTERVAL",BATCH_SIZE_RUN="$BATCH_SIZE_RUN",BATCH_SIZE="$BATCH_SIZE",BUFFER_SIZE="$BUFFER_SIZE",USE_WANDB="$USE_WANDB",WANDB_MODE="$WANDB_MODE",USE_CUDA="$USE_CUDA",RUN_NAME="$run_name",GROUP_NAME="$GROUP_NAME",EXTRA_ARGS="$EXTRA_ARGS" \
      scripts/ozstar_train_offline.sbatch
  done
done
