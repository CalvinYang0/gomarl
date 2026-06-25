#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl}"
MAP_NAME="${MAP_NAME:-MMM2}"
SEEDS="${SEEDS:-1}"
MODELS="${MODELS:-rpg_public_transformer_hypercond rpg_public_private_bias_transformer_hypercond rpg_public_private_bias_past_delta_token_transformer_hypercond rpg_public_private_bias_transformer_topk_hypercond rpg_public_private_bias_transformer_threshold_hypercond rpg_global_public_transformer_hypercond rpg_global_public_private_bias_past_delta_token_transformer_hypercond rpg_global_public_private_bias_past_delta_token_transformer_topk_hypercond rpg_global_public_private_bias_past_delta_token_transformer_threshold_hypercond}"
GROUP_NAME="${GROUP_NAME:-mmm2_transformer_nine_models_s1}"

CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
MEM="${MEM:-64G}"
TIME="${TIME:-48:00:00}"
BATCH_SIZE_RUN="${BATCH_SIZE_RUN:-8}"
BATCH_SIZE="${BATCH_SIZE:-32}"
BUFFER_SIZE="${BUFFER_SIZE:-500}"
T_MAX="${T_MAX:-10050000}"
TEST_INTERVAL="${TEST_INTERVAL:-50000}"
USE_WANDB="${USE_WANDB:-True}"
WANDB_MODE="${WANDB_MODE:-offline}"
USE_CUDA="${USE_CUDA:-False}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "$REPO_DIR"
mkdir -p ozstar_logs

echo "== Submit MMM2 Transformer core runs =="
echo "repo: $REPO_DIR"
echo "map: $MAP_NAME"
echo "models: $MODELS"
echo "seeds: $SEEDS"
echo "group_name: $GROUP_NAME"
echo "time: $TIME"
echo "t_max: $T_MAX"
echo "extra_args: $EXTRA_ARGS"

job_idx=0
for model_type in $MODELS; do
  for seed in $SEEDS; do
    job_idx=$((job_idx + 1))
    run_name="${MAP_NAME}_${model_type}_ozstar_s${seed}"
    job_name="${MAP_NAME}_${model_type}_s${seed}"
    job_name="${job_name:0:60}"
    echo "submit[$job_idx]: model=$model_type seed=$seed run=$run_name"
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
