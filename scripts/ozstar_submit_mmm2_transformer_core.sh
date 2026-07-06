#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl}"
SEEDS="${SEEDS:-1}"
GROUP_NAME="${GROUP_NAME:-mixed_3s5z_mmm2_transformer_nine_models_s1}"

CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
MEM="${MEM:-64G}"
TIME="${TIME:-48:00:00}"
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

echo "== Submit MMM2 Transformer core runs =="
echo "repo: $REPO_DIR"
echo "seeds: $SEEDS"
echo "group_name: $GROUP_NAME"
echo "time: $TIME"
echo "t_max: $T_MAX"
echo "extra_args: $EXTRA_ARGS"

job_idx=0
RUN_SPECS=(
  "3s5z_vs_3s6z rpg_public_private_bias_transformer_hypercond"
  "3s5z_vs_3s6z rpg_public_private_bias_transformer_topk_hypercond"
  "3s5z_vs_3s6z rpg_public_private_bias_transformer_threshold_hypercond"
  "3s5z_vs_3s6z rpg_global_public_private_bias_past_delta_token_transformer_hypercond"
  "3s5z_vs_3s6z rpg_global_public_private_bias_past_delta_token_transformer_topk_hypercond"
  "3s5z_vs_3s6z rpg_global_public_private_bias_past_delta_token_transformer_threshold_hypercond"
  "MMM2 rpg_public_transformer_hypercond"
  "MMM2 rpg_public_private_bias_transformer_hypercond"
  "MMM2 rpg_public_private_bias_past_delta_token_transformer_hypercond"
)

for run_spec in "${RUN_SPECS[@]}"; do
  read -r map_name model_type <<< "$run_spec"
  for seed in $SEEDS; do
    job_idx=$((job_idx + 1))
    run_name="${map_name}_${model_type}_ozstar_s${seed}"
    job_name="${map_name}_${model_type}_s${seed}"
    job_name="${job_name:0:60}"
    echo "submit[$job_idx]: map=$map_name model=$model_type seed=$seed run=$run_name"
    sbatch \
      --cpus-per-task="$CPUS_PER_TASK" \
      --mem="$MEM" \
      --time="$TIME" \
      --job-name="$job_name" \
      --output=ozstar_logs/%x_%j.out \
      --error=ozstar_logs/%x_%j.err \
      --export=ALL,MAP_NAME="$map_name",MODEL_TYPE="$model_type",SEED="$seed",T_MAX="$T_MAX",TEST_INTERVAL="$TEST_INTERVAL",BATCH_SIZE_RUN="$BATCH_SIZE_RUN",BATCH_SIZE="$BATCH_SIZE",BUFFER_SIZE="$BUFFER_SIZE",USE_WANDB="$USE_WANDB",WANDB_MODE="$WANDB_MODE",USE_CUDA="$USE_CUDA",RUN_NAME="$run_name",GROUP_NAME="$GROUP_NAME",EXTRA_ARGS="$EXTRA_ARGS" \
      scripts/ozstar_train_offline.sbatch
  done
done
