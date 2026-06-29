#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl}"
MAPS="${MAPS:-3s5z_vs_3s6z MMM2 5m_vs_6m}"
SEEDS="${SEEDS:-1}"
MODELS="${MODELS:-rpg_public_private_bias_transformer_hypercond rpg_public_private_bias_transformer_relation_token_head_hypercond rpg_public_private_bias_transformer_relation_pair_token_head_hypercond rpg_public_private_bias_transformer_relation_private_token_head_hypercond}"
GROUP_NAME="${GROUP_NAME:-relation_head_ablation_3maps_s1}"

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

echo "== Submit relation-head ablation on 3 maps =="
echo "repo: $REPO_DIR"
echo "maps: $MAPS"
echo "models: $MODELS"
echo "seeds: $SEEDS"
echo "group_name: $GROUP_NAME"
echo "time: $TIME"
echo "t_max: $T_MAX"
echo "extra_args: $EXTRA_ARGS"

job_idx=0
for map_name in $MAPS; do
  for model_type in $MODELS; do
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
done
