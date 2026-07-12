#!/bin/bash
set -euo pipefail

# Reproduce the pre-d3cd604 CPU experiment setting used by the earlier SMAC
# ablations. Keep all six stability variants on this exact setting.

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl}"
MAP_NAME="${MAP_NAME:-3s5z_vs_3s6z}"
SEEDS="${SEEDS:-1}"
MODELS="${MODELS:-rpg_public_private_simple_bias_transformer_q_residual_hypercond rpg_public_private_simple_bias_transformer_param_residual_hypercond rpg_public_private_simple_bias_transformer_smooth_hypercond rpg_public_private_simple_bias_transformer_param_residual_l2_hypercond rpg_public_private_simple_bias_transformer_param_residual_smooth_hypercond rpg_public_private_simple_bias_transformer_param_residual_l2_smooth_hypercond}"

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

echo "== Submit legacy-setting 3s5z stability ablation =="
echo "models: $MODELS"
echo "seeds: $SEEDS"
echo "cpus_per_task: $CPUS_PER_TASK"
echo "mem: $MEM"
echo "time: $TIME"
echo "batch_size_run: $BATCH_SIZE_RUN"
echo "batch_size: $BATCH_SIZE"
echo "buffer_size: $BUFFER_SIZE"
echo "t_max: $T_MAX"
echo "test_interval: $TEST_INTERVAL"

for model_type in $MODELS; do
  for seed in $SEEDS; do
    run_name="${MAP_NAME}_${model_type}_legacycfg_s${seed}"
    job_name="3s5z_legacy_${model_type#rpg_public_private_simple_bias_transformer_}_s${seed}"
    job_name="${job_name:0:60}"

    echo "submit: model=$model_type seed=$seed run=$run_name"
    sbatch \
      --cpus-per-task="$CPUS_PER_TASK" \
      --mem="$MEM" \
      --time="$TIME" \
      --job-name="$job_name" \
      --output=ozstar_logs/%x_%j.out \
      --error=ozstar_logs/%x_%j.err \
      --export=ALL,MAP_NAME="$MAP_NAME",MODEL_TYPE="$model_type",SEED="$seed",T_MAX="$T_MAX",TEST_INTERVAL="$TEST_INTERVAL",BATCH_SIZE_RUN="$BATCH_SIZE_RUN",BATCH_SIZE="$BATCH_SIZE",BUFFER_SIZE="$BUFFER_SIZE",USE_WANDB="$USE_WANDB",WANDB_MODE="$WANDB_MODE",USE_CUDA="$USE_CUDA",RUN_NAME="$run_name",EXTRA_ARGS="$EXTRA_ARGS" \
      scripts/ozstar_train_offline.sbatch
  done
done
