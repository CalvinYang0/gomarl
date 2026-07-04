#!/bin/bash
set -euo pipefail

SEEDS="${SEEDS:-1}"
ENVS="${ENVS:-academy_pass_and_shoot_with_keeper academy_3_vs_1_with_keeper academy_counterattack_easy}"
CONFIG="${CONFIG:-group}"
MODEL_TYPE="${MODEL_TYPE:-group}"
T_MAX="${T_MAX:-10050000}"
TEST_INTERVAL="${TEST_INTERVAL:-50000}"
BATCH_SIZE_RUN="${BATCH_SIZE_RUN:-8}"
BATCH_SIZE="${BATCH_SIZE:-32}"
BUFFER_SIZE="${BUFFER_SIZE:-500}"
TIME="${TIME:-2-00:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
MEM="${MEM:-64G}"
USE_WANDB="${USE_WANDB:-True}"
WANDB_MODE="${WANDB_MODE:-offline}"
RUN_PREFIX="${RUN_PREFIX:-grf_gomarl}"

mkdir -p ozstar_logs

echo "== GoMARL OzSTAR GRF submit =="
echo "envs: $ENVS"
echo "config: $CONFIG"
echo "model: $MODEL_TYPE"
echo "seeds: $SEEDS"
echo "t_max: $T_MAX"
echo "time: $TIME"

for env_config in $ENVS; do
  for seed in $SEEDS; do
    run_name="${RUN_PREFIX}_${env_config}_${MODEL_TYPE}_10m_s${seed}"
    job_name="grf_${env_config}_${MODEL_TYPE}_s${seed}"
    echo "submit env=${env_config} model=${MODEL_TYPE} seed=${seed}"
    sbatch \
      --cpus-per-task="$CPUS_PER_TASK" \
      --mem="$MEM" \
      --time="$TIME" \
      --job-name="$job_name" \
      --output=ozstar_logs/%x_%j.out \
      --error=ozstar_logs/%x_%j.err \
      --export=ALL,CONFIG="$CONFIG",ENV_CONFIG="$env_config",MAP_NAME="$env_config",MODEL_TYPE="$MODEL_TYPE",SEED="$seed",T_MAX="$T_MAX",TEST_INTERVAL="$TEST_INTERVAL",BATCH_SIZE_RUN="$BATCH_SIZE_RUN",BATCH_SIZE="$BATCH_SIZE",BUFFER_SIZE="$BUFFER_SIZE",USE_WANDB="$USE_WANDB",WANDB_MODE="$WANDB_MODE",RUN_NAME="$run_name",GROUP_NAME="$run_name" \
      scripts/ozstar_train_offline.sbatch
  done
done
