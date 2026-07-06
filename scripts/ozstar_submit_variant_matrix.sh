#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl}"
MAPS="${MAPS:-5m_vs_6m corridor}"
MODELS="${MODELS:-rpg_semantic_selfattn_relation_hypercond rpg_entity_selfattn_relation_hypercond rpg_delta_relation_hypercond rpg_relation_coarse_self_fine_head rpg_relation_prototype_single_head}"
SEEDS="${SEEDS:-1}"

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

echo "== GoMARL OzSTAR variant matrix submit =="
echo "repo: $REPO_DIR"
echo "maps: $MAPS"
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
echo "wandb_mode: $WANDB_MODE"
echo "extra_args: $EXTRA_ARGS"

for map_name in $MAPS; do
  for model_type in $MODELS; do
    for seed in $SEEDS; do
      run_name="${map_name}_${model_type}_ozstar_s${seed}"
      job_name="${map_name}_${model_type}_s${seed}"
      job_name="${job_name:0:60}"
      echo "submit: map=$map_name model=$model_type seed=$seed run=$run_name"
      sbatch \
        --cpus-per-task="$CPUS_PER_TASK" \
        --mem="$MEM" \
        --time="$TIME" \
        --job-name="$job_name" \
        --output=ozstar_logs/%x_%j.out \
        --error=ozstar_logs/%x_%j.err \
        --export=ALL,MAP_NAME="$map_name",MODEL_TYPE="$model_type",SEED="$seed",T_MAX="$T_MAX",TEST_INTERVAL="$TEST_INTERVAL",BATCH_SIZE_RUN="$BATCH_SIZE_RUN",BATCH_SIZE="$BATCH_SIZE",BUFFER_SIZE="$BUFFER_SIZE",USE_WANDB="$USE_WANDB",WANDB_MODE="$WANDB_MODE",USE_CUDA="$USE_CUDA",RUN_NAME="$run_name",EXTRA_ARGS="$EXTRA_ARGS" \
        scripts/ozstar_train_offline.sbatch
    done
  done
done
