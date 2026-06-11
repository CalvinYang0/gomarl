#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl}"
MAPS="${MAPS:-5m_vs_6m corridor}"
MODELS="${MODELS:-rpg_action_edge_public_pred_relation_private_single_head rpg_action_edge_public_pred_relation_private_decision_maker rpg_action_edge_graphormer_relation_private_single_head rpg_action_edge_graphit_relation_private_single_head rpg_action_edge_edgeset_relation_private_single_head rpg_action_edge_motif_transformer_relation_private_single_head}"
SEEDS="${SEEDS:-1}"
RUN_PREFIX="${RUN_PREFIX:-ozstar}"

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

echo "== GoMARL OzSTAR action-edge structure Transformer submit =="
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
  job_name="${map_name}_action_edge_struct6"
  job_name="${job_name:0:60}"
  echo "submit list job: map=$map_name models=6"
  sbatch \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --time="$TIME" \
    --job-name="$job_name" \
    --output=ozstar_logs/%x_%j.out \
    --error=ozstar_logs/%x_%j.err \
    --export=ALL,MAP_NAME="$map_name",MODELS="$MODELS",SEEDS="$SEEDS",T_MAX="$T_MAX",TEST_INTERVAL="$TEST_INTERVAL",BATCH_SIZE_RUN="$BATCH_SIZE_RUN",BATCH_SIZE="$BATCH_SIZE",BUFFER_SIZE="$BUFFER_SIZE",USE_WANDB="$USE_WANDB",WANDB_MODE="$WANDB_MODE",USE_CUDA="$USE_CUDA",RUN_PREFIX="$RUN_PREFIX",EXTRA_ARGS="$EXTRA_ARGS" \
    scripts/ozstar_train_model_list_offline.sbatch
done

