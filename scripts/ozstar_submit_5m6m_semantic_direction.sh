#!/bin/bash
set -euo pipefail

# Controlled direction ablation on 5m_vs_6m. The inverse variants use the
# same scores, threshold, and warmup, then swap the complete TOKEN and BIAS
# assignments selected by the normal route.

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl}"
MAP_NAME="${MAP_NAME:-5m_vs_6m}"
SEEDS="${SEEDS:-1}"
MODELS="${MODELS:-rpg_simple_bias_gradient_importance_router_hypercond rpg_simple_bias_gradient_importance_inverse_router_hypercond rpg_simple_bias_parameter_sensitivity_router_hypercond rpg_simple_bias_parameter_sensitivity_inverse_router_hypercond rpg_simple_bias_counterfactual_router_hypercond}"

CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
MEM="${MEM:-64G}"
TIME="${TIME:-12:00:00}"
BATCH_SIZE_RUN="${BATCH_SIZE_RUN:-8}"
BATCH_SIZE="${BATCH_SIZE:-32}"
BUFFER_SIZE="${BUFFER_SIZE:-500}"
TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-$CPUS_PER_TASK}"
TORCH_NUM_INTEROP_THREADS="${TORCH_NUM_INTEROP_THREADS:-1}"
T_MAX="${T_MAX:-5050000}"
TEST_INTERVAL="${TEST_INTERVAL:-50000}"
USE_WANDB="${USE_WANDB:-True}"
WANDB_MODE="${WANDB_MODE:-offline}"
USE_CUDA="${USE_CUDA:-False}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "$REPO_DIR"
mkdir -p ozstar_logs

short_name() {
  case "$1" in
    rpg_simple_bias_gradient_importance_router_hypercond) echo "gradimp_pos" ;;
    rpg_simple_bias_gradient_importance_inverse_router_hypercond) echo "gradimp_inv" ;;
    rpg_simple_bias_parameter_sensitivity_router_hypercond) echo "paramsens_pos" ;;
    rpg_simple_bias_parameter_sensitivity_inverse_router_hypercond) echo "paramsens_inv" ;;
    rpg_simple_bias_counterfactual_router_hypercond) echo "counterfact" ;;
    *) echo "semantic_dir" ;;
  esac
}

echo "== Submit 5m6m semantic direction ablation =="
echo "models: $MODELS"
echo "seeds: $SEEDS"
echo "setting: map=${MAP_NAME}, ${CPUS_PER_TASK}c ${MEM} ${TIME}, t_max=${T_MAX}, br=${BATCH_SIZE_RUN}, batch=${BATCH_SIZE}, buffer=${BUFFER_SIZE}, torch_threads=${TORCH_NUM_THREADS}/${TORCH_NUM_INTEROP_THREADS}"

for model_type in $MODELS; do
  for seed in $SEEDS; do
    tag="$(short_name "$model_type")"
    run_name="${MAP_NAME}_${model_type}_f5m32_s${seed}"
    job_name="5m6m_dir_${tag}_s${seed}"

    echo "submit: model=$model_type seed=$seed run=$run_name"
    sbatch \
      --cpus-per-task="$CPUS_PER_TASK" \
      --mem="$MEM" \
      --time="$TIME" \
      --job-name="$job_name" \
      --output=ozstar_logs/%x_%j.out \
      --error=ozstar_logs/%x_%j.err \
      --export=ALL,MAP_NAME="$MAP_NAME",MODEL_TYPE="$model_type",SEED="$seed",T_MAX="$T_MAX",TEST_INTERVAL="$TEST_INTERVAL",BATCH_SIZE_RUN="$BATCH_SIZE_RUN",BATCH_SIZE="$BATCH_SIZE",BUFFER_SIZE="$BUFFER_SIZE",USE_WANDB="$USE_WANDB",WANDB_MODE="$WANDB_MODE",USE_CUDA="$USE_CUDA",RUN_NAME="$run_name",OMP_NUM_THREADS="$TORCH_NUM_THREADS",MKL_NUM_THREADS="$TORCH_NUM_THREADS",EXTRA_ARGS="$EXTRA_ARGS torch_num_threads=$TORCH_NUM_THREADS torch_num_interop_threads=$TORCH_NUM_INTEROP_THREADS" \
      scripts/ozstar_train_offline.sbatch
  done
done
