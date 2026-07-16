#!/bin/bash
set -euo pipefail

# Every raw self/ally/enemy observation scalar is threshold-routed
# independently. SC2 rollout simulation, not the small CPU MLP, is the
# bottleneck, so this profile assigns one worker per requested CPU.

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl}"
MAP_NAME="${MAP_NAME:-3s5z_vs_3s6z}"
SEEDS="${SEEDS:-1}"
MODELS="${MODELS:-rpg_simple_bias_gradient_importance_router_hypercond rpg_simple_bias_parameter_sensitivity_router_hypercond rpg_simple_bias_counterfactual_router_hypercond}"

# Preserve the legacy training setting. The batch script derives OMP/MKL thread
# counts from SLURM_CPUS_PER_TASK, so do not cap PyTorch to four threads here.
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
MEM="${MEM:-64G}"
TIME="${TIME:-20:00:00}"
BATCH_SIZE_RUN="${BATCH_SIZE_RUN:-32}"
BATCH_SIZE="${BATCH_SIZE:-32}"
BUFFER_SIZE="${BUFFER_SIZE:-500}"
REFERENCE_BATCH_SIZE_RUN="${REFERENCE_BATCH_SIZE_RUN:-8}"
LEARNER_UPDATES_PER_COLLECT="${LEARNER_UPDATES_PER_COLLECT:-$(( (BATCH_SIZE_RUN + REFERENCE_BATCH_SIZE_RUN - 1) / REFERENCE_BATCH_SIZE_RUN ))}"
TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"
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
    rpg_simple_bias_observer_consistency_router_hypercond) echo "obscons" ;;
    rpg_simple_bias_temporal_stability_router_hypercond) echo "tempstable" ;;
    rpg_simple_bias_gradient_importance_router_hypercond) echo "gradimp" ;;
    rpg_simple_bias_gradient_consistency_router_hypercond) echo "gradcons" ;;
    rpg_simple_bias_parameter_sensitivity_router_hypercond) echo "paramsens" ;;
    rpg_simple_bias_counterfactual_router_hypercond) echo "counterfact" ;;
    *) echo "semantic_router" ;;
  esac
}

echo "== Submit 3s5z semantic-router ablation =="
echo "models: $MODELS"
echo "seeds: $SEEDS"
echo "setting: ${CPUS_PER_TASK}c ${MEM} ${TIME}, t_max=${T_MAX}, env_workers=${BATCH_SIZE_RUN}, learner_updates=${LEARNER_UPDATES_PER_COLLECT}, batch=${BATCH_SIZE}, buffer=${BUFFER_SIZE}, torch_threads=${TORCH_NUM_THREADS}/${TORCH_NUM_INTEROP_THREADS}"

for model_type in $MODELS; do
  for seed in $SEEDS; do
    tag="$(short_name "$model_type")"
    run_name="${MAP_NAME}_${model_type}_env${BATCH_SIZE_RUN}_u${LEARNER_UPDATES_PER_COLLECT}_s${seed}"
    job_name="3s5z_env${BATCH_SIZE_RUN}_${tag}_s${seed}"

    echo "submit: model=$model_type seed=$seed run=$run_name"
    sbatch \
      --cpus-per-task="$CPUS_PER_TASK" \
      --mem="$MEM" \
      --time="$TIME" \
      --job-name="$job_name" \
      --output=ozstar_logs/%x_%j.out \
      --error=ozstar_logs/%x_%j.err \
      --export=ALL,MAP_NAME="$MAP_NAME",MODEL_TYPE="$model_type",SEED="$seed",T_MAX="$T_MAX",TEST_INTERVAL="$TEST_INTERVAL",BATCH_SIZE_RUN="$BATCH_SIZE_RUN",BATCH_SIZE="$BATCH_SIZE",BUFFER_SIZE="$BUFFER_SIZE",USE_WANDB="$USE_WANDB",WANDB_MODE="$WANDB_MODE",USE_CUDA="$USE_CUDA",RUN_NAME="$run_name",OMP_NUM_THREADS="$TORCH_NUM_THREADS",MKL_NUM_THREADS="$TORCH_NUM_THREADS",OPENBLAS_NUM_THREADS="$TORCH_NUM_THREADS",NUMEXPR_NUM_THREADS="$TORCH_NUM_THREADS",EXTRA_ARGS="$EXTRA_ARGS torch_num_threads=$TORCH_NUM_THREADS torch_num_interop_threads=$TORCH_NUM_INTEROP_THREADS learner_updates_per_collect=$LEARNER_UPDATES_PER_COLLECT" \
      scripts/ozstar_train_offline.sbatch
  done
done
