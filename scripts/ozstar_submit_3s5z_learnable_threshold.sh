#!/bin/bash
set -euo pipefail

# Controlled four-way comparison: score type x fixed/learnable threshold.
# The learnable variants add one scalar threshold trained only by TD loss.

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl}"
MAP_NAME="${MAP_NAME:-3s5z_vs_3s6z}"
SEEDS="${SEEDS:-1}"
MODELS="${MODELS:-rpg_simple_bias_gradient_importance_router_hypercond rpg_simple_bias_gradient_importance_learnable_threshold_router_hypercond rpg_simple_bias_parameter_sensitivity_router_hypercond rpg_simple_bias_parameter_sensitivity_learnable_threshold_router_hypercond}"

CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
MEM="${MEM:-64G}"
TIME="${TIME:-2-00:00:00}"
BATCH_SIZE_RUN="${BATCH_SIZE_RUN:-32}"
BATCH_SIZE="${BATCH_SIZE:-32}"
BUFFER_SIZE="${BUFFER_SIZE:-500}"
REFERENCE_BATCH_SIZE_RUN="${REFERENCE_BATCH_SIZE_RUN:-8}"
LEARNER_UPDATES_PER_COLLECT="${LEARNER_UPDATES_PER_COLLECT:-$(( (BATCH_SIZE_RUN + REFERENCE_BATCH_SIZE_RUN - 1) / REFERENCE_BATCH_SIZE_RUN ))}"
TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"
TORCH_NUM_INTEROP_THREADS="${TORCH_NUM_INTEROP_THREADS:-1}"
T_MAX="${T_MAX:-10050000}"
TEST_INTERVAL="${TEST_INTERVAL:-50000}"
USE_WANDB="${USE_WANDB:-True}"
WANDB_MODE="${WANDB_MODE:-offline}"
USE_CUDA="${USE_CUDA:-False}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "$REPO_DIR"
mkdir -p ozstar_logs

# One SC2 process is created per rollout worker. Keep enough workers to occupy
# the requested cores; PyTorch/BLAS remain single-threaded to avoid nested CPU
# pools oversubscribing the node.
if (( BATCH_SIZE_RUN < CPUS_PER_TASK )); then
  echo "error: BATCH_SIZE_RUN=$BATCH_SIZE_RUN is smaller than CPUS_PER_TASK=$CPUS_PER_TASK" >&2
  echo "request fewer CPUs or increase BATCH_SIZE_RUN so the allocation is not idle" >&2
  exit 2
fi

short_name() {
  case "$1" in
    rpg_simple_bias_gradient_importance_router_hypercond) echo "gimp_fixed" ;;
    rpg_simple_bias_gradient_importance_learnable_threshold_router_hypercond) echo "gimp_lthr" ;;
    rpg_simple_bias_parameter_sensitivity_router_hypercond) echo "psens_fixed" ;;
    rpg_simple_bias_parameter_sensitivity_learnable_threshold_router_hypercond) echo "psens_lthr" ;;
    *) echo "threshold" ;;
  esac
}

echo "== Submit 3s5z fixed-vs-learnable threshold ablation =="
echo "models: $MODELS"
echo "setting: 1 node, 1 task, ${CPUS_PER_TASK}c ${MEM} ${TIME}, t_max=${T_MAX}, env_workers=${BATCH_SIZE_RUN}, learner_updates=${LEARNER_UPDATES_PER_COLLECT}"
echo "cpu plan: ${BATCH_SIZE_RUN} single-threaded SC2 workers + one single-threaded learner/controller process"

for model_type in $MODELS; do
  for seed in $SEEDS; do
    tag="$(short_name "$model_type")"
    run_name="${MAP_NAME}_${tag}_10m_env${BATCH_SIZE_RUN}_u${LEARNER_UPDATES_PER_COLLECT}_s${seed}"
    job_name="3s5z_${tag}_10m_s${seed}"

    echo "submit: model=$model_type seed=$seed run=$run_name"
    sbatch \
      --nodes=1 \
      --ntasks=1 \
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
