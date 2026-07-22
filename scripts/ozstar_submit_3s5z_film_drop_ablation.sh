#!/bin/bash
set -euo pipefail

# Controlled semantic-use ablation on one SMAC map:
# FiLM modulation, direct DROP, hierarchical TOKEN/BIAS/DROP, sparse Top-K DROP.

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl}"
MAP_NAME="${MAP_NAME:-3s5z_vs_3s6z}"
SEEDS="${SEEDS:-1}"
MODELS="${MODELS:-rpg_simple_bias_gradient_importance_film_router_hypercond rpg_simple_bias_gradient_importance_drop_router_hypercond rpg_simple_bias_gradient_importance_hierarchical_drop_router_hypercond rpg_simple_bias_gradient_importance_sparse_drop_router_hypercond}"

CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
MEM="${MEM:-48G}"
TIME="${TIME:-1-00:00:00}"
BATCH_SIZE_RUN="${BATCH_SIZE_RUN:-32}"
BATCH_SIZE="${BATCH_SIZE:-32}"
BUFFER_SIZE="${BUFFER_SIZE:-500}"
REFERENCE_BATCH_SIZE_RUN="${REFERENCE_BATCH_SIZE_RUN:-8}"
LEARNER_UPDATES_PER_COLLECT="${LEARNER_UPDATES_PER_COLLECT:-$(( (BATCH_SIZE_RUN + REFERENCE_BATCH_SIZE_RUN - 1) / REFERENCE_BATCH_SIZE_RUN ))}"
TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"
TORCH_NUM_INTEROP_THREADS="${TORCH_NUM_INTEROP_THREADS:-1}"
T_MAX="${T_MAX:-10050000}"
TEST_INTERVAL="${TEST_INTERVAL:-50000}"
KEEP_THRESHOLD="${KEEP_THRESHOLD:-0.35}"
KEEP_RATIO="${KEEP_RATIO:-0.5}"
USE_WANDB="${USE_WANDB:-True}"
WANDB_MODE="${WANDB_MODE:-offline}"
USE_CUDA="${USE_CUDA:-False}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
PYTHON_BIN="${PYTHON_BIN:-/home/kyang/.conda/envs/marl_cpu/bin/python}"

cd "$REPO_DIR"
mkdir -p ozstar_logs

echo "== Pre-submit FiLM/DROP tensor smoke test =="
PYTHONPATH=src "$PYTHON_BIN" scripts/smoke_test_semantic_film_drop.py

short_name() {
  case "$1" in
    rpg_simple_bias_gradient_importance_film_router_hypercond) echo "gimp_film" ;;
    rpg_simple_bias_gradient_importance_drop_router_hypercond) echo "gimp_drop" ;;
    rpg_simple_bias_gradient_importance_hierarchical_drop_router_hypercond) echo "gimp_hdrop" ;;
    rpg_simple_bias_gradient_importance_sparse_drop_router_hypercond) echo "gimp_sdrop" ;;
    *) echo "semantic_use" ;;
  esac
}

map_tag="$(printf '%s' "$MAP_NAME" | tr -c '[:alnum:]_' '_')"
echo "== Submit ${MAP_NAME} FiLM/DROP ablation =="
echo "resources: ${CPUS_PER_TASK}c ${MEM} ${TIME}; env_workers=${BATCH_SIZE_RUN}; learner_updates=${LEARNER_UPDATES_PER_COLLECT}"
echo "drop: keep_threshold=${KEEP_THRESHOLD}; sparse_keep_ratio=${KEEP_RATIO}"

for model_type in $MODELS; do
  for seed in $SEEDS; do
    tag="$(short_name "$model_type")"
    run_name="${MAP_NAME}_${tag}_10m_env${BATCH_SIZE_RUN}_u${LEARNER_UPDATES_PER_COLLECT}_s${seed}"
    job_name="${map_tag}_${tag}_s${seed}"

    echo "submit: model=${model_type} seed=${seed} run=${run_name}"
    sbatch \
      --nodes=1 \
      --ntasks=1 \
      --cpus-per-task="$CPUS_PER_TASK" \
      --mem="$MEM" \
      --time="$TIME" \
      --job-name="$job_name" \
      --output=ozstar_logs/%x_%j.out \
      --error=ozstar_logs/%x_%j.err \
      --export=ALL,PYTHON_BIN="$PYTHON_BIN",MAP_NAME="$MAP_NAME",MODEL_TYPE="$model_type",SEED="$seed",T_MAX="$T_MAX",TEST_INTERVAL="$TEST_INTERVAL",BATCH_SIZE_RUN="$BATCH_SIZE_RUN",EXPECTED_BATCH_SIZE_RUN="$BATCH_SIZE_RUN",BATCH_SIZE="$BATCH_SIZE",BUFFER_SIZE="$BUFFER_SIZE",USE_WANDB="$USE_WANDB",WANDB_MODE="$WANDB_MODE",USE_CUDA="$USE_CUDA",RUN_NAME="$run_name",OMP_NUM_THREADS="$TORCH_NUM_THREADS",MKL_NUM_THREADS="$TORCH_NUM_THREADS",OPENBLAS_NUM_THREADS="$TORCH_NUM_THREADS",NUMEXPR_NUM_THREADS="$TORCH_NUM_THREADS",EXTRA_ARGS="$EXTRA_ARGS torch_num_threads=$TORCH_NUM_THREADS torch_num_interop_threads=$TORCH_NUM_INTEROP_THREADS learner_updates_per_collect=$LEARNER_UPDATES_PER_COLLECT clean_semantic_router_keep_threshold=$KEEP_THRESHOLD clean_semantic_router_keep_ratio=$KEEP_RATIO clean_relation_distill_coef=0.0 clean_relation_teacher_td_coef=0.0 clean_smooth_head_loss_coef=0.0" \
      scripts/ozstar_train_offline.sbatch
  done
done
