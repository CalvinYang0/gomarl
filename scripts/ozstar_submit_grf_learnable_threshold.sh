#!/bin/bash
set -euo pipefail

# GRF score type x fixed/learnable threshold ablation. By default this submits
# four independent jobs on the pass-and-shoot scenario; override ENVS to add
# the other academy scenarios.

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl}"
ENVS="${ENVS:-academy_pass_and_shoot_with_keeper}"
SEEDS="${SEEDS:-1}"
MODELS="${MODELS:-grf_abs_simple_bias_gradient_importance_router_hypercond grf_abs_simple_bias_gradient_importance_learnable_threshold_router_hypercond grf_abs_simple_bias_parameter_sensitivity_router_hypercond grf_abs_simple_bias_parameter_sensitivity_learnable_threshold_router_hypercond}"

CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-64G}"
TIME="${TIME:-2-00:00:00}"
BATCH_SIZE_RUN="${BATCH_SIZE_RUN:-8}"
BATCH_SIZE="${BATCH_SIZE:-128}"
BUFFER_SIZE="${BUFFER_SIZE:-5000}"
TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"
TORCH_NUM_INTEROP_THREADS="${TORCH_NUM_INTEROP_THREADS:-1}"
T_MAX="${T_MAX:-10050000}"
TEST_INTERVAL="${TEST_INTERVAL:-10000}"
ROUTER_EMA="${ROUTER_EMA:-0.99}"
ROUTER_THRESHOLD="${ROUTER_THRESHOLD:-0.5}"
ROUTER_TEMPERATURE="${ROUTER_TEMPERATURE:-0.1}"
ROUTER_WARMUP_STEPS="${ROUTER_WARMUP_STEPS:-250000}"
ROUTER_FREEZE_STEPS="${ROUTER_FREEZE_STEPS:-5000000}"
USE_WANDB="${USE_WANDB:-True}"
WANDB_MODE="${WANDB_MODE:-offline}"
USE_CUDA="${USE_CUDA:-False}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "$REPO_DIR"
mkdir -p ozstar_logs

if (( BATCH_SIZE_RUN < CPUS_PER_TASK )); then
  echo "error: BATCH_SIZE_RUN=$BATCH_SIZE_RUN is smaller than CPUS_PER_TASK=$CPUS_PER_TASK" >&2
  exit 2
fi

short_model() {
  case "$1" in
    grf_abs_simple_bias_gradient_importance_router_hypercond) echo "gimp_fixed" ;;
    grf_abs_simple_bias_gradient_importance_learnable_threshold_router_hypercond) echo "gimp_lthr" ;;
    grf_abs_simple_bias_parameter_sensitivity_router_hypercond) echo "psens_fixed" ;;
    grf_abs_simple_bias_parameter_sensitivity_learnable_threshold_router_hypercond) echo "psens_lthr" ;;
    *) echo "router" ;;
  esac
}

short_env() {
  case "$1" in
    academy_pass_and_shoot_with_keeper) echo "pass" ;;
    academy_3_vs_1_with_keeper) echo "3v1" ;;
    academy_counterattack_easy) echo "counter" ;;
    *) echo "grf" ;;
  esac
}

echo "== Submit GRF fixed-vs-learnable threshold ablation =="
echo "envs: $ENVS"
echo "models: $MODELS"
echo "setting: ${CPUS_PER_TASK}c ${MEM} ${TIME}, t_max=${T_MAX}, env_workers=${BATCH_SIZE_RUN}"

for env_config in $ENVS; do
  for model_type in $MODELS; do
    for seed in $SEEDS; do
      env_tag="$(short_env "$env_config")"
      model_tag="$(short_model "$model_type")"
      run_name="grf_${env_tag}_${model_tag}_10m_e${BATCH_SIZE_RUN}_s${seed}"
      job_name="grf_${env_tag}_${model_tag}_s${seed}"

      echo "submit: env=$env_config model=$model_type seed=$seed run=$run_name"
      sbatch \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task="$CPUS_PER_TASK" \
        --mem="$MEM" \
        --time="$TIME" \
        --job-name="$job_name" \
        --output=ozstar_logs/%x_%j.out \
        --error=ozstar_logs/%x_%j.err \
        --export=ALL,CONFIG=clean_hyper,ENV_CONFIG="$env_config",MAP_NAME="$env_config",MODEL_TYPE="$model_type",SEED="$seed",T_MAX="$T_MAX",TEST_INTERVAL="$TEST_INTERVAL",BATCH_SIZE_RUN="$BATCH_SIZE_RUN",BATCH_SIZE="$BATCH_SIZE",BUFFER_SIZE="$BUFFER_SIZE",USE_WANDB="$USE_WANDB",WANDB_MODE="$WANDB_MODE",USE_CUDA="$USE_CUDA",RUN_NAME="$run_name",OMP_NUM_THREADS="$TORCH_NUM_THREADS",MKL_NUM_THREADS="$TORCH_NUM_THREADS",OPENBLAS_NUM_THREADS="$TORCH_NUM_THREADS",NUMEXPR_NUM_THREADS="$TORCH_NUM_THREADS",EXTRA_ARGS="$EXTRA_ARGS torch_num_threads=$TORCH_NUM_THREADS torch_num_interop_threads=$TORCH_NUM_INTEROP_THREADS env_args.write_video=False clean_semantic_router_ema=$ROUTER_EMA clean_semantic_router_threshold=$ROUTER_THRESHOLD clean_semantic_router_temperature=$ROUTER_TEMPERATURE clean_semantic_router_warmup_steps=$ROUTER_WARMUP_STEPS clean_semantic_router_freeze_steps=$ROUTER_FREEZE_STEPS clean_relation_teacher_td_coef=0.0 clean_relation_distill_coef=0.0 clean_smooth_head_loss_coef=0.0 clean_action_pred_loss_coef=0.0 clean_public_delta_loss_coef=0.0" \
        scripts/ozstar_train_offline.sbatch
    done
  done
done
