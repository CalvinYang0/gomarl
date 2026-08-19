#!/bin/bash
set -euo pipefail

# Two Counter controls on the old joint framework:
# 1) suppress every adjacent generated-parameter change;
# 2) suppress only relative changes below margin/2 and leave larger switches
#    entirely to TD loss. In both cases the auxiliary updates only the gate.

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl-dual-branch}"
PYTHON_BIN="${PYTHON_BIN:-/home/kyang/.conda/envs/marl_cpu/bin/python}"
SEED="${SEED:-1}"
T_MAX="${T_MAX:-10050000}"
TEST_INTERVAL="${TEST_INTERVAL:-50000}"
TIME="${TIME:-2-00:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-28}"
MEMORY="${MEMORY:-24G}"
BATCH_SIZE_RUN="${BATCH_SIZE_RUN:-8}"
BATCH_SIZE="${BATCH_SIZE:-128}"
BUFFER_SIZE="${BUFFER_SIZE:-5000}"
TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-$CPUS_PER_TASK}"
INITIAL_KEEP_PROBABILITY="${INITIAL_KEEP_PROBABILITY:-0.95}"
BINARY_CONCRETE_TEMPERATURE="${BINARY_CONCRETE_TEMPERATURE:-0.5}"
AUXILIARY_TARGET_RATIO="${AUXILIARY_TARGET_RATIO:-0.10}"
SWITCH_MARGIN="${SWITCH_MARGIN:-0.10}"
USE_WANDB="${USE_WANDB:-True}"
WANDB_MODE="${WANDB_MODE:-offline}"
USE_CUDA="${USE_CUDA:-False}"
SUBMIT_GAP_SECONDS="${SUBMIT_GAP_SECONDS:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "$REPO_DIR"
mkdir -p ozstar_logs

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 2
fi
if (( BATCH_SIZE_RUN > CPUS_PER_TASK )); then
  echo "ERROR: BATCH_SIZE_RUN=$BATCH_SIZE_RUN exceeds CPUS_PER_TASK=$CPUS_PER_TASK" >&2
  exit 2
fi

"$PYTHON_BIN" scripts/smoke_test_dual_branch_dynamic_gate.py

common_args="$EXTRA_ARGS torch_num_threads=$TORCH_NUM_THREADS torch_num_interop_threads=1 learner_updates_per_collect=1 clean_relation_teacher_td_coef=0.0 clean_relation_distill_coef=0.0 clean_smooth_head_loss_coef=0.0 clean_action_pred_loss_coef=0.0 clean_public_delta_loss_coef=0.0 clean_condition_gradient_consistency_coef=0.0 clean_generated_parameter_stability_coef=0.0 clean_td_weighted_parameter_likelihood_coef=0.0 clean_hard_gate_initial_keep_probability=$INITIAL_KEEP_PROBABILITY clean_binary_concrete_temperature=$BINARY_CONCRETE_TEMPERATURE clean_dynamic_branch_gate_warmup_steps=250000 clean_importance_auxiliary_warmup_steps=250000 clean_adaptive_auxiliary_target_ratio=$AUXILIARY_TARGET_RATIO clean_temporal_param_switch_margin=$SWITCH_MARGIN clean_importance_alternating_training=False env_worker_startup_stagger=0.25 env_worker_reset_retries=5 env_worker_reset_retry_delay=2.0 env_worker_response_timeout=180.0 env_args.write_video=False save_battle_trace=False"

active_job() {
  local exact_name="$1"
  squeue -u "$USER" -h -o "%i|%j|%T" |
    awk -F'|' -v expected="$exact_name" \
      '$2 == expected && ($3 == "RUNNING" || $3 == "PENDING") {print $1; exit}'
}

submit_one() {
  local label="$1" model_type="$2"
  local job_name="grf_counter_${label}_s${SEED}"
  local run_name="grf_counter_${label}_10m_s${SEED}"
  local existing job_id

  existing=$(active_job "$job_name")
  if [[ -n "$existing" ]]; then
    echo "reused active job=$existing name=$job_name"
    return
  fi

  job_id=$(sbatch --parsable \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEMORY" \
    --time="$TIME" \
    --job-name="$job_name" \
    --output=ozstar_logs/%x_%j.out \
    --error=ozstar_logs/%x_%j.err \
    --export=ALL,PYTHON_BIN="$PYTHON_BIN",CONFIG=clean_hyper,ENV_CONFIG=academy_counterattack_easy,MAP_NAME=academy_counterattack_easy,MODEL_TYPE="$model_type",SEED="$SEED",T_MAX="$T_MAX",TEST_INTERVAL="$TEST_INTERVAL",BATCH_SIZE_RUN="$BATCH_SIZE_RUN",EXPECTED_BATCH_SIZE_RUN="$BATCH_SIZE_RUN",BATCH_SIZE="$BATCH_SIZE",BUFFER_SIZE="$BUFFER_SIZE",USE_WANDB="$USE_WANDB",WANDB_MODE="$WANDB_MODE",USE_CUDA="$USE_CUDA",RUN_NAME="$run_name",GROUP_NAME="$run_name",OMP_NUM_THREADS="$TORCH_NUM_THREADS",MKL_NUM_THREADS="$TORCH_NUM_THREADS",OPENBLAS_NUM_THREADS="$TORCH_NUM_THREADS",NUMEXPR_NUM_THREADS="$TORCH_NUM_THREADS",EXTRA_ARGS="$common_args" \
    scripts/ozstar_train_offline.sbatch)
  printf 'submitted job=%s name=%s resources=%sc/%s time=%s model=%s framework=joint\n' \
    "${job_id%%;*}" "$job_name" "$CPUS_PER_TASK" "$MEMORY" "$TIME" "$model_type"
}

submit_one \
  "temporal_param_stability" \
  "grf_abs_dual_branch_binary_concrete_temporal_param_stability_hypercond"
sleep "$SUBMIT_GAP_SECONDS"
submit_one \
  "temporal_param_small_change" \
  "grf_abs_dual_branch_binary_concrete_temporal_param_small_change_hypercond"
