#!/bin/bash
set -euo pipefail

# Matched Counter single-Linear learned-mask pair:
#   1) observation-conditioned Binary-Concrete slot mask;
#   2) learned mask multiplied by a timestep-wise Bernoulli mask.
# The random auxiliary has keep=0.5 and TD weight=1.0.

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
SUBMIT_GAP_SECONDS="${SUBMIT_GAP_SECONDS:-1}"
USE_WANDB="${USE_WANDB:-True}"
WANDB_MODE="${WANDB_MODE:-offline}"
USE_CUDA="${USE_CUDA:-False}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "$REPO_DIR"
mkdir -p ozstar_logs

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 2
fi

"$PYTHON_BIN" -m py_compile \
  src/modules/agents/clean_hyper_agent.py \
  src/controllers/clean_controller.py \
  src/learners/clean_learner.py
"$PYTHON_BIN" scripts/smoke_test_dual_branch_dynamic_gate.py

active_job() {
  local exact_name="$1"
  squeue -u "$USER" -h -o "%i|%j|%T" |
    awk -F'|' -v expected="$exact_name" \
      '$2 == expected && ($3 == "RUNNING" || $3 == "PENDING") {print $1; exit}'
}

common_args="$EXTRA_ARGS torch_num_threads=$TORCH_NUM_THREADS torch_num_interop_threads=1 learner_updates_per_collect=1 clean_relation_teacher_td_coef=0.0 clean_relation_distill_coef=0.0 clean_smooth_head_loss_coef=0.0 clean_action_pred_loss_coef=0.0 clean_public_delta_loss_coef=0.0 clean_condition_gradient_consistency_coef=0.0 clean_generated_parameter_stability_coef=0.0 clean_td_weighted_parameter_likelihood_coef=0.0 clean_mask_parameter_relation_coef=0.0 clean_mask_parameter_relation_temporal_coef=0.0 clean_mask_parameter_relation_perturbed_head_coef=0.0 clean_mask_parameter_relation_gate_regularization_coef=0.0 clean_hard_gate_initial_keep_probability=0.95 clean_binary_concrete_temperature=0.5 clean_dynamic_branch_gate_warmup_steps=250000 clean_dynamic_branch_gate_scope=both clean_dynamic_branch_gate_regularizer=none clean_importance_auxiliary_warmup_steps=250000 clean_importance_alternating_training=False env_worker_startup_stagger=0.25 env_worker_reset_retries=5 env_worker_reset_retry_delay=2.0 env_worker_response_timeout=180.0 env_args.write_video=False save_battle_trace=False"

submit_one() {
  local job_name="$1" run_name="$2" model_type="$3" variant_args="$4"
  local existing job_id
  existing=$(active_job "$job_name")
  if [[ -n "$existing" ]]; then
    echo "reused active job=$existing name=$job_name"
    return
  fi
  job_id=$(sbatch --parsable \
    --nodes=1 --ntasks=1 \
    --cpus-per-task="$CPUS_PER_TASK" --mem="$MEMORY" --time="$TIME" \
    --job-name="$job_name" \
    --output=ozstar_logs/%x_%j.out --error=ozstar_logs/%x_%j.err \
    --export=ALL,PYTHON_BIN="$PYTHON_BIN",CONFIG=clean_hyper,ENV_CONFIG=academy_counterattack_easy,MAP_NAME=academy_counterattack_easy,MODEL_TYPE="$model_type",SEED="$SEED",T_MAX="$T_MAX",TEST_INTERVAL="$TEST_INTERVAL",BATCH_SIZE_RUN="$BATCH_SIZE_RUN",EXPECTED_BATCH_SIZE_RUN="$BATCH_SIZE_RUN",BATCH_SIZE="$BATCH_SIZE",BUFFER_SIZE="$BUFFER_SIZE",USE_WANDB="$USE_WANDB",WANDB_MODE="$WANDB_MODE",USE_CUDA="$USE_CUDA",RUN_NAME="$run_name",GROUP_NAME="$run_name",OMP_NUM_THREADS="$TORCH_NUM_THREADS",MKL_NUM_THREADS="$TORCH_NUM_THREADS",OPENBLAS_NUM_THREADS="$TORCH_NUM_THREADS",NUMEXPR_NUM_THREADS="$TORCH_NUM_THREADS",EXTRA_ARGS="$common_args $variant_args" \
    scripts/ozstar_train_offline.sbatch)
  echo "submitted job=${job_id%%;*} name=$job_name model=$model_type"
}

submit_one \
  "grf_counter_linear_only_obs_gate_s${SEED}" \
  "grf_counter_linear_only_obs_gate_10m_s${SEED}" \
  "grf_abs_single_linear_branch_binary_concrete_gate_hypercond" \
  "clean_random_drop_auxiliary_coef=0.0"
sleep "$SUBMIT_GAP_SECONDS"

submit_one \
  "grf_counter_linear_only_obs_gate_randommask_d50_w10_timestep_s${SEED}" \
  "grf_counter_linear_only_obs_gate_randommask_d50_w10_timestep_10m_s${SEED}" \
  "grf_abs_single_linear_branch_binary_concrete_gate_random_drop_aux_hypercond" \
  "clean_random_drop_auxiliary_keep_probability=0.50 clean_random_drop_auxiliary_coef=1.0 clean_random_drop_auxiliary_scope=timestep clean_random_drop_auxiliary_combine_mode=multiply"

echo "processed total: 2"
squeue -u "$USER" -o "%.18i %.90j %.10T %.12M %.10m %R"
