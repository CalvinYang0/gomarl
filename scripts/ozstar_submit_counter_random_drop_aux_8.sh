#!/bin/bash
set -euo pipefail

# Eight experiments requested for GRF Counter:
#   4 equal1 relation/pairing variants submitted by the companion script;
#   4 equal1 + random-drop auxiliary variants:
#     drop probability {0.8, 0.5} x auxiliary weight {0.5, 1.0}.
# The random mask is resampled per timestep and directly replaces the learned
# mask only on the auxiliary TD path.

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
RELATION_SCALE="${RELATION_SCALE:-0.10}"
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

"$PYTHON_BIN" -m py_compile \
  src/modules/agents/clean_hyper_agent.py \
  src/controllers/clean_controller.py \
  src/learners/clean_learner.py
"$PYTHON_BIN" scripts/smoke_test_counter_mask_parameter_relation.py

if [[ "${CANCEL_OLD:-NO}" == "YES" ]]; then
  mapfile -t old_jobs < <(squeue -h -u "$USER" -o "%A" | sort -u)
  if (( ${#old_jobs[@]} > 0 )); then
    echo "cancelling existing jobs: ${old_jobs[*]}"
    scancel "${old_jobs[@]}"
  else
    echo "no existing jobs to cancel"
  fi
else
  echo "ERROR: set CANCEL_OLD=YES to confirm cancellation of all existing jobs" >&2
  exit 2
fi

# Submit the four non-random relation variants first.  This script performs no
# cancellation, so the all-jobs cancellation above remains the single explicit
# destructive action.
REPO_DIR="$REPO_DIR" \
PYTHON_BIN="$PYTHON_BIN" \
SEED="$SEED" \
T_MAX="$T_MAX" \
TEST_INTERVAL="$TEST_INTERVAL" \
TIME="$TIME" \
CPUS_PER_TASK="$CPUS_PER_TASK" \
MEMORY="$MEMORY" \
BATCH_SIZE_RUN="$BATCH_SIZE_RUN" \
BATCH_SIZE="$BATCH_SIZE" \
BUFFER_SIZE="$BUFFER_SIZE" \
TORCH_NUM_THREADS="$TORCH_NUM_THREADS" \
RELATION_SCALE="$RELATION_SCALE" \
SUBMIT_GAP_SECONDS="$SUBMIT_GAP_SECONDS" \
USE_WANDB="$USE_WANDB" \
WANDB_MODE="$WANDB_MODE" \
USE_CUDA="$USE_CUDA" \
EXTRA_ARGS="$EXTRA_ARGS" \
bash scripts/ozstar_submit_counter_equal1_relation_pairing_4.sh

common_args="$EXTRA_ARGS torch_num_threads=$TORCH_NUM_THREADS torch_num_interop_threads=1 learner_updates_per_collect=1 clean_relation_teacher_td_coef=0.0 clean_relation_distill_coef=0.0 clean_smooth_head_loss_coef=0.0 clean_action_pred_loss_coef=0.0 clean_public_delta_loss_coef=0.0 clean_condition_gradient_consistency_coef=0.0 clean_generated_parameter_stability_coef=0.0 clean_td_weighted_parameter_likelihood_coef=0.0 clean_hard_gate_initial_keep_probability=0.95 clean_binary_concrete_temperature=0.5 clean_dynamic_branch_gate_warmup_steps=250000 clean_importance_auxiliary_warmup_steps=250000 clean_importance_alternating_training=False clean_mask_parameter_relation_scale=$RELATION_SCALE clean_mask_parameter_relation_coef=1.0 clean_mask_parameter_relation_temporal_coef=0.0 clean_mask_parameter_relation_perturbed_head_coef=0.0 clean_mask_parameter_relation_gate_regularization_coef=0.0 env_worker_startup_stagger=0.25 env_worker_reset_retries=5 env_worker_reset_retry_delay=2.0 env_worker_response_timeout=180.0 env_args.write_video=False save_battle_trace=False"

submit_one() {
  local drop_label="$1" keep_probability="$2" weight_label="$3"
  local weight="$4" scope="$5"
  local label="random_drop_d${drop_label}_w${weight_label}_${scope}"
  local job_name="grf_counter_${label}_s${SEED}"
  local run_name="grf_counter_${label}_10m_s${SEED}"
  local scoped_args job_id

  scoped_args="$common_args clean_random_drop_auxiliary_keep_probability=$keep_probability clean_random_drop_auxiliary_coef=$weight clean_random_drop_auxiliary_scope=$scope"
  job_id=$(sbatch --parsable \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEMORY" \
    --time="$TIME" \
    --job-name="$job_name" \
    --output=ozstar_logs/%x_%j.out \
    --error=ozstar_logs/%x_%j.err \
    --export=ALL,PYTHON_BIN="$PYTHON_BIN",CONFIG=clean_hyper,ENV_CONFIG=academy_counterattack_easy,MAP_NAME=academy_counterattack_easy,MODEL_TYPE=grf_abs_dual_branch_binary_concrete_random_drop_aux_hypercond,SEED="$SEED",T_MAX="$T_MAX",TEST_INTERVAL="$TEST_INTERVAL",BATCH_SIZE_RUN="$BATCH_SIZE_RUN",EXPECTED_BATCH_SIZE_RUN="$BATCH_SIZE_RUN",BATCH_SIZE="$BATCH_SIZE",BUFFER_SIZE="$BUFFER_SIZE",USE_WANDB="$USE_WANDB",WANDB_MODE="$WANDB_MODE",USE_CUDA="$USE_CUDA",RUN_NAME="$run_name",GROUP_NAME="$run_name",OMP_NUM_THREADS="$TORCH_NUM_THREADS",MKL_NUM_THREADS="$TORCH_NUM_THREADS",OPENBLAS_NUM_THREADS="$TORCH_NUM_THREADS",NUMEXPR_NUM_THREADS="$TORCH_NUM_THREADS",EXTRA_ARGS="$scoped_args" \
    scripts/ozstar_train_offline.sbatch)
  printf 'submitted job=%s name=%s drop=%s keep=%s weight=%s scope=%s\n' \
    "${job_id%%;*}" "$job_name" "$drop_label" "$keep_probability" "$weight" "$scope"
}

processed=0
for spec in \
  "80|0.20|05|0.5|timestep" \
  "80|0.20|10|1.0|timestep" \
  "50|0.50|05|0.5|timestep" \
  "50|0.50|10|1.0|timestep"
do
  if (( processed > 0 )); then
    sleep "$SUBMIT_GAP_SECONDS"
  fi
  IFS='|' read -r drop_label keep_probability weight_label weight scope <<< "$spec"
  submit_one "$drop_label" "$keep_probability" "$weight_label" "$weight" "$scope"
  processed=$((processed + 1))
done

echo "submitted random-drop total: $processed; requested grand total: 8"
squeue -u "$USER" -o "%.18i %.70j %.10T %.12M %.10m %R"
