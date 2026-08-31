#!/bin/bash
set -euo pipefail

# Strict KL80 ablation on academy_counterattack_easy:
#   - dual linear + Transformer relation encoder and hypernetwork unchanged;
#   - no observation-conditioned gate network;
#   - no KL, relation, temporal, or random-mask auxiliary loss;
#   - first 250k steps keep every semantic slot;
#   - thereafter the online network independently keeps each branch/slot with
#     fixed probability 0.8 on every timestep;
#   - the target network uses the 0.8 expectation and evaluation keeps all
#     slots, matching KL80's train/target/test asymmetry as closely as possible.

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl-dual-branch}"
PYTHON_BIN="${PYTHON_BIN:-/home/kyang/.conda/envs/marl_cpu/bin/python}"
SEED="${SEED:-1}"
T_MAX="${T_MAX:-10050000}"
TEST_INTERVAL="${TEST_INTERVAL:-50000}"
TIME="${TIME:-2-00:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-28}"
MEMORY="${MEMORY:-16G}"
BATCH_SIZE_RUN="${BATCH_SIZE_RUN:-8}"
BATCH_SIZE="${BATCH_SIZE:-128}"
BUFFER_SIZE="${BUFFER_SIZE:-5000}"
TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-$CPUS_PER_TASK}"
USE_WANDB="${USE_WANDB:-True}"
WANDB_MODE="${WANDB_MODE:-offline}"
USE_CUDA="${USE_CUDA:-False}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
KEEP_PROBABILITY="${KEEP_PROBABILITY:-0.80}"

case "$KEEP_PROBABILITY" in
  0.8|0.80|.8|.80)
    KEEP_LABEL="k80"
    MODEL_TYPE="grf_abs_dual_branch_fixed_random_drop80_hypercond"
    ;;
  0.5|0.50|.5|.50)
    KEEP_LABEL="k50"
    MODEL_TYPE="grf_abs_dual_branch_fixed_random_drop50_hypercond"
    ;;
  *)
    echo "ERROR: KEEP_PROBABILITY must be 0.80 or 0.50" >&2
    exit 2
    ;;
esac

JOB_NAME="grf_counter_fixed_random_drop_${KEEP_LABEL}_s${SEED}${RUN_SUFFIX}"
RUN_NAME="grf_counter_fixed_random_drop_${KEEP_LABEL}_10m_s${SEED}${RUN_SUFFIX}"

cd "$REPO_DIR"
mkdir -p ozstar_logs wandb results

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 2
fi
if (( BATCH_SIZE_RUN > CPUS_PER_TASK )); then
  echo "ERROR: BATCH_SIZE_RUN=$BATCH_SIZE_RUN exceeds CPUS_PER_TASK=$CPUS_PER_TASK" >&2
  exit 2
fi

"$PYTHON_BIN" scripts/smoke_test_dual_branch_dynamic_gate.py

existing=$(
  squeue -u "$USER" -h -o "%i|%j|%T" |
    awk -F'|' -v expected="$JOB_NAME" \
      '$2 == expected && ($3 == "RUNNING" || $3 == "PENDING") {print $1; exit}'
)
if [[ -n "$existing" ]]; then
  echo "reused active job=$existing name=$JOB_NAME"
  exit 0
fi

COMMON_ARGS="$EXTRA_ARGS torch_num_threads=$TORCH_NUM_THREADS torch_num_interop_threads=1 learner_updates_per_collect=1 clean_relation_teacher_td_coef=0.0 clean_relation_distill_coef=0.0 clean_smooth_head_loss_coef=0.0 clean_action_pred_loss_coef=0.0 clean_public_delta_loss_coef=0.0 clean_condition_gradient_consistency_coef=0.0 clean_generated_parameter_stability_coef=0.0 clean_td_weighted_parameter_likelihood_coef=0.0 clean_mask_parameter_relation_coef=0.0 clean_mask_parameter_relation_temporal_coef=0.0 clean_mask_parameter_relation_perturbed_head_coef=0.0 clean_mask_parameter_relation_gate_regularization_coef=0.0 clean_random_drop_auxiliary_coef=0.0 clean_dynamic_branch_gate_warmup_steps=250000 clean_importance_auxiliary_warmup_steps=250000 clean_importance_alternating_training=False env_worker_startup_stagger=0.25 env_worker_reset_retries=5 env_worker_reset_retry_delay=2.0 env_worker_response_timeout=180.0 env_args.write_video=False save_battle_trace=False"

job_id=$(sbatch --parsable \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="$CPUS_PER_TASK" \
  --mem="$MEMORY" \
  --time="$TIME" \
  --job-name="$JOB_NAME" \
  --output=ozstar_logs/%x_%j.out \
  --error=ozstar_logs/%x_%j.err \
  --export=ALL,PYTHON_BIN="$PYTHON_BIN",CONFIG=clean_hyper,ENV_CONFIG=academy_counterattack_easy,MAP_NAME=academy_counterattack_easy,MODEL_TYPE="$MODEL_TYPE",SEED="$SEED",T_MAX="$T_MAX",TEST_INTERVAL="$TEST_INTERVAL",BATCH_SIZE_RUN="$BATCH_SIZE_RUN",EXPECTED_BATCH_SIZE_RUN="$BATCH_SIZE_RUN",BATCH_SIZE="$BATCH_SIZE",BUFFER_SIZE="$BUFFER_SIZE",USE_WANDB="$USE_WANDB",WANDB_MODE="$WANDB_MODE",USE_CUDA="$USE_CUDA",RUN_NAME="$RUN_NAME",GROUP_NAME="$RUN_NAME",OMP_NUM_THREADS="$TORCH_NUM_THREADS",MKL_NUM_THREADS="$TORCH_NUM_THREADS",OPENBLAS_NUM_THREADS="$TORCH_NUM_THREADS",NUMEXPR_NUM_THREADS="$TORCH_NUM_THREADS",EXTRA_ARGS="$COMMON_ARGS" \
  scripts/ozstar_train_offline.sbatch)

printf 'submitted job=%s name=%s resources=%sc/%s time=%s run=%s model=%s\n' \
  "${job_id%%;*}" "$JOB_NAME" "$CPUS_PER_TASK" "$MEMORY" "$TIME" "$RUN_NAME" "$MODEL_TYPE"

squeue -j "${job_id%%;*}" -o "%.18i %.80j %.10T %.12M %.10m %R"
