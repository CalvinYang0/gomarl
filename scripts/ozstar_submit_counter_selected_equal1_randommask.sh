#!/bin/bash
set -euo pipefail

# Keep exactly the two user-selected legacy runs, cancel every other active
# user job, and submit the controlled equal1/random-mask comparison:
#   - four equal1 relation variants;
#   - equal1 + multiply random mask, episode scope, weight 1, drop {0.5, 0.8};
#   - TD-only + multiply random mask, episode scope, weight 1, drop 0.5.
# The TD-only d50 job is reused when the selected legacy run is still active.

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
RELATION_SCALE="${RELATION_SCALE:-0.10}"
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
"$PYTHON_BIN" scripts/smoke_test_counter_mask_parameter_relation.py

keep_d80="grf_counter_random_drop_d80_w05_timestep_s${SEED}"
keep_td_d50="grf_counter_random_drop_d50_w10_episode_s${SEED}"

mapfile -t cancel_ids < <(
  squeue -h -u "$USER" -o "%A|%j" |
    awk -F'|' -v keep_a="$keep_d80" -v keep_b="$keep_td_d50" \
      '$2 != keep_a && $2 != keep_b {print $1}' |
    sort -u
)
if (( ${#cancel_ids[@]} > 0 )); then
  echo "cancelling all active jobs except $keep_d80 and $keep_td_d50: ${cancel_ids[*]}"
  scancel "${cancel_ids[@]}"
else
  echo "no non-preserved active jobs to cancel"
fi

# Submit the four requested equal1 relation variants.
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

active_job() {
  local exact_name="$1"
  squeue -u "$USER" -h -o "%i|%j|%T" |
    awk -F'|' -v expected="$exact_name" \
      '$2 == expected && ($3 == "RUNNING" || $3 == "PENDING") {print $1; exit}'
}

common_args="$EXTRA_ARGS torch_num_threads=$TORCH_NUM_THREADS torch_num_interop_threads=1 learner_updates_per_collect=1 clean_relation_teacher_td_coef=0.0 clean_relation_distill_coef=0.0 clean_smooth_head_loss_coef=0.0 clean_action_pred_loss_coef=0.0 clean_public_delta_loss_coef=0.0 clean_condition_gradient_consistency_coef=0.0 clean_generated_parameter_stability_coef=0.0 clean_td_weighted_parameter_likelihood_coef=0.0 clean_hard_gate_initial_keep_probability=0.95 clean_binary_concrete_temperature=0.5 clean_dynamic_branch_gate_warmup_steps=250000 clean_importance_auxiliary_warmup_steps=250000 clean_importance_alternating_training=False clean_mask_parameter_relation_scale=$RELATION_SCALE clean_mask_parameter_relation_temporal_coef=0.0 clean_mask_parameter_relation_perturbed_head_coef=0.0 clean_mask_parameter_relation_gate_regularization_coef=0.0 clean_random_drop_auxiliary_coef=1.0 clean_random_drop_auxiliary_scope=episode clean_random_drop_auxiliary_combine_mode=multiply env_worker_startup_stagger=0.25 env_worker_reset_retries=5 env_worker_reset_retry_delay=2.0 env_worker_response_timeout=180.0 env_args.write_video=False save_battle_trace=False"

submit_random() {
  local label="$1" job_name="$2" run_name="$3" keep_probability="$4" relation_coef="$5"
  local existing scoped_args job_id
  existing=$(active_job "$job_name")
  if [[ -n "$existing" ]]; then
    echo "reused active job=$existing name=$job_name"
    return
  fi
  scoped_args="$common_args clean_random_drop_auxiliary_keep_probability=$keep_probability clean_mask_parameter_relation_coef=$relation_coef"
  job_id=$(sbatch --parsable \
    --nodes=1 --ntasks=1 \
    --cpus-per-task="$CPUS_PER_TASK" --mem="$MEMORY" --time="$TIME" \
    --job-name="$job_name" \
    --output=ozstar_logs/%x_%j.out --error=ozstar_logs/%x_%j.err \
    --export=ALL,PYTHON_BIN="$PYTHON_BIN",CONFIG=clean_hyper,ENV_CONFIG=academy_counterattack_easy,MAP_NAME=academy_counterattack_easy,MODEL_TYPE=grf_abs_dual_branch_binary_concrete_random_drop_aux_hypercond,SEED="$SEED",T_MAX="$T_MAX",TEST_INTERVAL="$TEST_INTERVAL",BATCH_SIZE_RUN="$BATCH_SIZE_RUN",EXPECTED_BATCH_SIZE_RUN="$BATCH_SIZE_RUN",BATCH_SIZE="$BATCH_SIZE",BUFFER_SIZE="$BUFFER_SIZE",USE_WANDB="$USE_WANDB",WANDB_MODE="$WANDB_MODE",USE_CUDA="$USE_CUDA",RUN_NAME="$run_name",GROUP_NAME="$run_name",OMP_NUM_THREADS="$TORCH_NUM_THREADS",MKL_NUM_THREADS="$TORCH_NUM_THREADS",OPENBLAS_NUM_THREADS="$TORCH_NUM_THREADS",NUMEXPR_NUM_THREADS="$TORCH_NUM_THREADS",EXTRA_ARGS="$scoped_args" \
    scripts/ozstar_train_offline.sbatch)
  echo "submitted job=${job_id%%;*} name=$job_name $label"
}

sleep "$SUBMIT_GAP_SECONDS"
submit_random \
  "equal1 multiply-mask drop=0.5 weight=1 episode" \
  "grf_counter_equal1_randommask_d50_w10_episode_s${SEED}" \
  "grf_counter_equal1_randommask_d50_w10_episode_10m_s${SEED}" \
  "0.50" "1.0"
sleep "$SUBMIT_GAP_SECONDS"
submit_random \
  "equal1 multiply-mask drop=0.8 weight=1 episode" \
  "grf_counter_equal1_randommask_d80_w10_episode_s${SEED}" \
  "grf_counter_equal1_randommask_d80_w10_episode_10m_s${SEED}" \
  "0.20" "1.0"
sleep "$SUBMIT_GAP_SECONDS"
submit_random \
  "TD-only multiply-mask drop=0.5 weight=1 episode" \
  "$keep_td_d50" \
  "grf_counter_random_drop_d50_w10_episode_10m_s${SEED}" \
  "0.50" "0.0"

echo "preserved legacy job name: $keep_d80"
echo "preserved/reused TD-only job name: $keep_td_d50"
squeue -u "$USER" -o "%.18i %.90j %.10T %.12M %.10m %R"
