#!/bin/bash
set -euo pipefail

# Six one-seed, 10M-step TD-only observation-gating jobs:
#   SMAC: corridor, 5m_vs_6m, 3s5z_vs_3s6z
#   GRF:  academy_counterattack_easy, academy_pass_and_shoot_with_keeper,
#         academy_3_vs_1_with_keeper
#
# These jobs intentionally reuse the r000 trajectory-likelihood model type so
# their gate implementation exactly matches the existing r000 result. The
# trajectory auxiliary target ratio is zero, hence the learned Binary-Concrete
# gates receive task gradients only from the TD objective.

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl-dual-branch}"
PYTHON_BIN="${PYTHON_BIN:-/home/kyang/.conda/envs/marl_cpu/bin/python}"
SEED="${SEED:-1}"
T_MAX="${T_MAX:-10050000}"
TEST_INTERVAL="${TEST_INTERVAL:-50000}"
TIME="${TIME:-2-00:00:00}"
USE_WANDB="${USE_WANDB:-True}"
WANDB_MODE="${WANDB_MODE:-offline}"
USE_CUDA="${USE_CUDA:-False}"
INITIAL_KEEP_PROBABILITY="${INITIAL_KEEP_PROBABILITY:-0.95}"
BINARY_CONCRETE_TEMPERATURE="${BINARY_CONCRETE_TEMPERATURE:-0.5}"
PROJECTION_DIM="${PROJECTION_DIM:-64}"
SUBMIT_GAP_SECONDS="${SUBMIT_GAP_SECONDS:-1}"
ENV_WORKER_STARTUP_STAGGER="${ENV_WORKER_STARTUP_STAGGER:-0.25}"
ENV_WORKER_RESET_RETRIES="${ENV_WORKER_RESET_RETRIES:-5}"
ENV_WORKER_RESET_RETRY_DELAY="${ENV_WORKER_RESET_RETRY_DELAY:-2.0}"
ENV_WORKER_RESPONSE_TIMEOUT="${ENV_WORKER_RESPONSE_TIMEOUT:-180.0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "$REPO_DIR"
mkdir -p ozstar_logs

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 2
fi

"$PYTHON_BIN" scripts/smoke_test_dual_branch_dynamic_gate.py

active_job() {
  local exact_name="$1"
  squeue -u "$USER" -h -o "%i|%j|%T" |
    awk -F'|' -v expected="$exact_name" \
      '$2 == expected && ($3 == "RUNNING" || $3 == "PENDING") {print $1; exit}'
}

submit_one() {
  local job_name="$1" run_name="$2" env_config="$3" map_name="$4"
  local model="$5" cpus="$6" memory="$7" workers="$8"
  local batch_size="$9" buffer_size="${10}" torch_threads="${11}"
  local learner_updates="${12}" scene_args="${13:-}"
  local existing job_id common_args

  existing=$(active_job "$job_name")
  if [[ -n "$existing" ]]; then
    echo "reused active job=$existing name=$job_name"
    return
  fi

  if (( workers > cpus )); then
    echo "ERROR: $job_name workers=$workers exceeds cpus=$cpus" >&2
    return 2
  fi

  common_args="${EXTRA_ARGS} torch_num_threads=${torch_threads} torch_num_interop_threads=1 learner_updates_per_collect=${learner_updates} clean_relation_teacher_td_coef=0.0 clean_relation_distill_coef=0.0 clean_smooth_head_loss_coef=0.0 clean_action_pred_loss_coef=0.0 clean_public_delta_loss_coef=0.0 clean_condition_gradient_consistency_coef=0.0 clean_generated_parameter_stability_coef=0.0 clean_td_weighted_parameter_likelihood_coef=0.0 clean_hard_gate_initial_keep_probability=${INITIAL_KEEP_PROBABILITY} clean_binary_concrete_temperature=${BINARY_CONCRETE_TEMPERATURE} clean_trajectory_parameter_projection_dim=${PROJECTION_DIM} clean_adaptive_auxiliary_target_ratio=0.0 clean_trajectory_parameter_likelihood_warmup_steps=250000 save_battle_trace=False ${scene_args}"

  job_id=$(sbatch --parsable \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$cpus" \
    --mem="$memory" \
    --time="$TIME" \
    --job-name="$job_name" \
    --output=ozstar_logs/%x_%j.out \
    --error=ozstar_logs/%x_%j.err \
    --export=ALL,PYTHON_BIN="$PYTHON_BIN",CONFIG=clean_hyper,ENV_CONFIG="$env_config",MAP_NAME="$map_name",MODEL_TYPE="$model",SEED="$SEED",T_MAX="$T_MAX",TEST_INTERVAL="$TEST_INTERVAL",BATCH_SIZE_RUN="$workers",EXPECTED_BATCH_SIZE_RUN="$workers",BATCH_SIZE="$batch_size",BUFFER_SIZE="$buffer_size",USE_WANDB="$USE_WANDB",WANDB_MODE="$WANDB_MODE",USE_CUDA="$USE_CUDA",RUN_NAME="$run_name",GROUP_NAME="$run_name",OMP_NUM_THREADS="$torch_threads",MKL_NUM_THREADS="$torch_threads",OPENBLAS_NUM_THREADS="$torch_threads",NUMEXPR_NUM_THREADS="$torch_threads",EXTRA_ARGS="$common_args" \
    scripts/ozstar_train_offline.sbatch)
  printf 'submitted job=%s name=%s map=%s time=%s cpus=%s memory=%s model=%s\n' \
    "${job_id%%;*}" "$job_name" "$map_name" "$TIME" "$cpus" "$memory" "$model"
}

grf_model="grf_abs_dual_branch_binary_concrete_adaptive_trajectory_parameter_likelihood_hypercond"
rpg_model="rpg_dual_branch_binary_concrete_adaptive_trajectory_parameter_likelihood_hypercond"
worker_args="env_worker_startup_stagger=${ENV_WORKER_STARTUP_STAGGER} env_worker_reset_retries=${ENV_WORKER_RESET_RETRIES} env_worker_reset_retry_delay=${ENV_WORKER_RESET_RETRY_DELAY} env_worker_response_timeout=${ENV_WORKER_RESPONSE_TIMEOUT}"
grf_args="${worker_args} env_args.write_video=False"

# job|run|env config|map|model|CPUs|memory|workers|batch|buffer|threads|updates|extra
declare -a specs=(
  "grf_counter_td_only_gate_s${SEED}|grf_counter_td_only_concrete_gate_10m_s${SEED}|academy_counterattack_easy|academy_counterattack_easy|${grf_model}|28|16G|8|128|5000|28|1|${grf_args}"
  "grf_pass_td_only_gate_s${SEED}|grf_pass_td_only_concrete_gate_10m_s${SEED}|academy_pass_and_shoot_with_keeper|academy_pass_and_shoot_with_keeper|${grf_model}|28|16G|8|128|5000|28|1|${grf_args}"
  "grf_3v1_td_only_gate_s${SEED}|grf_3v1_td_only_concrete_gate_10m_s${SEED}|academy_3_vs_1_with_keeper|academy_3_vs_1_with_keeper|${grf_model}|28|16G|8|128|5000|28|1|${grf_args}"
  "corridor_td_only_gate_s${SEED}|corridor_td_only_concrete_gate_10m_s${SEED}|sc2|corridor|${rpg_model}|32|96G|8|128|5000|32|1|${worker_args}"
  "5m6m_td_only_gate_s${SEED}|5m_vs_6m_td_only_concrete_gate_10m_s${SEED}|sc2|5m_vs_6m|${rpg_model}|32|32G|8|32|500|32|1|${worker_args}"
  "3s5z_td_only_gate_s${SEED}|3s5z_vs_3s6z_td_only_concrete_gate_10m_s${SEED}|sc2|3s5z_vs_3s6z|${rpg_model}|32|64G|32|32|500|1|4|${worker_args}"
)

submitted=0
for spec in "${specs[@]}"; do
  if (( submitted > 0 )); then
    sleep "$SUBMIT_GAP_SECONDS"
  fi
  IFS='|' read -r job_name run_name env_config map_name model cpus memory \
    workers batch_size buffer_size torch_threads learner_updates scene_args \
    <<< "$spec"
  submit_one "$job_name" "$run_name" "$env_config" "$map_name" "$model" \
    "$cpus" "$memory" "$workers" "$batch_size" "$buffer_size" \
    "$torch_threads" "$learner_updates" "$scene_args"
  submitted=$((submitted + 1))
done

echo "processed total: $submitted"
