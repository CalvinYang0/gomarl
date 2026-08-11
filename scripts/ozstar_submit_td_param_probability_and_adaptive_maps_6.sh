#!/bin/bash
set -euo pipefail

# Six one-seed jobs:
#   1) TD-advantage-weighted generated-parameter likelihood on Counter/Corridor.
#   2) Existing adaptive Binary-Concrete + gradient-consistency model on the
#      remaining GRF maps and SMAC 5m_vs_6m / 3s5z_vs_3s6z.

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl-dual-branch}"
PYTHON_BIN="${PYTHON_BIN:-/home/kyang/.conda/envs/marl_cpu/bin/python}"
SEED="${SEED:-1}"
T_MAX="${T_MAX:-10050000}"
TEST_INTERVAL="${TEST_INTERVAL:-50000}"
TIME="${TIME:-1-00:00:00}"
USE_WANDB="${USE_WANDB:-True}"
WANDB_MODE="${WANDB_MODE:-offline}"
USE_CUDA="${USE_CUDA:-False}"
INITIAL_KEEP_PROBABILITY="${INITIAL_KEEP_PROBABILITY:-0.95}"
BINARY_CONCRETE_TEMPERATURE="${BINARY_CONCRETE_TEMPERATURE:-0.5}"
ADAPTIVE_TARGET_RATIO="${ADAPTIVE_TARGET_RATIO:-0.1}"
TD_PARAMETER_RELATIVE_STD="${TD_PARAMETER_RELATIVE_STD:-0.02}"
TD_PARAMETER_LIKELIHOOD_COEF="${TD_PARAMETER_LIKELIHOOD_COEF:-0.01}"
SUBMIT_GAP_SECONDS="${SUBMIT_GAP_SECONDS:-1}"
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
  local learner_updates="${12}" variant_args="${13}" scene_args="${14:-}"
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

  common_args="${EXTRA_ARGS} torch_num_threads=${torch_threads} torch_num_interop_threads=1 learner_updates_per_collect=${learner_updates} clean_relation_teacher_td_coef=0.0 clean_relation_distill_coef=0.0 clean_smooth_head_loss_coef=0.0 clean_action_pred_loss_coef=0.0 clean_public_delta_loss_coef=0.0 clean_hard_gate_initial_keep_probability=${INITIAL_KEEP_PROBABILITY} clean_binary_concrete_temperature=${BINARY_CONCRETE_TEMPERATURE} clean_adaptive_auxiliary_target_ratio=${ADAPTIVE_TARGET_RATIO} save_battle_trace=False"

  job_id=$(sbatch --parsable \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$cpus" \
    --mem="$memory" \
    --time="$TIME" \
    --job-name="$job_name" \
    --output=ozstar_logs/%x_%j.out \
    --error=ozstar_logs/%x_%j.err \
    --export=ALL,PYTHON_BIN="$PYTHON_BIN",CONFIG=clean_hyper,ENV_CONFIG="$env_config",MAP_NAME="$map_name",MODEL_TYPE="$model",SEED="$SEED",T_MAX="$T_MAX",TEST_INTERVAL="$TEST_INTERVAL",BATCH_SIZE_RUN="$workers",EXPECTED_BATCH_SIZE_RUN="$workers",BATCH_SIZE="$batch_size",BUFFER_SIZE="$buffer_size",USE_WANDB="$USE_WANDB",WANDB_MODE="$WANDB_MODE",USE_CUDA="$USE_CUDA",RUN_NAME="$run_name",GROUP_NAME="$run_name",OMP_NUM_THREADS="$torch_threads",MKL_NUM_THREADS="$torch_threads",OPENBLAS_NUM_THREADS="$torch_threads",NUMEXPR_NUM_THREADS="$torch_threads",EXTRA_ARGS="$common_args $variant_args $scene_args" \
    scripts/ozstar_train_offline.sbatch)
  printf 'submitted job=%s name=%s cpus=%s memory=%s model=%s\n' \
    "${job_id%%;*}" "$job_name" "$cpus" "$memory" "$model"
}

td_probability_args="clean_condition_gradient_consistency_coef=0.0 clean_generated_parameter_stability_coef=0.0 clean_td_parameter_relative_std=${TD_PARAMETER_RELATIVE_STD} clean_td_weighted_parameter_likelihood_coef=${TD_PARAMETER_LIKELIHOOD_COEF}"
adaptive_args="clean_condition_gradient_consistency_coef=0.1 clean_generated_parameter_stability_coef=0.0 clean_td_weighted_parameter_likelihood_coef=0.0"
grf_env_args="env_worker_startup_stagger=0.25 env_worker_reset_retries=3 env_worker_reset_retry_delay=2.0 env_worker_response_timeout=180.0 env_args.write_video=False"

declare -a specs=(
  "grf_counter_td_param_prob_s${SEED}|grf_counter_td_param_probability_10m_s${SEED}|academy_counterattack_easy|academy_counterattack_easy|grf_abs_dual_branch_binary_concrete_adaptive_td_weighted_param_likelihood_hypercond|28|16G|8|128|5000|28|1|$td_probability_args|$grf_env_args"
  "corridor_td_param_prob_s${SEED}|corridor_td_param_probability_10m_s${SEED}|sc2|corridor|rpg_dual_branch_binary_concrete_adaptive_td_weighted_param_likelihood_hypercond|32|96G|8|128|5000|32|1|$td_probability_args|"
  "grf_pass_adaptive_concrete_s${SEED}|grf_pass_adaptive_concrete_10m_s${SEED}|academy_pass_and_shoot_with_keeper|academy_pass_and_shoot_with_keeper|grf_abs_dual_branch_binary_concrete_adaptive_grad_consistency_hypercond|28|16G|8|128|5000|28|1|$adaptive_args|$grf_env_args"
  "grf_3v1_adaptive_concrete_s${SEED}|grf_3v1_adaptive_concrete_10m_s${SEED}|academy_3_vs_1_with_keeper|academy_3_vs_1_with_keeper|grf_abs_dual_branch_binary_concrete_adaptive_grad_consistency_hypercond|28|16G|8|128|5000|28|1|$adaptive_args|$grf_env_args"
  "5m6m_adaptive_concrete_s${SEED}|5m_vs_6m_adaptive_concrete_10m_s${SEED}|sc2|5m_vs_6m|rpg_dual_branch_binary_concrete_adaptive_grad_consistency_hypercond|32|32G|8|32|500|32|1|$adaptive_args|"
  "3s5z_adaptive_concrete_s${SEED}|3s5z_vs_3s6z_adaptive_concrete_10m_s${SEED}|sc2|3s5z_vs_3s6z|rpg_dual_branch_binary_concrete_adaptive_grad_consistency_hypercond|32|64G|32|32|500|1|4|$adaptive_args|"
)

submitted=0
for spec in "${specs[@]}"; do
  if (( submitted > 0 )); then
    sleep "$SUBMIT_GAP_SECONDS"
  fi
  IFS='|' read -r job_name run_name env_config map_name model cpus memory \
    workers batch_size buffer_size torch_threads learner_updates variant_args \
    scene_args <<< "$spec"
  submit_one "$job_name" "$run_name" "$env_config" "$map_name" "$model" \
    "$cpus" "$memory" "$workers" "$batch_size" "$buffer_size" \
    "$torch_threads" "$learner_updates" "$variant_args" "$scene_args"
  submitted=$((submitted + 1))
done

echo "processed total: $submitted"
