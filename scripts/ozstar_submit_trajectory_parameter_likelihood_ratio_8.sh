#!/bin/bash
set -euo pipefail

# Eight one-seed jobs. Counter and Corridor use the same trajectory-observed
# 64-D parameter-projection likelihood; only the detached-EMA target ratio
# changes: 0%, 2.5%, 5%, or 10% of TD loss.

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
PROJECTION_DIM="${PROJECTION_DIM:-64}"
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
  local target_ratio="${12}" scene_args="${13:-}"
  local existing job_id common_args variant_args

  existing=$(active_job "$job_name")
  if [[ -n "$existing" ]]; then
    echo "reused active job=$existing name=$job_name"
    return
  fi

  common_args="${EXTRA_ARGS} torch_num_threads=${torch_threads} torch_num_interop_threads=1 learner_updates_per_collect=1 clean_relation_teacher_td_coef=0.0 clean_relation_distill_coef=0.0 clean_smooth_head_loss_coef=0.0 clean_action_pred_loss_coef=0.0 clean_public_delta_loss_coef=0.0 clean_condition_gradient_consistency_coef=0.0 clean_generated_parameter_stability_coef=0.0 clean_td_weighted_parameter_likelihood_coef=0.0 clean_hard_gate_initial_keep_probability=${INITIAL_KEEP_PROBABILITY} clean_binary_concrete_temperature=${BINARY_CONCRETE_TEMPERATURE} clean_trajectory_parameter_projection_dim=${PROJECTION_DIM} clean_adaptive_auxiliary_target_ratio=${target_ratio} save_battle_trace=False"
  variant_args="clean_trajectory_parameter_likelihood_warmup_steps=250000"

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
  printf 'submitted job=%s name=%s ratio=%s cpus=%s memory=%s\n' \
    "${job_id%%;*}" "$job_name" "$target_ratio" "$cpus" "$memory"
}

grf_model="grf_abs_dual_branch_binary_concrete_adaptive_trajectory_parameter_likelihood_hypercond"
corridor_model="rpg_dual_branch_binary_concrete_adaptive_trajectory_parameter_likelihood_hypercond"
grf_env_args="env_worker_startup_stagger=0.25 env_worker_reset_retries=3 env_worker_reset_retry_delay=2.0 env_worker_response_timeout=180.0 env_args.write_video=False"

# label|ratio
declare -a ratios=(
  "r000|0.0"
  "r025|0.025"
  "r050|0.05"
  "r100|0.1"
)

submitted=0
for ratio_spec in "${ratios[@]}"; do
  IFS='|' read -r label ratio <<< "$ratio_spec"
  if (( submitted > 0 )); then sleep "$SUBMIT_GAP_SECONDS"; fi
  submit_one \
    "grf_counter_traj_param_${label}_s${SEED}" \
    "grf_counter_traj_param_likelihood_${label}_10m_s${SEED}" \
    "academy_counterattack_easy" "academy_counterattack_easy" \
    "$grf_model" 28 16G 8 128 5000 28 "$ratio" "$grf_env_args"
  submitted=$((submitted + 1))

  sleep "$SUBMIT_GAP_SECONDS"
  submit_one \
    "corridor_traj_param_${label}_s${SEED}" \
    "corridor_traj_param_likelihood_${label}_10m_s${SEED}" \
    "sc2" "corridor" \
    "$corridor_model" 32 96G 8 128 5000 32 "$ratio" ""
  submitted=$((submitted + 1))
done

echo "processed total: $submitted"
