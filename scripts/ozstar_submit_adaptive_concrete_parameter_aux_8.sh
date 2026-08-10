#!/bin/bash
set -euo pipefail

# Eight jobs: parameter stability versus conditional parameter likelihood,
# each with both-branch or Attention-only adaptive Binary-Concrete gates,
# crossed with Counter and Corridor. One seed by default.

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl-dual-branch}"
PYTHON_BIN="${PYTHON_BIN:-/home/kyang/.conda/envs/marl_cpu/bin/python}"
SEEDS="${SEEDS:-1}"
SCENES="${SCENES:-counter corridor}"
TIME="${TIME:-1-00:00:00}"
T_MAX="${T_MAX:-10050000}"
TEST_INTERVAL="${TEST_INTERVAL:-50000}"
GRF_CPUS="${GRF_CPUS:-28}"
CORRIDOR_CPUS="${CORRIDOR_CPUS:-32}"
GRF_MEM="${GRF_MEM:-16G}"
CORRIDOR_MEM="${CORRIDOR_MEM:-96G}"
ADAPTIVE_TARGET_RATIO="${ADAPTIVE_TARGET_RATIO:-0.1}"
INITIAL_KEEP_PROBABILITY="${INITIAL_KEEP_PROBABILITY:-0.95}"
BINARY_CONCRETE_TEMPERATURE="${BINARY_CONCRETE_TEMPERATURE:-0.5}"
PARAMETER_LIKELIHOOD_STD="${PARAMETER_LIKELIHOOD_STD:-1.0}"
BATCH_SIZE_RUN="${BATCH_SIZE_RUN:-8}"
BATCH_SIZE="${BATCH_SIZE:-128}"
BUFFER_SIZE="${BUFFER_SIZE:-5000}"
TORCH_NUM_INTEROP_THREADS="${TORCH_NUM_INTEROP_THREADS:-1}"
LEARNER_UPDATES_PER_COLLECT="${LEARNER_UPDATES_PER_COLLECT:-1}"
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

"$PYTHON_BIN" scripts/smoke_test_dual_branch_dynamic_gate.py

active_job() {
  local exact_name="$1"
  squeue -u "$USER" -h -o "%i|%j|%T" |
    awk -F'|' -v expected="$exact_name" \
      '$2 == expected && ($3 == "RUNNING" || $3 == "PENDING") {print $1; exit}'
}

submit_one() {
  local scene="$1"
  local variant="$2"
  local seed="$3"
  local tag env_config map_name cpus memory scene_args prefix model suffix

  case "$scene" in
    counter)
      tag="grf_counter"
      env_config="academy_counterattack_easy"
      map_name="$env_config"
      cpus="$GRF_CPUS"
      memory="$GRF_MEM"
      prefix="grf_abs"
      scene_args="env_worker_startup_stagger=0.25 env_worker_reset_retries=3 env_worker_reset_retry_delay=2.0 env_worker_response_timeout=180.0 env_args.write_video=False"
      ;;
    corridor)
      tag="corridor"
      env_config="sc2"
      map_name="corridor"
      cpus="$CORRIDOR_CPUS"
      memory="$CORRIDOR_MEM"
      prefix="rpg"
      scene_args=""
      ;;
    *) echo "ERROR: unsupported scene: $scene" >&2; return 2 ;;
  esac

  case "$variant" in
    param)
      model="${prefix}_dual_branch_binary_concrete_adaptive_param_stability_hypercond"
      suffix="adaptive_concrete_param_stability"
      ;;
    attention_param)
      model="${prefix}_dual_branch_binary_concrete_adaptive_attention_only_param_stability_hypercond"
      suffix="adaptive_concrete_attention_only_param_stability"
      ;;
    likelihood)
      model="${prefix}_dual_branch_binary_concrete_adaptive_parameter_likelihood_hypercond"
      suffix="adaptive_concrete_param_likelihood"
      ;;
    attention_likelihood)
      model="${prefix}_dual_branch_binary_concrete_adaptive_attention_only_parameter_likelihood_hypercond"
      suffix="adaptive_concrete_attention_only_param_likelihood"
      ;;
    *) echo "ERROR: unsupported variant: $variant" >&2; return 2 ;;
  esac

  local run_name="${tag}_${suffix}_10m_s${seed}"
  local job_name="${tag}_${suffix}_s${seed}"
  local existing job_id common_args
  existing=$(active_job "$job_name")
  if [[ -n "$existing" ]]; then
    echo "reused active job=$existing name=$job_name"
    return
  fi

  common_args="${EXTRA_ARGS} torch_num_threads=${cpus} torch_num_interop_threads=${TORCH_NUM_INTEROP_THREADS} learner_updates_per_collect=${LEARNER_UPDATES_PER_COLLECT} clean_relation_teacher_td_coef=0.0 clean_relation_distill_coef=0.0 clean_smooth_head_loss_coef=0.0 clean_action_pred_loss_coef=0.0 clean_public_delta_loss_coef=0.0 clean_condition_gradient_consistency_coef=0.0 clean_generated_parameter_stability_coef=1.0 clean_adaptive_auxiliary_target_ratio=${ADAPTIVE_TARGET_RATIO} clean_hard_gate_initial_keep_probability=${INITIAL_KEEP_PROBABILITY} clean_binary_concrete_temperature=${BINARY_CONCRETE_TEMPERATURE} clean_generated_parameter_likelihood_std=${PARAMETER_LIKELIHOOD_STD} save_battle_trace=False"

  job_id=$(sbatch --parsable \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$cpus" \
    --mem="$memory" \
    --time="$TIME" \
    --job-name="$job_name" \
    --output=ozstar_logs/%x_%j.out \
    --error=ozstar_logs/%x_%j.err \
    --export=ALL,PYTHON_BIN="$PYTHON_BIN",CONFIG=clean_hyper,ENV_CONFIG="$env_config",MAP_NAME="$map_name",MODEL_TYPE="$model",SEED="$seed",T_MAX="$T_MAX",TEST_INTERVAL="$TEST_INTERVAL",BATCH_SIZE_RUN="$BATCH_SIZE_RUN",EXPECTED_BATCH_SIZE_RUN="$BATCH_SIZE_RUN",BATCH_SIZE="$BATCH_SIZE",BUFFER_SIZE="$BUFFER_SIZE",USE_WANDB="$USE_WANDB",WANDB_MODE="$WANDB_MODE",USE_CUDA="$USE_CUDA",RUN_NAME="$run_name",GROUP_NAME="$run_name",OMP_NUM_THREADS="$cpus",MKL_NUM_THREADS="$cpus",OPENBLAS_NUM_THREADS="$cpus",NUMEXPR_NUM_THREADS="$cpus",EXTRA_ARGS="$common_args $scene_args" \
    scripts/ozstar_train_offline.sbatch)
  printf 'submitted job=%s scene=%s variant=%s seed=%s cpus=%s memory=%s model=%s\n' \
    "${job_id%%;*}" "$scene" "$variant" "$seed" "$cpus" "$memory" "$model"
}

submitted=0
for scene in $SCENES; do
  for variant in param attention_param likelihood attention_likelihood; do
    for seed in $SEEDS; do
      if (( submitted > 0 )); then
        sleep "$SUBMIT_GAP_SECONDS"
      fi
      submit_one "$scene" "$variant" "$seed"
      submitted=$((submitted + 1))
    done
  done
done

echo "processed total: $submitted"
