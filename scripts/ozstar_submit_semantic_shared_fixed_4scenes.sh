#!/bin/bash
set -euo pipefail

# Submit exactly eight experiments: a shared-field discovery run and a fresh
# fixed-mask compact run for each of three GRF scenarios plus SMAC corridor.
# The compact run depends on its discovery run and reuses the final learned mask.

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl}"
SEED="${SEED:-1}"
TIME="${TIME:-2-00:00:00}"
T_MAX="${T_MAX:-10050000}"
TEST_INTERVAL="${TEST_INTERVAL:-50000}"
BATCH_SIZE_RUN="${BATCH_SIZE_RUN:-32}"
BATCH_SIZE="${BATCH_SIZE:-128}"
BUFFER_SIZE="${BUFFER_SIZE:-5000}"
LEARNER_UPDATES_PER_COLLECT="${LEARNER_UPDATES_PER_COLLECT:-4}"
TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-4}"
TORCH_NUM_INTEROP_THREADS="${TORCH_NUM_INTEROP_THREADS:-1}"
ENV_WORKER_STARTUP_STAGGER="${ENV_WORKER_STARTUP_STAGGER:-0.25}"
ENV_WORKER_RESET_RETRIES="${ENV_WORKER_RESET_RETRIES:-3}"
ENV_WORKER_RESET_RETRY_DELAY="${ENV_WORKER_RESET_RETRY_DELAY:-2.0}"
ENV_WORKER_RESPONSE_TIMEOUT="${ENV_WORKER_RESPONSE_TIMEOUT:-180.0}"
ROUTER_EMA="${ROUTER_EMA:-0.99}"
ROUTER_THRESHOLD="${ROUTER_THRESHOLD:-0.5}"
ROUTER_TEMPERATURE="${ROUTER_TEMPERATURE:-0.1}"
ROUTER_WARMUP_STEPS="${ROUTER_WARMUP_STEPS:-250000}"
ROUTER_FREEZE_STEPS="${ROUTER_FREEZE_STEPS:-5000000}"
USE_WANDB="${USE_WANDB:-True}"
WANDB_MODE="${WANDB_MODE:-offline}"
USE_CUDA="${USE_CUDA:-False}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

SCENARIOS="${SCENARIOS:-academy_pass_and_shoot_with_keeper academy_3_vs_1_with_keeper academy_counterattack_easy corridor}"

cd "$REPO_DIR"
mkdir -p ozstar_logs

scenario_tag() {
  case "$1" in
    academy_pass_and_shoot_with_keeper) echo "grf_pass" ;;
    academy_3_vs_1_with_keeper) echo "grf_3v1" ;;
    academy_counterattack_easy) echo "grf_counter" ;;
    corridor) echo "corridor" ;;
    *) echo "$1" ;;
  esac
}

common_args() {
  printf '%s' "$EXTRA_ARGS torch_num_threads=$TORCH_NUM_THREADS torch_num_interop_threads=$TORCH_NUM_INTEROP_THREADS learner_updates_per_collect=$LEARNER_UPDATES_PER_COLLECT clean_semantic_router_ema=$ROUTER_EMA clean_semantic_router_threshold=$ROUTER_THRESHOLD clean_semantic_router_temperature=$ROUTER_TEMPERATURE clean_semantic_router_warmup_steps=$ROUTER_WARMUP_STEPS clean_semantic_router_freeze_steps=$ROUTER_FREEZE_STEPS clean_relation_teacher_td_coef=0.0 clean_relation_distill_coef=0.0 clean_smooth_head_loss_coef=0.0 clean_action_pred_loss_coef=0.0 clean_public_delta_loss_coef=0.0 save_battle_trace=False"
}

submit_job() {
  local dependency="$1"
  local script="$2"
  local job_name="$3"
  local run_name="$4"
  local env_config="$5"
  local map_name="$6"
  local model_type="$7"
  local cpus="$8"
  local memory="$9"
  local source_job_id="${10:-}"
  local scenario_extra_args="${11:-}"
  local sbatch_args=(
    --parsable
    --nodes=1
    --ntasks=1
    --cpus-per-task="$cpus"
    --mem="$memory"
    --time="$TIME"
    --job-name="$job_name"
    --output=ozstar_logs/%x_%j.out
    --error=ozstar_logs/%x_%j.err
  )

  if [[ -n "$dependency" ]]; then
    sbatch_args+=(--dependency="$dependency")
  fi

  sbatch "${sbatch_args[@]}" \
    --export=ALL,CONFIG=clean_hyper,ENV_CONFIG="$env_config",MAP_NAME="$map_name",MODEL_TYPE="$model_type",SEED="$SEED",T_MAX="$T_MAX",TEST_INTERVAL="$TEST_INTERVAL",BATCH_SIZE_RUN="$BATCH_SIZE_RUN",BATCH_SIZE="$BATCH_SIZE",BUFFER_SIZE="$BUFFER_SIZE",USE_WANDB="$USE_WANDB",WANDB_MODE="$WANDB_MODE",USE_CUDA="$USE_CUDA",RUN_NAME="$run_name",GROUP_NAME="$run_name",SOURCE_JOB_ID="$source_job_id",OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,EXTRA_ARGS="$(common_args) $scenario_extra_args" \
    "$script"
}

echo "== Semantic routing suite: 4 scenes x 2 variants =="
echo "resources: GRF=32c/10G, corridor=32c/40G"
echo "training: time=$TIME t_max=$T_MAX env_workers=$BATCH_SIZE_RUN learner_updates=$LEARNER_UPDATES_PER_COLLECT"

for scenario in $SCENARIOS; do
  tag=$(scenario_tag "$scenario")
  if [[ "$scenario" == "corridor" ]]; then
    env_config="sc2"
    map_name="corridor"
    cpus=32
    memory="40G"
    shared_model="rpg_simple_bias_gradient_importance_shared_field_router_hypercond"
    fixed_model="rpg_simple_bias_gradient_importance_fixed_mask_router_hypercond"
    scenario_extra_args=""
  else
    env_config="$scenario"
    map_name="$scenario"
    cpus=32
    memory="10G"
    shared_model="grf_abs_simple_bias_gradient_importance_shared_field_router_hypercond"
    fixed_model="grf_abs_simple_bias_gradient_importance_fixed_mask_router_hypercond"
    scenario_extra_args="env_worker_startup_stagger=$ENV_WORKER_STARTUP_STAGGER env_worker_reset_retries=$ENV_WORKER_RESET_RETRIES env_worker_reset_retry_delay=$ENV_WORKER_RESET_RETRY_DELAY env_worker_response_timeout=$ENV_WORKER_RESPONSE_TIMEOUT env_args.write_video=False"
  fi

  shared_run="${tag}_gimp_shared_s${SEED}"
  shared_job=$(submit_job "" scripts/ozstar_train_offline.sbatch \
    "${tag}_gshr_s${SEED}" "$shared_run" "$env_config" "$map_name" \
    "$shared_model" "$cpus" "$memory" "" "$scenario_extra_args")
  shared_job="${shared_job%%;*}"
  echo "submitted discovery: scenario=$scenario job=$shared_job run=$shared_run"

  fixed_run="${tag}_gimp_fixedmask_s${SEED}"
  fixed_job=$(submit_job "afterany:$shared_job" \
    scripts/ozstar_train_fixed_semantic_mask.sbatch \
    "${tag}_gfix_s${SEED}" "$fixed_run" "$env_config" "$map_name" \
    "$fixed_model" "$cpus" "$memory" "$shared_job" "$scenario_extra_args")
  fixed_job="${fixed_job%%;*}"
  echo "submitted compact:   scenario=$scenario job=$fixed_job after=$shared_job run=$fixed_run"
done
