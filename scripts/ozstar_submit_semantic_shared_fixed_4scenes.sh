#!/bin/bash
set -euo pipefail

# Submit eight independent experiments across three GRF scenarios and corridor:
#   1. adaptive routing shared by slots with the same semantic field;
#   2. two-stage adaptive routing followed by a rebuilt compact network.
#
# The compact variant discovers its mask inside the same Slurm job. It never
# reads a historical mask and never depends on the shared-field experiment.

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl}"
SEED="${SEED:-1}"
TIME="${TIME:-2-00:00:00}"
T_MAX="${T_MAX:-10050000}"
DISCOVERY_T_MAX="${DISCOVERY_T_MAX:-5000000}"
TEST_INTERVAL="${TEST_INTERVAL:-50000}"
# Keep this profile independent from stale generic variables exported by an
# earlier experiment. The measured high-utilization CPU profile uses eight
# rollout workers and a full-allocation learner/controller thread pool.
ENV_WORKERS_PER_JOB="${ENV_WORKERS_PER_JOB:-8}"
BATCH_SIZE="${BATCH_SIZE:-128}"
BUFFER_SIZE="${BUFFER_SIZE:-5000}"
REFERENCE_BATCH_SIZE_RUN="${REFERENCE_BATCH_SIZE_RUN:-8}"
LEARNER_UPDATES_PER_COLLECT="${LEARNER_UPDATES_PER_COLLECT:-$(( (ENV_WORKERS_PER_JOB + REFERENCE_BATCH_SIZE_RUN - 1) / REFERENCE_BATCH_SIZE_RUN ))}"
LEARNER_THREADS_PER_JOB="${LEARNER_THREADS_PER_JOB:-32}"
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
VARIANTS="${VARIANTS:-shared compact}"

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
  printf '%s' "$EXTRA_ARGS torch_num_threads=$LEARNER_THREADS_PER_JOB torch_num_interop_threads=$TORCH_NUM_INTEROP_THREADS learner_updates_per_collect=$LEARNER_UPDATES_PER_COLLECT clean_semantic_router_ema=$ROUTER_EMA clean_semantic_router_threshold=$ROUTER_THRESHOLD clean_semantic_router_temperature=$ROUTER_TEMPERATURE clean_semantic_router_warmup_steps=$ROUTER_WARMUP_STEPS clean_semantic_router_freeze_steps=$ROUTER_FREEZE_STEPS clean_relation_teacher_td_coef=0.0 clean_relation_distill_coef=0.0 clean_smooth_head_loss_coef=0.0 clean_action_pred_loss_coef=0.0 clean_public_delta_loss_coef=0.0 save_battle_trace=False"
}

submit_job() {
  local train_script="$1"
  local job_name="$2"
  local run_name="$3"
  local env_config="$4"
  local map_name="$5"
  local model_type="$6"
  local discovery_model_type="$7"
  local cpus="$8"
  local memory="$9"
  local scenario_extra_args="${10:-}"

  if (( ENV_WORKERS_PER_JOB > cpus )); then
    echo "ERROR: ENV_WORKERS_PER_JOB=$ENV_WORKERS_PER_JOB exceeds allocated CPUs=$cpus." >&2
    return 2
  fi
  if (( LEARNER_THREADS_PER_JOB != cpus )); then
    echo "ERROR: LEARNER_THREADS_PER_JOB=$LEARNER_THREADS_PER_JOB must equal allocated CPUs=$cpus." >&2
    echo "This suite uses the verified full-allocation learner profile." >&2
    return 2
  fi

  sbatch --parsable \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$cpus" \
    --mem="$memory" \
    --time="$TIME" \
    --job-name="$job_name" \
    --output=ozstar_logs/%x_%j.out \
    --error=ozstar_logs/%x_%j.err \
    --export=ALL,CONFIG=clean_hyper,ENV_CONFIG="$env_config",MAP_NAME="$map_name",MODEL_TYPE="$model_type",DISCOVERY_MODEL_TYPE="$discovery_model_type",DISCOVERY_T_MAX="$DISCOVERY_T_MAX",SEED="$SEED",T_MAX="$T_MAX",TEST_INTERVAL="$TEST_INTERVAL",BATCH_SIZE_RUN="$ENV_WORKERS_PER_JOB",EXPECTED_BATCH_SIZE_RUN="$ENV_WORKERS_PER_JOB",BATCH_SIZE="$BATCH_SIZE",BUFFER_SIZE="$BUFFER_SIZE",USE_WANDB="$USE_WANDB",WANDB_MODE="$WANDB_MODE",USE_CUDA="$USE_CUDA",RUN_NAME="$run_name",GROUP_NAME="$run_name",OMP_NUM_THREADS="$LEARNER_THREADS_PER_JOB",MKL_NUM_THREADS="$LEARNER_THREADS_PER_JOB",OPENBLAS_NUM_THREADS="$LEARNER_THREADS_PER_JOB",NUMEXPR_NUM_THREADS="$LEARNER_THREADS_PER_JOB",EXTRA_ARGS="$(common_args) $scenario_extra_args" \
    "$train_script"
}

variant_enabled() {
  [[ " $VARIANTS " == *" $1 "* ]]
}

for variant in $VARIANTS; do
  if [[ "$variant" != "shared" && "$variant" != "compact" ]]; then
    echo "ERROR: unsupported variant '$variant'; use shared and/or compact." >&2
    exit 2
  fi
done

echo "== Adaptive semantic routing suite: 4 scenes =="
echo "variants: $VARIANTS"
echo "resources: GRF=32c/10G, corridor=32c/40G"
echo "cpu profile: ${ENV_WORKERS_PER_JOB} rollout workers; ${LEARNER_THREADS_PER_JOB}-thread learner; ${LEARNER_UPDATES_PER_COLLECT} update(s)/collect"
echo "shared: adaptive field-shared routing for $T_MAX steps"
echo "compact: adaptive discovery for $DISCOVERY_T_MAX steps, then rebuilt compact training for $T_MAX steps"

for scenario in $SCENARIOS; do
  tag=$(scenario_tag "$scenario")
  if [[ "$scenario" == "corridor" ]]; then
    env_config="sc2"
    map_name="corridor"
    cpus=32
    memory="40G"
    discovery_model="rpg_simple_bias_gradient_importance_router_hypercond"
    shared_model="rpg_simple_bias_gradient_importance_shared_field_router_hypercond"
    compact_model="rpg_simple_bias_gradient_importance_fixed_mask_router_hypercond"
    scenario_extra_args=""
  else
    env_config="$scenario"
    map_name="$scenario"
    cpus=32
    memory="10G"
    discovery_model="grf_abs_simple_bias_gradient_importance_router_hypercond"
    shared_model="grf_abs_simple_bias_gradient_importance_shared_field_router_hypercond"
    compact_model="grf_abs_simple_bias_gradient_importance_fixed_mask_router_hypercond"
    scenario_extra_args="env_worker_startup_stagger=$ENV_WORKER_STARTUP_STAGGER env_worker_reset_retries=$ENV_WORKER_RESET_RETRIES env_worker_reset_retry_delay=$ENV_WORKER_RESET_RETRY_DELAY env_worker_response_timeout=$ENV_WORKER_RESPONSE_TIMEOUT env_args.write_video=False"
  fi

  if variant_enabled shared; then
    shared_run="${tag}_gimp_shared_s${SEED}"
    shared_job=$(submit_job scripts/ozstar_train_offline.sbatch \
      "${tag}_gshr_s${SEED}" "$shared_run" "$env_config" "$map_name" \
      "$shared_model" "" "$cpus" "$memory" "$scenario_extra_args")
    shared_job="${shared_job%%;*}"
    echo "submitted shared:  scenario=$scenario job=$shared_job run=$shared_run"
  fi

  if variant_enabled compact; then
    compact_run="${tag}_gimp_adaptive_compact_s${SEED}"
    compact_job=$(submit_job scripts/ozstar_train_two_stage_semantic_mask.sbatch \
      "${tag}_gcmp_s${SEED}" "$compact_run" "$env_config" "$map_name" \
      "$compact_model" "$discovery_model" "$cpus" "$memory" \
      "$scenario_extra_args")
    compact_job="${compact_job%%;*}"
    echo "submitted compact: scenario=$scenario job=$compact_job run=$compact_run"
  fi
done
