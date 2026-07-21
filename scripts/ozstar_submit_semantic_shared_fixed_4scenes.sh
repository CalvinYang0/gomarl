#!/bin/bash
set -euo pipefail

# Submit exactly eight independent experiments: a shared-field routing run and
# a fixed-mask compact run for each of three GRF scenarios plus SMAC corridor.
#
# Fixed-mask is a controlled ablation of the ordinary (non-shared-field)
# gradient-importance router. Its mask must therefore be supplied explicitly;
# it must never be learned by or depend on the shared-field run in this suite.

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl}"
PYTHON_BIN="${PYTHON_BIN:-/home/kyang/.conda/envs/marl_cpu/bin/python}"
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

# Binary masks exported from ordinary gradient-importance routing runs. Use
# scripts/show_semantic_slot_routes.py <job-id> --flat-mask to obtain them.
GRF_PASS_FIXED_MASK="${GRF_PASS_FIXED_MASK:-}"
GRF_3V1_FIXED_MASK="${GRF_3V1_FIXED_MASK:-}"
GRF_COUNTER_FIXED_MASK="${GRF_COUNTER_FIXED_MASK:-}"
CORRIDOR_FIXED_MASK="${CORRIDOR_FIXED_MASK:-}"
GRF_PASS_GIMP_SOURCE_JOB_ID="${GRF_PASS_GIMP_SOURCE_JOB_ID:-}"
GRF_3V1_GIMP_SOURCE_JOB_ID="${GRF_3V1_GIMP_SOURCE_JOB_ID:-}"
GRF_COUNTER_GIMP_SOURCE_JOB_ID="${GRF_COUNTER_GIMP_SOURCE_JOB_ID:-}"
CORRIDOR_GIMP_SOURCE_JOB_ID="${CORRIDOR_GIMP_SOURCE_JOB_ID:-}"

SCENARIOS="${SCENARIOS:-academy_pass_and_shoot_with_keeper academy_3_vs_1_with_keeper academy_counterattack_easy corridor}"
VARIANTS="${VARIANTS:-shared fixed}"

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
  local job_name="$1"
  local run_name="$2"
  local env_config="$3"
  local map_name="$4"
  local model_type="$5"
  local cpus="$6"
  local memory="$7"
  local model_extra_args="${8:-}"
  local scenario_extra_args="${9:-}"
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

  sbatch "${sbatch_args[@]}" \
    --export=ALL,CONFIG=clean_hyper,ENV_CONFIG="$env_config",MAP_NAME="$map_name",MODEL_TYPE="$model_type",SEED="$SEED",T_MAX="$T_MAX",TEST_INTERVAL="$TEST_INTERVAL",BATCH_SIZE_RUN="$BATCH_SIZE_RUN",BATCH_SIZE="$BATCH_SIZE",BUFFER_SIZE="$BUFFER_SIZE",USE_WANDB="$USE_WANDB",WANDB_MODE="$WANDB_MODE",USE_CUDA="$USE_CUDA",RUN_NAME="$run_name",GROUP_NAME="$run_name",OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,EXTRA_ARGS="$(common_args) $model_extra_args $scenario_extra_args" \
    scripts/ozstar_train_offline.sbatch
}

fixed_mask_for_scenario() {
  case "$1" in
    academy_pass_and_shoot_with_keeper) printf '%s' "$GRF_PASS_FIXED_MASK" ;;
    academy_3_vs_1_with_keeper) printf '%s' "$GRF_3V1_FIXED_MASK" ;;
    academy_counterattack_easy) printf '%s' "$GRF_COUNTER_FIXED_MASK" ;;
    corridor) printf '%s' "$CORRIDOR_FIXED_MASK" ;;
    *) return 1 ;;
  esac
}

ordinary_gimp_model_for_scenario() {
  if [[ "$1" == "corridor" ]]; then
    printf '%s' "rpg_simple_bias_gradient_importance_router_hypercond"
  else
    printf '%s' "grf_abs_simple_bias_gradient_importance_router_hypercond"
  fi
}

source_job_for_scenario() {
  case "$1" in
    academy_pass_and_shoot_with_keeper) printf '%s' "$GRF_PASS_GIMP_SOURCE_JOB_ID" ;;
    academy_3_vs_1_with_keeper) printf '%s' "$GRF_3V1_GIMP_SOURCE_JOB_ID" ;;
    academy_counterattack_easy) printf '%s' "$GRF_COUNTER_GIMP_SOURCE_JOB_ID" ;;
    corridor) printf '%s' "$CORRIDOR_GIMP_SOURCE_JOB_ID" ;;
    *) return 1 ;;
  esac
}

extract_mask_from_job() {
  local job_id="$1"
  local output mask
  output=$("$PYTHON_BIN" scripts/show_semantic_slot_routes.py \
    "$job_id" --log-dir ozstar_logs --flat-mask 2>/dev/null) || return 1
  mask=$(printf '%s\n' "$output" | tail -n 1)
  [[ "$mask" =~ ^[01]+$ ]] || return 1
  printf '%s' "$mask"
}

resolve_fixed_mask() {
  local scenario="$1"
  local explicit source_job model log_file job_id mask

  explicit=$(fixed_mask_for_scenario "$scenario")
  if [[ "$explicit" =~ ^[01]+$ ]]; then
    printf '%s' "$explicit"
    return 0
  fi

  source_job=$(source_job_for_scenario "$scenario")
  if [[ -n "$source_job" ]]; then
    mask=$(extract_mask_from_job "$source_job") || {
      echo "ERROR: could not extract an ordinary GIMP mask from job $source_job." >&2
      return 1
    }
    echo "resolved fixed mask: scenario=$scenario source_job=$source_job" >&2
    printf '%s' "$mask"
    return 0
  fi

  model=$(ordinary_gimp_model_for_scenario "$scenario")
  while IFS= read -r log_file; do
    grep -qF "model: $model" "$log_file" || continue
    grep -qF "map: $scenario" "$log_file" || continue
    job_id="${log_file%.out}"
    job_id="${job_id##*_}"
    [[ "$job_id" =~ ^[0-9]+$ ]] || continue
    mask=$(extract_mask_from_job "$job_id") || continue
    echo "resolved fixed mask: scenario=$scenario source_job=$job_id" >&2
    printf '%s' "$mask"
    return 0
  done < <(find ozstar_logs -maxdepth 1 -type f -name '*.out' -print0 \
    | xargs -0 -r ls -1t 2>/dev/null)

  return 1
}

set_fixed_mask_for_scenario() {
  local scenario="$1"
  local mask="$2"
  case "$scenario" in
    academy_pass_and_shoot_with_keeper) GRF_PASS_FIXED_MASK="$mask" ;;
    academy_3_vs_1_with_keeper) GRF_3V1_FIXED_MASK="$mask" ;;
    academy_counterattack_easy) GRF_COUNTER_FIXED_MASK="$mask" ;;
    corridor) CORRIDOR_FIXED_MASK="$mask" ;;
    *) return 1 ;;
  esac
}

variant_enabled() {
  [[ " $VARIANTS " == *" $1 "* ]]
}

for variant in $VARIANTS; do
  if [[ "$variant" != "shared" && "$variant" != "fixed" ]]; then
    echo "ERROR: unsupported variant '$variant'; use shared and/or fixed." >&2
    exit 2
  fi
done

# Validate every mask before submitting anything, avoiding a partially
# submitted suite when one scenario was omitted or copied incorrectly.
if variant_enabled fixed; then
  for scenario in $SCENARIOS; do
    if ! fixed_mask=$(resolve_fixed_mask "$scenario"); then
      tag=$(scenario_tag "$scenario")
      echo "ERROR: no ordinary GIMP mask is available for $scenario ($tag)." >&2
      echo "Run ordinary non-shared GIMP first, or set its *_GIMP_SOURCE_JOB_ID." >&2
      exit 2
    fi
    set_fixed_mask_for_scenario "$scenario" "$fixed_mask"
  done
fi

echo "== Semantic routing suite: 4 scenes =="
echo "variants: $VARIANTS"
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

  if variant_enabled shared; then
    shared_run="${tag}_gimp_shared_s${SEED}"
    shared_job=$(submit_job "${tag}_gshr_s${SEED}" "$shared_run" \
      "$env_config" "$map_name" "$shared_model" "$cpus" "$memory" \
      "" "$scenario_extra_args")
    shared_job="${shared_job%%;*}"
    echo "submitted shared:  scenario=$scenario job=$shared_job run=$shared_run"
  fi

  if variant_enabled fixed; then
    fixed_mask=$(fixed_mask_for_scenario "$scenario")
    fixed_run="${tag}_gimp_fixedmask_s${SEED}"
    fixed_job=$(submit_job "${tag}_gfix_s${SEED}" "$fixed_run" \
      "$env_config" "$map_name" "$fixed_model" "$cpus" "$memory" \
      "clean_semantic_router_fixed_mask=$fixed_mask" "$scenario_extra_args")
    fixed_job="${fixed_job%%;*}"
    echo "submitted compact: scenario=$scenario job=$fixed_job run=$fixed_run"
  fi
done
