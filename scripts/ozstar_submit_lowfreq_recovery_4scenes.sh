#!/bin/bash
set -euo pipefail

# Four-scene comparison:
#   Full: no semantic route.
#   Soft: coupled importance estimation with a low-frequency soft route.
#   Audit: hard route updated from an independent full-input gradient audit.
#
# By default, running Counter/Corridor Full jobs are reused. Pass and 3v1
# receive all three variants, yielding 12 simultaneous jobs when both reusable
# Full jobs are present.

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl}"
PYTHON_BIN="${PYTHON_BIN:-/home/kyang/.conda/envs/marl_cpu/bin/python}"
SEED="${SEED:-1}"
TIME="${TIME:-2-00:00:00}"
T_MAX="${T_MAX:-10050000}"
TEST_INTERVAL="${TEST_INTERVAL:-50000}"

CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
GRF_MEM="${GRF_MEM:-12G}"
CORRIDOR_MEM="${CORRIDOR_MEM:-40G}"
BATCH_SIZE_RUN="${BATCH_SIZE_RUN:-8}"
BATCH_SIZE="${BATCH_SIZE:-128}"
BUFFER_SIZE="${BUFFER_SIZE:-5000}"
LEARNER_UPDATES_PER_COLLECT="${LEARNER_UPDATES_PER_COLLECT:-1}"
TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-$CPUS_PER_TASK}"
TORCH_NUM_INTEROP_THREADS="${TORCH_NUM_INTEROP_THREADS:-1}"

ROUTER_UPDATE_INTERVAL="${ROUTER_UPDATE_INTERVAL:-8000}"
ROUTER_AUDIT_INTERVAL="${ROUTER_AUDIT_INTERVAL:-8000}"
ROUTER_EMA="${ROUTER_EMA:-0.99}"
ROUTER_EMA_UP="${ROUTER_EMA_UP:-0.5}"
ROUTER_EMA_DOWN="${ROUTER_EMA_DOWN:-0.99}"
ROUTER_THRESHOLD="${ROUTER_THRESHOLD:-0.5}"
ROUTER_TEMPERATURE="${ROUTER_TEMPERATURE:-0.1}"
ROUTER_WARMUP_STEPS="${ROUTER_WARMUP_STEPS:-250000}"
ROUTER_FREEZE_STEPS="${ROUTER_FREEZE_STEPS:-20000000}"

ENV_WORKER_STARTUP_STAGGER="${ENV_WORKER_STARTUP_STAGGER:-0.25}"
ENV_WORKER_RESET_RETRIES="${ENV_WORKER_RESET_RETRIES:-3}"
ENV_WORKER_RESET_RETRY_DELAY="${ENV_WORKER_RESET_RETRY_DELAY:-2.0}"
ENV_WORKER_RESPONSE_TIMEOUT="${ENV_WORKER_RESPONSE_TIMEOUT:-180.0}"
USE_WANDB="${USE_WANDB:-True}"
WANDB_MODE="${WANDB_MODE:-offline}"
USE_CUDA="${USE_CUDA:-False}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
SUBMIT_GAP_SECONDS="${SUBMIT_GAP_SECONDS:-1}"
REUSE_RUNNING_FULL="${REUSE_RUNNING_FULL:-1}"

cd "$REPO_DIR"
mkdir -p ozstar_logs

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 2
fi
if (( BATCH_SIZE_RUN > CPUS_PER_TASK )); then
  echo "ERROR: BATCH_SIZE_RUN exceeds CPUS_PER_TASK" >&2
  exit 2
fi
if (( TORCH_NUM_THREADS != CPUS_PER_TASK )); then
  echo "ERROR: TORCH_NUM_THREADS must match CPUS_PER_TASK" >&2
  exit 2
fi

"$PYTHON_BIN" scripts/smoke_test_mlp_drop_relation.py

common_args() {
  printf '%s' "$EXTRA_ARGS torch_num_threads=$TORCH_NUM_THREADS torch_num_interop_threads=$TORCH_NUM_INTEROP_THREADS learner_updates_per_collect=$LEARNER_UPDATES_PER_COLLECT clean_semantic_router_ema=$ROUTER_EMA clean_semantic_router_ema_up=$ROUTER_EMA_UP clean_semantic_router_ema_down=$ROUTER_EMA_DOWN clean_semantic_router_update_interval=$ROUTER_UPDATE_INTERVAL clean_semantic_router_audit_interval=$ROUTER_AUDIT_INTERVAL clean_semantic_router_threshold=$ROUTER_THRESHOLD clean_semantic_router_temperature=$ROUTER_TEMPERATURE clean_semantic_router_warmup_steps=$ROUTER_WARMUP_STEPS clean_semantic_router_freeze_steps=$ROUTER_FREEZE_STEPS clean_relation_teacher_td_coef=0.0 clean_relation_distill_coef=0.0 clean_smooth_head_loss_coef=0.0 clean_action_pred_loss_coef=0.0 clean_public_delta_loss_coef=0.0 save_battle_trace=False"
}

submit_one() {
  local scene="$1"
  local tag="$2"
  local model="$3"
  local variant="$4"
  local memory="$5"
  local env_config map_name scene_args

  if [[ "$scene" == "corridor" ]]; then
    env_config="sc2"
    map_name="corridor"
    scene_args=""
  else
    case "$scene" in
      pass) env_config="academy_pass_and_shoot_with_keeper" ;;
      3v1) env_config="academy_3_vs_1_with_keeper" ;;
      counter) env_config="academy_counterattack_easy" ;;
      *)
        echo "ERROR: unsupported scene: $scene" >&2
        return 2
        ;;
    esac
    map_name="$env_config"
    scene_args="env_worker_startup_stagger=$ENV_WORKER_STARTUP_STAGGER env_worker_reset_retries=$ENV_WORKER_RESET_RETRIES env_worker_reset_retry_delay=$ENV_WORKER_RESET_RETRY_DELAY env_worker_response_timeout=$ENV_WORKER_RESPONSE_TIMEOUT env_args.write_video=False"
  fi

  local run_name="${tag}_mlp_lf8k_${variant}_10m_s${SEED}"
  local job_name="${tag}_lf8k_${variant}_s${SEED}"
  local job_id
  job_id=$(sbatch --parsable \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$memory" \
    --time="$TIME" \
    --job-name="$job_name" \
    --output=ozstar_logs/%x_%j.out \
    --error=ozstar_logs/%x_%j.err \
    --export=ALL,PYTHON_BIN="$PYTHON_BIN",CONFIG=clean_hyper,ENV_CONFIG="$env_config",MAP_NAME="$map_name",MODEL_TYPE="$model",SEED="$SEED",T_MAX="$T_MAX",TEST_INTERVAL="$TEST_INTERVAL",BATCH_SIZE_RUN="$BATCH_SIZE_RUN",EXPECTED_BATCH_SIZE_RUN="$BATCH_SIZE_RUN",BATCH_SIZE="$BATCH_SIZE",BUFFER_SIZE="$BUFFER_SIZE",USE_WANDB="$USE_WANDB",WANDB_MODE="$WANDB_MODE",USE_CUDA="$USE_CUDA",RUN_NAME="$run_name",GROUP_NAME="$run_name",OMP_NUM_THREADS="$TORCH_NUM_THREADS",MKL_NUM_THREADS="$TORCH_NUM_THREADS",OPENBLAS_NUM_THREADS="$TORCH_NUM_THREADS",NUMEXPR_NUM_THREADS="$TORCH_NUM_THREADS",EXTRA_ARGS="$(common_args) $scene_args" \
    scripts/ozstar_train_offline.sbatch)
  printf 'submitted job=%s scene=%s variant=%s model=%s memory=%s\n' \
    "${job_id%%;*}" "$scene" "$variant" "$model" "$memory"
}

find_reusable_full() {
  local exact_name="$1"
  squeue -u "$USER" -h -o "%i|%j|%T" |
    awk -F'|' -v expected="$exact_name" \
      '$2 == expected && ($3 == "RUNNING" || $3 == "PENDING") {print $1; exit}'
}

submitted=0
reuse_or_submit_full() {
  local scene="$1"
  local tag="$2"
  local model="$3"
  local memory="$4"
  local legacy_name="${tag}_mlp_full_s${SEED}"
  local reusable=""

  if [[ "$REUSE_RUNNING_FULL" == "1" ]]; then
    reusable=$(find_reusable_full "$legacy_name")
  fi
  if [[ -n "$reusable" ]]; then
    echo "reused full: job=$reusable scene=$scene name=$legacy_name"
    return
  fi
  if (( submitted > 0 )); then
    sleep "$SUBMIT_GAP_SECONDS"
  fi
  submit_one "$scene" "$tag" "$model" full "$memory"
  submitted=$((submitted + 1))
}

submit_scene() {
  local scene="$1"
  local tag="$2"
  local full_model="$3"
  local soft_model="$4"
  local audit_model="$5"
  local memory="$6"

  reuse_or_submit_full "$scene" "$tag" "$full_model" "$memory"
  if (( submitted > 0 )); then
    sleep "$SUBMIT_GAP_SECONDS"
  fi
  submit_one "$scene" "$tag" "$soft_model" soft "$memory"
  submitted=$((submitted + 1))
  sleep "$SUBMIT_GAP_SECONDS"
  submit_one "$scene" "$tag" "$audit_model" audit "$memory"
  submitted=$((submitted + 1))
}

echo "== Low-frequency semantic recovery: four scenes =="
echo "route: deploy every ${ROUTER_UPDATE_INTERVAL} t_env; audit every ${ROUTER_AUDIT_INTERVAL} t_env"
echo "EMA: fast-up=${ROUTER_EMA_UP}; slow-down=${ROUTER_EMA_DOWN}; no in-run freeze"
echo "resources: GRF=${CPUS_PER_TASK}c/${GRF_MEM}; corridor=${CPUS_PER_TASK}c/${CORRIDOR_MEM}"

submit_scene pass grf_pass \
  grf_abs_mlp_relation_hypercond \
  grf_abs_gimp_lowfreq_soft_mlp_relation_hypercond \
  grf_abs_gimp_lowfreq_audit_mlp_relation_hypercond \
  "$GRF_MEM"

submit_scene 3v1 grf_3v1 \
  grf_abs_mlp_relation_hypercond \
  grf_abs_gimp_lowfreq_soft_mlp_relation_hypercond \
  grf_abs_gimp_lowfreq_audit_mlp_relation_hypercond \
  "$GRF_MEM"

submit_scene counter grf_counter \
  grf_abs_mlp_relation_hypercond \
  grf_abs_gimp_lowfreq_soft_mlp_relation_hypercond \
  grf_abs_gimp_lowfreq_audit_mlp_relation_hypercond \
  "$GRF_MEM"

submit_scene corridor corridor \
  rpg_mlp_relation_hypercond \
  rpg_gimp_lowfreq_soft_mlp_relation_hypercond \
  rpg_gimp_lowfreq_audit_mlp_relation_hypercond \
  "$CORRIDOR_MEM"

echo "newly submitted: $submitted"
echo "Expected active total is 12 when both existing Counter/Corridor Full jobs were reused."
