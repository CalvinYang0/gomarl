#!/bin/bash
set -euo pipefail

# Controlled exploratory hard-gate experiment:
#   gradient importance -> keep probability -> sampled 0/1 observation mask.
# Only the gate changes; all scenes retain the proven MLP/hypernetwork setting.

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl}"
PYTHON_BIN="${PYTHON_BIN:-/home/kyang/.conda/envs/marl_cpu/bin/python}"
RUN_SMOKE_TEST="${RUN_SMOKE_TEST:-True}"
SEED="${SEED:-1}"
TIME="${TIME:-1-00:00:00}"
T_MAX="${T_MAX:-10050000}"
TEST_INTERVAL="${TEST_INTERVAL:-50000}"

CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
GRF_PASS_MEM="${GRF_PASS_MEM:-12G}"
GRF_COUNTER_MEM="${GRF_COUNTER_MEM:-16G}"
CORRIDOR_MEM="${CORRIDOR_MEM:-40G}"
BATCH_SIZE_RUN="${BATCH_SIZE_RUN:-8}"
BATCH_SIZE="${BATCH_SIZE:-128}"
BUFFER_SIZE="${BUFFER_SIZE:-5000}"
LEARNER_UPDATES_PER_COLLECT="${LEARNER_UPDATES_PER_COLLECT:-1}"
TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-$CPUS_PER_TASK}"
TORCH_NUM_INTEROP_THREADS="${TORCH_NUM_INTEROP_THREADS:-1}"

ROUTER_UPDATE_INTERVAL="${ROUTER_UPDATE_INTERVAL:-8000}"
ROUTER_EMA="${ROUTER_EMA:-0.99}"
ROUTER_EMA_UP="${ROUTER_EMA_UP:-0.5}"
ROUTER_EMA_DOWN="${ROUTER_EMA_DOWN:-0.99}"
ROUTER_THRESHOLD="${ROUTER_THRESHOLD:-0.5}"
ROUTER_TEMPERATURE="${ROUTER_TEMPERATURE:-0.1}"
ROUTER_WARMUP_STEPS="${ROUTER_WARMUP_STEPS:-250000}"
ROUTER_FREEZE_STEPS="${ROUTER_FREEZE_STEPS:-20000000}"
EXPLORATION_FLOOR="${EXPLORATION_FLOOR:-0.05}"

ENV_WORKER_STARTUP_STAGGER="${ENV_WORKER_STARTUP_STAGGER:-0.25}"
ENV_WORKER_RESET_RETRIES="${ENV_WORKER_RESET_RETRIES:-3}"
ENV_WORKER_RESET_RETRY_DELAY="${ENV_WORKER_RESET_RETRY_DELAY:-2.0}"
ENV_WORKER_RESPONSE_TIMEOUT="${ENV_WORKER_RESPONSE_TIMEOUT:-180.0}"
USE_WANDB="${USE_WANDB:-True}"
WANDB_MODE="${WANDB_MODE:-offline}"
USE_CUDA="${USE_CUDA:-False}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
SUBMIT_GAP_SECONDS="${SUBMIT_GAP_SECONDS:-1}"

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

if [[ "$RUN_SMOKE_TEST" == "True" ]]; then
  "$PYTHON_BIN" scripts/smoke_test_stochastic_hard_gate.py
fi

common_args() {
  printf '%s' "$EXTRA_ARGS torch_num_threads=$TORCH_NUM_THREADS torch_num_interop_threads=$TORCH_NUM_INTEROP_THREADS learner_updates_per_collect=$LEARNER_UPDATES_PER_COLLECT clean_semantic_router_ema=$ROUTER_EMA clean_semantic_router_ema_up=$ROUTER_EMA_UP clean_semantic_router_ema_down=$ROUTER_EMA_DOWN clean_semantic_router_update_interval=$ROUTER_UPDATE_INTERVAL clean_semantic_router_threshold=$ROUTER_THRESHOLD clean_semantic_router_temperature=$ROUTER_TEMPERATURE clean_semantic_router_warmup_steps=$ROUTER_WARMUP_STEPS clean_semantic_router_freeze_steps=$ROUTER_FREEZE_STEPS clean_semantic_stochastic_exploration_floor=$EXPLORATION_FLOOR clean_relation_teacher_td_coef=0.0 clean_relation_distill_coef=0.0 clean_smooth_head_loss_coef=0.0 clean_action_pred_loss_coef=0.0 clean_public_delta_loss_coef=0.0 save_battle_trace=False"
}

active_job() {
  local exact_name="$1"
  squeue -u "$USER" -h -o "%i|%j|%T" |
    awk -F'|' -v expected="$exact_name" \
      '$2 == expected && ($3 == "RUNNING" || $3 == "PENDING") {print $1; exit}'
}

submit_one() {
  local scene="$1"
  local model memory env_config map_name scene_args tag

  case "$scene" in
    pass)
      tag="grf_pass"
      env_config="academy_pass_and_shoot_with_keeper"
      map_name="$env_config"
      model="grf_abs_gimp_lowfreq_stochastic_hard_mlp_relation_hypercond"
      memory="$GRF_PASS_MEM"
      scene_args="env_worker_startup_stagger=$ENV_WORKER_STARTUP_STAGGER env_worker_reset_retries=$ENV_WORKER_RESET_RETRIES env_worker_reset_retry_delay=$ENV_WORKER_RESET_RETRY_DELAY env_worker_response_timeout=$ENV_WORKER_RESPONSE_TIMEOUT env_args.write_video=False"
      ;;
    counter)
      tag="grf_counter"
      env_config="academy_counterattack_easy"
      map_name="$env_config"
      model="grf_abs_gimp_lowfreq_stochastic_hard_mlp_relation_hypercond"
      memory="$GRF_COUNTER_MEM"
      scene_args="env_worker_startup_stagger=$ENV_WORKER_STARTUP_STAGGER env_worker_reset_retries=$ENV_WORKER_RESET_RETRIES env_worker_reset_retry_delay=$ENV_WORKER_RESET_RETRY_DELAY env_worker_response_timeout=$ENV_WORKER_RESPONSE_TIMEOUT env_args.write_video=False"
      ;;
    corridor)
      tag="corridor"
      env_config="sc2"
      map_name="corridor"
      model="rpg_gimp_lowfreq_stochastic_hard_mlp_relation_hypercond"
      memory="$CORRIDOR_MEM"
      scene_args=""
      ;;
    *)
      echo "ERROR: unsupported scene: $scene" >&2
      return 2
      ;;
  esac

  local run_name="${tag}_mlp_explore_hard_10m_s${SEED}"
  local job_name="${tag}_mlp_ehard_s${SEED}"
  local existing job_id
  existing=$(active_job "$job_name")
  if [[ -n "$existing" ]]; then
    echo "reused active job=$existing name=$job_name"
    return
  fi

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
  printf 'submitted job=%s scene=%s model=%s resources=%sc/%s\n' \
    "${job_id%%;*}" "$scene" "$model" "$CPUS_PER_TASK" "$memory"
}

echo "== Stochastic hard MLP gate: Counter + Pass + Corridor =="
echo "route: update=${ROUTER_UPDATE_INTERVAL}, ema_up=${ROUTER_EMA_UP}, ema_down=${ROUTER_EMA_DOWN}, exploration_floor=${EXPLORATION_FLOOR}"
echo "resources: GRF Pass=${CPUS_PER_TASK}c/${GRF_PASS_MEM}; GRF Counter=${CPUS_PER_TASK}c/${GRF_COUNTER_MEM}; Corridor=${CPUS_PER_TASK}c/${CORRIDOR_MEM}"

submit_one counter
sleep "$SUBMIT_GAP_SECONDS"
submit_one pass
sleep "$SUBMIT_GAP_SECONDS"
submit_one corridor
