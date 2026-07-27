#!/bin/bash
set -euo pipefail

# Compare a lightweight two-layer MLP relation encoder with two input-drop
# mechanisms. All variants keep the temporal GRU, hypernetwork and decision
# maker unchanged.

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl}"
PYTHON_BIN="${PYTHON_BIN:-/home/kyang/.conda/envs/marl_cpu/bin/python}"
SEED="${SEED:-1}"
TIME="${TIME:-1-00:00:00}"
T_MAX="${T_MAX:-10050000}"
TEST_INTERVAL="${TEST_INTERVAL:-50000}"

CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
GRF_MEM="${GRF_MEM:-16G}"
CORRIDOR_MEM="${CORRIDOR_MEM:-40G}"
BATCH_SIZE_RUN="${BATCH_SIZE_RUN:-8}"
BATCH_SIZE="${BATCH_SIZE:-128}"
BUFFER_SIZE="${BUFFER_SIZE:-5000}"
REFERENCE_BATCH_SIZE_RUN="${REFERENCE_BATCH_SIZE_RUN:-8}"
LEARNER_UPDATES_PER_COLLECT="${LEARNER_UPDATES_PER_COLLECT:-$(( (BATCH_SIZE_RUN + REFERENCE_BATCH_SIZE_RUN - 1) / REFERENCE_BATCH_SIZE_RUN ))}"
TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-$CPUS_PER_TASK}"
TORCH_NUM_INTEROP_THREADS="${TORCH_NUM_INTEROP_THREADS:-1}"

ROUTER_EMA="${ROUTER_EMA:-0.99}"
ROUTER_THRESHOLD="${ROUTER_THRESHOLD:-0.5}"
ROUTER_TEMPERATURE="${ROUTER_TEMPERATURE:-0.1}"
ROUTER_WARMUP_STEPS="${ROUTER_WARMUP_STEPS:-250000}"
ROUTER_FREEZE_STEPS="${ROUTER_FREEZE_STEPS:-5000000}"
SPARSE_COEF="${SPARSE_COEF:-0.001}"

ENV_WORKER_STARTUP_STAGGER="${ENV_WORKER_STARTUP_STAGGER:-0.25}"
ENV_WORKER_RESET_RETRIES="${ENV_WORKER_RESET_RETRIES:-3}"
ENV_WORKER_RESET_RETRY_DELAY="${ENV_WORKER_RESET_RETRY_DELAY:-2.0}"
ENV_WORKER_RESPONSE_TIMEOUT="${ENV_WORKER_RESPONSE_TIMEOUT:-180.0}"
USE_WANDB="${USE_WANDB:-True}"
WANDB_MODE="${WANDB_MODE:-offline}"
USE_CUDA="${USE_CUDA:-False}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
SUBMIT_GAP_SECONDS="${SUBMIT_GAP_SECONDS:-1}"
SUBMIT_GRF="${SUBMIT_GRF:-1}"
SUBMIT_CORRIDOR="${SUBMIT_CORRIDOR:-1}"
SUBMIT_FULL="${SUBMIT_FULL:-1}"
SUBMIT_HARD_GIMP="${SUBMIT_HARD_GIMP:-1}"
SUBMIT_SOFT_GIMP="${SUBMIT_SOFT_GIMP:-0}"
SUBMIT_L0="${SUBMIT_L0:-1}"
GRF_SCENES="${GRF_SCENES:-counter}"

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
if [[ "$SUBMIT_GRF" != "0" && "$SUBMIT_GRF" != "1" ]]; then
  echo "ERROR: SUBMIT_GRF must be 0 or 1" >&2
  exit 2
fi
if [[ "$SUBMIT_CORRIDOR" != "0" && "$SUBMIT_CORRIDOR" != "1" ]]; then
  echo "ERROR: SUBMIT_CORRIDOR must be 0 or 1" >&2
  exit 2
fi
for toggle_name in SUBMIT_FULL SUBMIT_HARD_GIMP SUBMIT_SOFT_GIMP SUBMIT_L0; do
  toggle_value="${!toggle_name}"
  if [[ "$toggle_value" != "0" && "$toggle_value" != "1" ]]; then
    echo "ERROR: $toggle_name must be 0 or 1" >&2
    exit 2
  fi
done

"$PYTHON_BIN" scripts/smoke_test_mlp_drop_relation.py

common_args() {
  printf '%s' "$EXTRA_ARGS torch_num_threads=$TORCH_NUM_THREADS torch_num_interop_threads=$TORCH_NUM_INTEROP_THREADS learner_updates_per_collect=$LEARNER_UPDATES_PER_COLLECT clean_semantic_router_ema=$ROUTER_EMA clean_semantic_router_threshold=$ROUTER_THRESHOLD clean_semantic_router_temperature=$ROUTER_TEMPERATURE clean_semantic_router_warmup_steps=$ROUTER_WARMUP_STEPS clean_semantic_router_freeze_steps=$ROUTER_FREEZE_STEPS clean_semantic_router_sparse_coef=$SPARSE_COEF clean_relation_teacher_td_coef=0.0 clean_relation_distill_coef=0.0 clean_smooth_head_loss_coef=0.0 clean_action_pred_loss_coef=0.0 clean_public_delta_loss_coef=0.0 save_battle_trace=False"
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
      pass)
        env_config="academy_pass_and_shoot_with_keeper"
        ;;
      3v1)
        env_config="academy_3_vs_1_with_keeper"
        ;;
      counter)
        env_config="academy_counterattack_easy"
        ;;
      *)
        echo "ERROR: unsupported GRF scene: $scene" >&2
        return 2
        ;;
    esac
    map_name="$env_config"
    scene_args="env_worker_startup_stagger=$ENV_WORKER_STARTUP_STAGGER env_worker_reset_retries=$ENV_WORKER_RESET_RETRIES env_worker_reset_retry_delay=$ENV_WORKER_RESET_RETRY_DELAY env_worker_response_timeout=$ENV_WORKER_RESPONSE_TIMEOUT env_args.write_video=False"
  fi

  local run_name="${tag}_mlp_${variant}_10m_s${SEED}"
  local job_name="${tag}_mlp_${variant}_s${SEED}"
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

echo "== Lightweight MLP relation/drop ablation =="
echo "resources: GRF=${CPUS_PER_TASK}c/${GRF_MEM}; corridor=${CPUS_PER_TASK}c/${CORRIDOR_MEM}"
echo "training: ${TIME}, t_max=${T_MAX}, workers=${BATCH_SIZE_RUN}, learner_threads=${TORCH_NUM_THREADS}"
echo "selection: GRF_SCENES='${GRF_SCENES}', full=${SUBMIT_FULL}, hard_gimp=${SUBMIT_HARD_GIMP}, soft_gimp=${SUBMIT_SOFT_GIMP}, l0=${SUBMIT_L0}"

submitted=0
if [[ "$SUBMIT_GRF" == "1" ]]; then
  for scene in $GRF_SCENES; do
    case "$scene" in
      pass) tag="grf_pass" ;;
      3v1) tag="grf_3v1" ;;
      counter) tag="grf_counter" ;;
      *)
        echo "ERROR: unsupported GRF scene in GRF_SCENES: $scene" >&2
        exit 2
        ;;
    esac

    if [[ "$SUBMIT_FULL" == "1" ]]; then
      if (( submitted > 0 )); then
        sleep "$SUBMIT_GAP_SECONDS"
      fi
      submit_one "$scene" "$tag" grf_abs_mlp_relation_hypercond full "$GRF_MEM"
      submitted=$((submitted + 1))
    fi
    if [[ "$SUBMIT_HARD_GIMP" == "1" ]]; then
      if (( submitted > 0 )); then
        sleep "$SUBMIT_GAP_SECONDS"
      fi
      submit_one "$scene" "$tag" grf_abs_gimp_lthr_drop_mlp_relation_hypercond gimp_drop "$GRF_MEM"
      submitted=$((submitted + 1))
    fi
    if [[ "$SUBMIT_SOFT_GIMP" == "1" ]]; then
      if (( submitted > 0 )); then
        sleep "$SUBMIT_GAP_SECONDS"
      fi
      submit_one "$scene" "$tag" grf_abs_gimp_lthr_soft_mlp_relation_hypercond gimp_soft "$GRF_MEM"
      submitted=$((submitted + 1))
    fi
    if [[ "$SUBMIT_L0" == "1" ]]; then
      if (( submitted > 0 )); then
        sleep "$SUBMIT_GAP_SECONDS"
      fi
      submit_one "$scene" "$tag" grf_abs_l0_drop_mlp_relation_hypercond l0_drop "$GRF_MEM"
      submitted=$((submitted + 1))
    fi
  done
fi

if [[ "$SUBMIT_CORRIDOR" == "1" ]]; then
  if (( submitted > 0 )); then
    sleep "$SUBMIT_GAP_SECONDS"
  fi
  if [[ "$SUBMIT_FULL" == "1" ]]; then
    submit_one corridor corridor rpg_mlp_relation_hypercond full "$CORRIDOR_MEM"
    submitted=$((submitted + 1))
  fi
  if [[ "$SUBMIT_HARD_GIMP" == "1" ]]; then
    if (( submitted > 0 )); then
      sleep "$SUBMIT_GAP_SECONDS"
    fi
    submit_one corridor corridor rpg_gimp_lthr_drop_mlp_relation_hypercond gimp_drop "$CORRIDOR_MEM"
    submitted=$((submitted + 1))
  fi
  if [[ "$SUBMIT_SOFT_GIMP" == "1" ]]; then
    if (( submitted > 0 )); then
      sleep "$SUBMIT_GAP_SECONDS"
    fi
    submit_one corridor corridor rpg_gimp_lthr_soft_mlp_relation_hypercond gimp_soft "$CORRIDOR_MEM"
    submitted=$((submitted + 1))
  fi
  if [[ "$SUBMIT_L0" == "1" ]]; then
    if (( submitted > 0 )); then
      sleep "$SUBMIT_GAP_SECONDS"
    fi
    submit_one corridor corridor rpg_l0_drop_mlp_relation_hypercond l0_drop "$CORRIDOR_MEM"
    submitted=$((submitted + 1))
  fi
fi

echo "submitted total: $submitted"
