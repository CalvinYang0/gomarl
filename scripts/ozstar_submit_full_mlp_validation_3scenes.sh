#!/bin/bash
set -euo pipefail

# Clean validation of the same full-observation two-layer MLP condition encoder
# on one GRF task and two SMAC tasks. No semantic routing, Drop, Transformer,
# or auxiliary losses are enabled.

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl}"
PYTHON_BIN="${PYTHON_BIN:-/home/kyang/.conda/envs/marl_cpu/bin/python}"
SEED="${SEED:-1}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
SCENES="${SCENES:-3v1 5m6m 3s5z}"
T_MAX="${T_MAX:-10050000}"
TEST_INTERVAL="${TEST_INTERVAL:-50000}"
USE_WANDB="${USE_WANDB:-True}"
WANDB_MODE="${WANDB_MODE:-offline}"
USE_CUDA="${USE_CUDA:-False}"
FULL_MLP_EXTRA_ARGS="${FULL_MLP_EXTRA_ARGS:-}"
SUBMIT_GAP_SECONDS="${SUBMIT_GAP_SECONDS:-1}"

# GRF: measured high-utilization profile is about 8-9 GiB.
GRF_CPUS="${GRF_CPUS:-32}"
GRF_MEM="${GRF_MEM:-10G}"
GRF_TIME="${GRF_TIME:-2-00:00:00}"
GRF_BATCH_SIZE_RUN="${GRF_BATCH_SIZE_RUN:-8}"
GRF_BATCH_SIZE="${GRF_BATCH_SIZE:-128}"
GRF_BUFFER_SIZE="${GRF_BUFFER_SIZE:-5000}"
GRF_TORCH_THREADS="${GRF_TORCH_THREADS:-32}"
GRF_LEARNER_UPDATES="${GRF_LEARNER_UPDATES:-1}"

# 5m_vs_6m: reuse the profile that previously saturated the 32-core node.
M5_CPUS="${M5_CPUS:-32}"
M5_MEM="${M5_MEM:-12G}"
M5_TIME="${M5_TIME:-1-00:00:00}"
M5_BATCH_SIZE_RUN="${M5_BATCH_SIZE_RUN:-8}"
M5_BATCH_SIZE="${M5_BATCH_SIZE:-32}"
M5_BUFFER_SIZE="${M5_BUFFER_SIZE:-500}"
M5_TORCH_THREADS="${M5_TORCH_THREADS:-32}"
M5_LEARNER_UPDATES="${M5_LEARNER_UPDATES:-1}"

# 3s5z_vs_3s6z: parallel SC2 rollout workers dominate throughput and memory.
S3_CPUS="${S3_CPUS:-32}"
S3_MEM="${S3_MEM:-48G}"
S3_TIME="${S3_TIME:-2-00:00:00}"
S3_BATCH_SIZE_RUN="${S3_BATCH_SIZE_RUN:-32}"
S3_BATCH_SIZE="${S3_BATCH_SIZE:-32}"
S3_BUFFER_SIZE="${S3_BUFFER_SIZE:-500}"
S3_TORCH_THREADS="${S3_TORCH_THREADS:-1}"
S3_LEARNER_UPDATES="${S3_LEARNER_UPDATES:-4}"

TORCH_NUM_INTEROP_THREADS="${TORCH_NUM_INTEROP_THREADS:-1}"
ENV_WORKER_STARTUP_STAGGER="${ENV_WORKER_STARTUP_STAGGER:-0.25}"
ENV_WORKER_RESET_RETRIES="${ENV_WORKER_RESET_RETRIES:-3}"
ENV_WORKER_RESET_RETRY_DELAY="${ENV_WORKER_RESET_RETRY_DELAY:-2.0}"
ENV_WORKER_RESPONSE_TIMEOUT="${ENV_WORKER_RESPONSE_TIMEOUT:-180.0}"

cd "$REPO_DIR"
mkdir -p ozstar_logs

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 2
fi

active_job() {
  local exact_name="$1"
  squeue -u "$USER" -h -o "%i|%j|%T" |
    awk -F'|' -v expected="$exact_name" \
      '$2 == expected && ($3 == "RUNNING" || $3 == "PENDING") {print $1; exit}'
}

validate_profile() {
  local label="$1"
  local cpus="$2"
  local workers="$3"
  local torch_threads="$4"

  if (( workers > cpus )); then
    echo "ERROR: $label workers=$workers exceeds CPUs=$cpus" >&2
    exit 2
  fi
  if (( torch_threads > cpus )); then
    echo "ERROR: $label torch_threads=$torch_threads exceeds CPUs=$cpus" >&2
    exit 2
  fi
}

common_args() {
  local torch_threads="$1"
  local learner_updates="$2"
  printf '%s' \
    "$FULL_MLP_EXTRA_ARGS torch_num_threads=$torch_threads torch_num_interop_threads=$TORCH_NUM_INTEROP_THREADS learner_updates_per_collect=$learner_updates clean_relation_teacher_td_coef=0.0 clean_relation_distill_coef=0.0 clean_smooth_head_loss_coef=0.0 clean_action_pred_loss_coef=0.0 clean_public_delta_loss_coef=0.0 save_battle_trace=False"
}

submit_one() {
  local scene="$1"
  local job_name="$2"
  local run_name="$3"
  local env_config="$4"
  local map_name="$5"
  local model_type="$6"
  local cpus="$7"
  local memory="$8"
  local time_limit="$9"
  local workers="${10}"
  local batch_size="${11}"
  local buffer_size="${12}"
  local torch_threads="${13}"
  local learner_updates="${14}"
  local scene_args="${15:-}"
  local existing job_id

  existing=$(active_job "$job_name")
  if [[ -n "$existing" ]]; then
    echo "reused active job=$existing scene=$scene name=$job_name"
    return
  fi

  job_id=$(sbatch --parsable \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$cpus" \
    --mem="$memory" \
    --time="$time_limit" \
    --job-name="$job_name" \
    --output=ozstar_logs/%x_%j.out \
    --error=ozstar_logs/%x_%j.err \
    --export=ALL,PYTHON_BIN="$PYTHON_BIN",CONFIG=clean_hyper,ENV_CONFIG="$env_config",MAP_NAME="$map_name",MODEL_TYPE="$model_type",SEED="$SEED",T_MAX="$T_MAX",TEST_INTERVAL="$TEST_INTERVAL",BATCH_SIZE_RUN="$workers",EXPECTED_BATCH_SIZE_RUN="$workers",BATCH_SIZE="$batch_size",BUFFER_SIZE="$buffer_size",USE_WANDB="$USE_WANDB",WANDB_MODE="$WANDB_MODE",USE_CUDA="$USE_CUDA",RUN_NAME="$run_name",GROUP_NAME="$run_name",OMP_NUM_THREADS="$torch_threads",MKL_NUM_THREADS="$torch_threads",OPENBLAS_NUM_THREADS="$torch_threads",NUMEXPR_NUM_THREADS="$torch_threads",EXTRA_ARGS="$(common_args "$torch_threads" "$learner_updates") $scene_args" \
    scripts/ozstar_train_offline.sbatch)

  printf 'submitted job=%s scene=%s model=%s resources=%sc/%s/%s workers=%s\n' \
    "${job_id%%;*}" "$scene" "$model_type" "$cpus" "$memory" "$time_limit" "$workers"
}

validate_profile "3v1" "$GRF_CPUS" "$GRF_BATCH_SIZE_RUN" "$GRF_TORCH_THREADS"
validate_profile "5m6m" "$M5_CPUS" "$M5_BATCH_SIZE_RUN" "$M5_TORCH_THREADS"
validate_profile "3s5z" "$S3_CPUS" "$S3_BATCH_SIZE_RUN" "$S3_TORCH_THREADS"

echo "== Full MLP validation: 3v1, 5m6m, 3s5z =="
echo "model: complete local observation -> two-layer MLP -> hypernetwork condition"
echo "disabled: routing, Drop, Transformer, teacher/distillation/smoothing/action/delta losses"
echo "selected scenes: $SCENES"

submitted=0
for scene in $SCENES; do
  if (( submitted > 0 )); then
    sleep "$SUBMIT_GAP_SECONDS"
  fi

  case "$scene" in
    3v1)
      submit_one \
        3v1 \
        "grf_3v1_mlp_fullval_s${SEED}${RUN_SUFFIX}" \
        "grf_3v1_mlp_full_validation_10m_s${SEED}${RUN_SUFFIX}" \
        academy_3_vs_1_with_keeper \
        academy_3_vs_1_with_keeper \
        grf_abs_mlp_relation_hypercond \
        "$GRF_CPUS" "$GRF_MEM" "$GRF_TIME" \
        "$GRF_BATCH_SIZE_RUN" "$GRF_BATCH_SIZE" "$GRF_BUFFER_SIZE" \
        "$GRF_TORCH_THREADS" "$GRF_LEARNER_UPDATES" \
        "env_worker_startup_stagger=$ENV_WORKER_STARTUP_STAGGER env_worker_reset_retries=$ENV_WORKER_RESET_RETRIES env_worker_reset_retry_delay=$ENV_WORKER_RESET_RETRY_DELAY env_worker_response_timeout=$ENV_WORKER_RESPONSE_TIMEOUT env_args.write_video=False"
      ;;
    5m6m)
      submit_one \
        5m6m \
        "5m6m_mlp_fullval_s${SEED}${RUN_SUFFIX}" \
        "5m_vs_6m_mlp_full_validation_10m_s${SEED}${RUN_SUFFIX}" \
        sc2 \
        5m_vs_6m \
        rpg_mlp_relation_hypercond \
        "$M5_CPUS" "$M5_MEM" "$M5_TIME" \
        "$M5_BATCH_SIZE_RUN" "$M5_BATCH_SIZE" "$M5_BUFFER_SIZE" \
        "$M5_TORCH_THREADS" "$M5_LEARNER_UPDATES"
      ;;
    3s5z)
      submit_one \
        3s5z \
        "3s5z_mlp_fullval_s${SEED}${RUN_SUFFIX}" \
        "3s5z_vs_3s6z_mlp_full_validation_10m_s${SEED}${RUN_SUFFIX}" \
        sc2 \
        3s5z_vs_3s6z \
        rpg_mlp_relation_hypercond \
        "$S3_CPUS" "$S3_MEM" "$S3_TIME" \
        "$S3_BATCH_SIZE_RUN" "$S3_BATCH_SIZE" "$S3_BUFFER_SIZE" \
        "$S3_TORCH_THREADS" "$S3_LEARNER_UPDATES"
      ;;
    *)
      echo "ERROR: unsupported scene '$scene'; use: 3v1 5m6m 3s5z" >&2
      exit 2
      ;;
  esac
  submitted=$((submitted + 1))
done

