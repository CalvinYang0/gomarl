#!/bin/bash
set -euo pipefail

# Observation-conditioned c-STG and BayesG dual gates on GRF Counter and SMAC
# Corridor. Defaults to one seed and both scenes (4 jobs total).

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl-dual-branch}"
PYTHON_BIN="${PYTHON_BIN:-/home/kyang/.conda/envs/marl_cpu/bin/python}"
SEEDS="${SEEDS:-1}"
SCENES="${SCENES:-counter corridor}"
TIME="${TIME:-1-00:00:00}"
T_MAX="${T_MAX:-10050000}"
TEST_INTERVAL="${TEST_INTERVAL:-50000}"
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
GRF_MEM="${GRF_MEM:-16G}"
CORRIDOR_MEM="${CORRIDOR_MEM:-64G}"
BATCH_SIZE_RUN="${BATCH_SIZE_RUN:-8}"
BATCH_SIZE="${BATCH_SIZE:-128}"
BUFFER_SIZE="${BUFFER_SIZE:-5000}"
TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-$CPUS_PER_TASK}"
TORCH_NUM_INTEROP_THREADS="${TORCH_NUM_INTEROP_THREADS:-1}"
LEARNER_UPDATES_PER_COLLECT="${LEARNER_UPDATES_PER_COLLECT:-1}"
ENV_WORKER_STARTUP_STAGGER="${ENV_WORKER_STARTUP_STAGGER:-0.25}"
ENV_WORKER_RESET_RETRIES="${ENV_WORKER_RESET_RETRIES:-3}"
ENV_WORKER_RESET_RETRY_DELAY="${ENV_WORKER_RESET_RETRY_DELAY:-2.0}"
ENV_WORKER_RESPONSE_TIMEOUT="${ENV_WORKER_RESPONSE_TIMEOUT:-180.0}"
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

common_args() {
  printf '%s' "$EXTRA_ARGS torch_num_threads=$TORCH_NUM_THREADS torch_num_interop_threads=$TORCH_NUM_INTEROP_THREADS learner_updates_per_collect=$LEARNER_UPDATES_PER_COLLECT clean_relation_teacher_td_coef=0.0 clean_relation_distill_coef=0.0 clean_smooth_head_loss_coef=0.0 clean_action_pred_loss_coef=0.0 clean_public_delta_loss_coef=0.0 save_battle_trace=False"
}

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
  local tag env_config map_name memory model scene_args suffix

  case "$scene" in
    counter)
      tag="grf_counter"
      env_config="academy_counterattack_easy"
      map_name="$env_config"
      memory="$GRF_MEM"
      scene_args="env_worker_startup_stagger=$ENV_WORKER_STARTUP_STAGGER env_worker_reset_retries=$ENV_WORKER_RESET_RETRIES env_worker_reset_retry_delay=$ENV_WORKER_RESET_RETRY_DELAY env_worker_response_timeout=$ENV_WORKER_RESPONSE_TIMEOUT env_args.write_video=False"
      case "$variant" in
        cstg) model="grf_abs_dual_branch_cstg_gate_hypercond" ;;
        bayesg) model="grf_abs_dual_branch_bayesg_gate_hypercond" ;;
      esac
      ;;
    corridor)
      tag="corridor"
      env_config="sc2"
      map_name="corridor"
      memory="$CORRIDOR_MEM"
      scene_args=""
      case "$variant" in
        cstg) model="rpg_dual_branch_cstg_gate_hypercond" ;;
        bayesg) model="rpg_dual_branch_bayesg_gate_hypercond" ;;
      esac
      ;;
    *)
      echo "ERROR: unsupported scene: $scene" >&2
      return 2
      ;;
  esac

  case "$variant" in
    cstg) suffix="cstg" ;;
    bayesg) suffix="bayesg" ;;
    *)
      echo "ERROR: unsupported variant: $variant" >&2
      return 2
      ;;
  esac

  local run_name="${tag}_dual_${suffix}_gate_10m_s${seed}"
  local job_name="${tag}_dual_${suffix}_s${seed}"
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
    --export=ALL,PYTHON_BIN="$PYTHON_BIN",CONFIG=clean_hyper,ENV_CONFIG="$env_config",MAP_NAME="$map_name",MODEL_TYPE="$model",SEED="$seed",T_MAX="$T_MAX",TEST_INTERVAL="$TEST_INTERVAL",BATCH_SIZE_RUN="$BATCH_SIZE_RUN",EXPECTED_BATCH_SIZE_RUN="$BATCH_SIZE_RUN",BATCH_SIZE="$BATCH_SIZE",BUFFER_SIZE="$BUFFER_SIZE",USE_WANDB="$USE_WANDB",WANDB_MODE="$WANDB_MODE",USE_CUDA="$USE_CUDA",RUN_NAME="$run_name",GROUP_NAME="$run_name",OMP_NUM_THREADS="$TORCH_NUM_THREADS",MKL_NUM_THREADS="$TORCH_NUM_THREADS",OPENBLAS_NUM_THREADS="$TORCH_NUM_THREADS",NUMEXPR_NUM_THREADS="$TORCH_NUM_THREADS",EXTRA_ARGS="$(common_args) $scene_args" \
    scripts/ozstar_train_offline.sbatch)
  printf 'submitted job=%s scene=%s variant=%s seed=%s model=%s\n' \
    "${job_id%%;*}" "$scene" "$variant" "$seed" "$model"
}

submitted=0
for scene in $SCENES; do
  for variant in cstg bayesg; do
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
