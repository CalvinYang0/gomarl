#!/bin/bash
set -euo pipefail

# Compare three ways of consuming the same gradient-importance semantic score:
#   film  : TOKEN above a learned threshold; the complement applies FiLM.
#   hdrop : learned TOKEN / BIAS / DROP thresholds.
#   str   : soft-threshold TOKEN gates with exact zeros and a sparsity penalty.
#
# Jobs are submitted counterattack first so those three receive the earliest
# queue timestamps, followed by pass-and-shoot and 3-vs-1.

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl}"
PYTHON_BIN="${PYTHON_BIN:-/home/kyang/.conda/envs/marl_cpu/bin/python}"
SEEDS="${SEEDS:-1}"
TIME="${TIME:-2-00:00:00}"
T_MAX="${T_MAX:-10050000}"
TEST_INTERVAL="${TEST_INTERVAL:-50000}"

# This is the measured high-utilization GRF profile: eight rollout workers and
# a full 32-thread learner/controller pool. Prior runs used about 8.6 GiB, so
# 10 GiB leaves allocator headroom without substantially over-requesting RAM.
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
MEM="${MEM:-10G}"
BATCH_SIZE_RUN="${BATCH_SIZE_RUN:-8}"
BATCH_SIZE="${BATCH_SIZE:-128}"
BUFFER_SIZE="${BUFFER_SIZE:-5000}"
REFERENCE_BATCH_SIZE_RUN="${REFERENCE_BATCH_SIZE_RUN:-8}"
LEARNER_UPDATES_PER_COLLECT="${LEARNER_UPDATES_PER_COLLECT:-$(( (BATCH_SIZE_RUN + REFERENCE_BATCH_SIZE_RUN - 1) / REFERENCE_BATCH_SIZE_RUN ))}"
TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-$CPUS_PER_TASK}"
TORCH_NUM_INTEROP_THREADS="${TORCH_NUM_INTEROP_THREADS:-1}"

ROUTER_EMA="${ROUTER_EMA:-0.99}"
ROUTER_THRESHOLD="${ROUTER_THRESHOLD:-0.5}"
ROUTER_DROP_THRESHOLD="${ROUTER_DROP_THRESHOLD:-0.35}"
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

SCENARIOS="${SCENARIOS:-academy_counterattack_easy academy_pass_and_shoot_with_keeper academy_3_vs_1_with_keeper}"
MODELS="${MODELS:-grf_abs_simple_bias_gimp_lthr_film_hypercond grf_abs_simple_bias_gimp_lthr_hdrop_hypercond grf_abs_simple_bias_gimp_str_sparse_hypercond}"

cd "$REPO_DIR"
mkdir -p ozstar_logs

if [[ "${SKIP_SMOKE_TEST:-0}" != "1" ]]; then
  echo "== Running GRF semantic-use smoke test =="
  "$PYTHON_BIN" scripts/smoke_test_grf_semantic_use_variants.py
fi

if (( BATCH_SIZE_RUN > CPUS_PER_TASK )); then
  echo "ERROR: BATCH_SIZE_RUN=$BATCH_SIZE_RUN exceeds CPUS_PER_TASK=$CPUS_PER_TASK" >&2
  exit 2
fi
if (( TORCH_NUM_THREADS != CPUS_PER_TASK )); then
  echo "ERROR: TORCH_NUM_THREADS=$TORCH_NUM_THREADS must equal CPUS_PER_TASK=$CPUS_PER_TASK" >&2
  echo "This suite intentionally uses the verified full-allocation CPU profile." >&2
  exit 2
fi

scenario_tag() {
  case "$1" in
    academy_counterattack_easy) echo "counter" ;;
    academy_pass_and_shoot_with_keeper) echo "pass" ;;
    academy_3_vs_1_with_keeper) echo "3v1" ;;
    *) echo "grf" ;;
  esac
}

model_tag() {
  case "$1" in
    grf_abs_simple_bias_gimp_lthr_film_hypercond) echo "gfilm" ;;
    grf_abs_simple_bias_gimp_lthr_hdrop_hypercond) echo "ghdrop" ;;
    grf_abs_simple_bias_gimp_str_sparse_hypercond) echo "gstr" ;;
    *) echo "semantic" ;;
  esac
}

common_args() {
  printf '%s' "$EXTRA_ARGS torch_num_threads=$TORCH_NUM_THREADS torch_num_interop_threads=$TORCH_NUM_INTEROP_THREADS learner_updates_per_collect=$LEARNER_UPDATES_PER_COLLECT env_worker_startup_stagger=$ENV_WORKER_STARTUP_STAGGER env_worker_reset_retries=$ENV_WORKER_RESET_RETRIES env_worker_reset_retry_delay=$ENV_WORKER_RESET_RETRY_DELAY env_worker_response_timeout=$ENV_WORKER_RESPONSE_TIMEOUT env_args.write_video=False clean_semantic_router_ema=$ROUTER_EMA clean_semantic_router_threshold=$ROUTER_THRESHOLD clean_semantic_router_keep_threshold=$ROUTER_DROP_THRESHOLD clean_semantic_router_temperature=$ROUTER_TEMPERATURE clean_semantic_router_warmup_steps=$ROUTER_WARMUP_STEPS clean_semantic_router_freeze_steps=$ROUTER_FREEZE_STEPS clean_semantic_router_sparse_coef=$SPARSE_COEF clean_relation_teacher_td_coef=0.0 clean_relation_distill_coef=0.0 clean_smooth_head_loss_coef=0.0 clean_action_pred_loss_coef=0.0 clean_public_delta_loss_coef=0.0 save_battle_trace=False"
}

echo "== GRF semantic-use ablation: 3 variants x 3 scenes =="
echo "submission order: $SCENARIOS"
echo "models: $MODELS"
echo "resources/job: 1 node, ${CPUS_PER_TASK} CPUs, ${MEM}, ${TIME}"
echo "training: t_max=${T_MAX}, workers=${BATCH_SIZE_RUN}, learner_threads=${TORCH_NUM_THREADS}, updates/collect=${LEARNER_UPDATES_PER_COLLECT}"

submitted=0
for env_config in $SCENARIOS; do
  env_tag="$(scenario_tag "$env_config")"
  for model_type in $MODELS; do
    variant_tag="$(model_tag "$model_type")"
    for seed in $SEEDS; do
      run_name="grf_${env_tag}_${variant_tag}_10m_s${seed}"
      job_name="grf_${env_tag}_${variant_tag}_s${seed}"
      job_id=$(sbatch --parsable \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task="$CPUS_PER_TASK" \
        --mem="$MEM" \
        --time="$TIME" \
        --job-name="$job_name" \
        --output=ozstar_logs/%x_%j.out \
        --error=ozstar_logs/%x_%j.err \
        --export=ALL,PYTHON_BIN="$PYTHON_BIN",CONFIG=clean_hyper,ENV_CONFIG="$env_config",MAP_NAME="$env_config",MODEL_TYPE="$model_type",SEED="$seed",T_MAX="$T_MAX",TEST_INTERVAL="$TEST_INTERVAL",BATCH_SIZE_RUN="$BATCH_SIZE_RUN",EXPECTED_BATCH_SIZE_RUN="$BATCH_SIZE_RUN",BATCH_SIZE="$BATCH_SIZE",BUFFER_SIZE="$BUFFER_SIZE",USE_WANDB="$USE_WANDB",WANDB_MODE="$WANDB_MODE",USE_CUDA="$USE_CUDA",RUN_NAME="$run_name",GROUP_NAME="$run_name",OMP_NUM_THREADS="$TORCH_NUM_THREADS",MKL_NUM_THREADS="$TORCH_NUM_THREADS",OPENBLAS_NUM_THREADS="$TORCH_NUM_THREADS",NUMEXPR_NUM_THREADS="$TORCH_NUM_THREADS",EXTRA_ARGS="$(common_args)" \
        scripts/ozstar_train_offline.sbatch)
      job_id="${job_id%%;*}"
      submitted=$((submitted + 1))
      printf 'submitted %d: job=%s scene=%s variant=%s run=%s\n' \
        "$submitted" "$job_id" "$env_config" "$variant_tag" "$run_name"
      sleep "$SUBMIT_GAP_SECONDS"
    done
  done
done

echo "submitted total: $submitted"
