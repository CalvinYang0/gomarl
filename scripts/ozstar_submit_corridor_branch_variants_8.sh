#!/bin/bash
set -euo pipefail

# Corridor-only suite: four gate/branch variants crossed with parameter
# stability and gradient consistency. Before submitting, cancel only the old
# recovery/adaptive comparison jobs created by the preceding 8-job script.

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl-dual-branch}"
PYTHON_BIN="${PYTHON_BIN:-/home/kyang/.conda/envs/marl_cpu/bin/python}"
SEEDS="${SEEDS:-1}"
TIME="${TIME:-1-00:00:00}"
T_MAX="${T_MAX:-10050000}"
TEST_INTERVAL="${TEST_INTERVAL:-50000}"
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
MEMORY="${MEMORY:-96G}"
AUX_COEF="${AUX_COEF:-0.01}"
ADAPTIVE_TARGET_RATIO="${ADAPTIVE_TARGET_RATIO:-0.1}"
BINARY_CONCRETE_TEMPERATURE="${BINARY_CONCRETE_TEMPERATURE:-0.5}"
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
CANCEL_OLD_SUITE="${CANCEL_OLD_SUITE:-1}"

cd "$REPO_DIR"
mkdir -p ozstar_logs

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 2
fi

"$PYTHON_BIN" scripts/smoke_test_dual_branch_dynamic_gate.py

if [[ "$CANCEL_OLD_SUITE" == "1" ]]; then
  old_job_ids=$(
    squeue -u "$USER" -h -o "%i|%j" |
      awk -F'|' '
        $2 ~ /^(grf_counter|corridor)_(concrete|adaptive10)_(param_stability|grad_consistency)_s[0-9]+$/ {
          print $1
        }
      '
  )
  if [[ -n "$old_job_ids" ]]; then
    echo "cancelling old recovery/adaptive jobs: $old_job_ids"
    # shellcheck disable=SC2086
    scancel $old_job_ids
  else
    echo "no active old recovery/adaptive jobs found"
  fi
fi

active_job() {
  local exact_name="$1"
  squeue -u "$USER" -h -o "%i|%j|%T" |
    awk -F'|' -v expected="$exact_name" \
      '$2 == expected && ($3 == "RUNNING" || $3 == "PENDING") {print $1; exit}'
}

common_args() {
  printf '%s' "$EXTRA_ARGS torch_num_threads=$CPUS_PER_TASK torch_num_interop_threads=$TORCH_NUM_INTEROP_THREADS learner_updates_per_collect=$LEARNER_UPDATES_PER_COLLECT clean_relation_teacher_td_coef=0.0 clean_relation_distill_coef=0.0 clean_smooth_head_loss_coef=0.0 clean_action_pred_loss_coef=0.0 clean_public_delta_loss_coef=0.0 clean_binary_concrete_temperature=$BINARY_CONCRETE_TEMPERATURE clean_adaptive_auxiliary_target_ratio=$ADAPTIVE_TARGET_RATIO save_battle_trace=False"
}

submit_one() {
  local variant="$1"
  local auxiliary="$2"
  local seed="$3"
  local model suffix aux_args run_name job_name existing job_id

  case "$variant:$auxiliary" in
    concrete:param)
      model="rpg_dual_branch_binary_concrete_param_stability_hypercond"
      suffix="concrete_param_stability"
      ;;
    concrete:grad)
      model="rpg_dual_branch_binary_concrete_grad_consistency_hypercond"
      suffix="concrete_grad_consistency"
      ;;
    adaptive:param)
      model="rpg_dual_branch_hard_gate_adaptive_param_stability_hypercond"
      suffix="adaptive10_param_stability"
      ;;
    adaptive:grad)
      model="rpg_dual_branch_hard_gate_adaptive_grad_consistency_hypercond"
      suffix="adaptive10_grad_consistency"
      ;;
    attention_only:param)
      model="rpg_dual_branch_attention_only_hard_gate_param_stability_hypercond"
      suffix="attention_only_drop_param_stability"
      ;;
    attention_only:grad)
      model="rpg_dual_branch_attention_only_hard_gate_grad_consistency_hypercond"
      suffix="attention_only_drop_grad_consistency"
      ;;
    split_head:param)
      model="rpg_dual_branch_split_head_hard_gate_param_stability_hypercond"
      suffix="split_action_head_param_stability"
      ;;
    split_head:grad)
      model="rpg_dual_branch_split_head_hard_gate_grad_consistency_hypercond"
      suffix="split_action_head_grad_consistency"
      ;;
    *)
      echo "ERROR: unsupported variant/auxiliary: $variant/$auxiliary" >&2
      return 2
      ;;
  esac

  if [[ "$auxiliary" == "param" ]]; then
    aux_args="clean_condition_gradient_consistency_coef=0.0 clean_generated_parameter_stability_coef=$AUX_COEF"
  else
    aux_args="clean_condition_gradient_consistency_coef=$AUX_COEF clean_generated_parameter_stability_coef=0.0"
  fi

  run_name="corridor_${suffix}_10m_s${seed}"
  job_name="corridor_${suffix}_s${seed}"
  existing=$(active_job "$job_name")
  if [[ -n "$existing" ]]; then
    echo "reused active job=$existing name=$job_name"
    return
  fi

  job_id=$(sbatch --parsable \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEMORY" \
    --time="$TIME" \
    --job-name="$job_name" \
    --output=ozstar_logs/%x_%j.out \
    --error=ozstar_logs/%x_%j.err \
    --export=ALL,PYTHON_BIN="$PYTHON_BIN",CONFIG=clean_hyper,ENV_CONFIG=sc2,MAP_NAME=corridor,MODEL_TYPE="$model",SEED="$seed",T_MAX="$T_MAX",TEST_INTERVAL="$TEST_INTERVAL",BATCH_SIZE_RUN="$BATCH_SIZE_RUN",EXPECTED_BATCH_SIZE_RUN="$BATCH_SIZE_RUN",BATCH_SIZE="$BATCH_SIZE",BUFFER_SIZE="$BUFFER_SIZE",USE_WANDB="$USE_WANDB",WANDB_MODE="$WANDB_MODE",USE_CUDA="$USE_CUDA",RUN_NAME="$run_name",GROUP_NAME="$run_name",OMP_NUM_THREADS="$CPUS_PER_TASK",MKL_NUM_THREADS="$CPUS_PER_TASK",OPENBLAS_NUM_THREADS="$CPUS_PER_TASK",NUMEXPR_NUM_THREADS="$CPUS_PER_TASK",EXTRA_ARGS="$(common_args) $aux_args" \
    scripts/ozstar_train_offline.sbatch)
  printf 'submitted job=%s variant=%s auxiliary=%s seed=%s cpus=%s memory=%s model=%s\n' \
    "${job_id%%;*}" "$variant" "$auxiliary" "$seed" "$CPUS_PER_TASK" "$MEMORY" "$model"
}

submitted=0
for variant in concrete adaptive attention_only split_head; do
  for auxiliary in param grad; do
    for seed in $SEEDS; do
      if (( submitted > 0 )); then
        sleep "$SUBMIT_GAP_SECONDS"
      fi
      submit_one "$variant" "$auxiliary" "$seed"
      submitted=$((submitted + 1))
    done
  done
done

echo "processed total: $submitted"
