#!/bin/bash
set -euo pipefail

# Eight one-seed jobs: parameter/gradient auxiliaries crossed with either a
# stochastic continuous Binary-Concrete gate or a detached-EMA 10% auxiliary
# ratio, on GRF Counter and SMAC Corridor.

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl-dual-branch}"
PYTHON_BIN="${PYTHON_BIN:-/home/kyang/.conda/envs/marl_cpu/bin/python}"
SEEDS="${SEEDS:-1}"
TIME="${TIME:-1-00:00:00}"
T_MAX="${T_MAX:-10050000}"
TEST_INTERVAL="${TEST_INTERVAL:-50000}"
COUNTER_CPUS_PER_TASK="${COUNTER_CPUS_PER_TASK:-24}"
CORRIDOR_CPUS_PER_TASK="${CORRIDOR_CPUS_PER_TASK:-32}"
COUNTER_PARAM_MEMORY="${COUNTER_PARAM_MEMORY:-10G}"
COUNTER_GRAD_MEMORY="${COUNTER_GRAD_MEMORY:-16G}"
CORRIDOR_MEMORY="${CORRIDOR_MEMORY:-96G}"
COUNTER_PARAM_AUX_COEF="${COUNTER_PARAM_AUX_COEF:-0.03}"
COUNTER_GRAD_AUX_COEF="${COUNTER_GRAD_AUX_COEF:-0.1}"
CORRIDOR_AUX_COEF="${CORRIDOR_AUX_COEF:-0.01}"
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

cd "$REPO_DIR"
mkdir -p ozstar_logs

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 2
fi

"$PYTHON_BIN" scripts/smoke_test_dual_branch_dynamic_gate.py

active_job() {
  local exact_name="$1"
  squeue -u "$USER" -h -o "%i|%j|%T" |
    awk -F'|' -v expected="$exact_name" \
      '$2 == expected && ($3 == "RUNNING" || $3 == "PENDING") {print $1; exit}'
}

common_args() {
  local threads="$1"
  printf '%s' "$EXTRA_ARGS torch_num_threads=$threads torch_num_interop_threads=$TORCH_NUM_INTEROP_THREADS learner_updates_per_collect=$LEARNER_UPDATES_PER_COLLECT clean_relation_teacher_td_coef=0.0 clean_relation_distill_coef=0.0 clean_smooth_head_loss_coef=0.0 clean_action_pred_loss_coef=0.0 clean_public_delta_loss_coef=0.0 clean_binary_concrete_temperature=$BINARY_CONCRETE_TEMPERATURE clean_adaptive_auxiliary_target_ratio=$ADAPTIVE_TARGET_RATIO save_battle_trace=False"
}

submit_one() {
  local scene="$1"
  local improvement="$2"
  local auxiliary="$3"
  local seed="$4"
  local tag env_config map_name memory cpus model model_prefix suffix aux_args run_name job_name existing job_id

  case "$scene" in
    counter)
      tag="grf_counter"
      env_config="academy_counterattack_easy"
      map_name="academy_counterattack_easy"
      cpus="$COUNTER_CPUS_PER_TASK"
      if [[ "$auxiliary" == "param" ]]; then
        memory="$COUNTER_PARAM_MEMORY"
        aux_args="clean_condition_gradient_consistency_coef=0.0 clean_generated_parameter_stability_coef=$COUNTER_PARAM_AUX_COEF"
      else
        memory="$COUNTER_GRAD_MEMORY"
        aux_args="clean_condition_gradient_consistency_coef=$COUNTER_GRAD_AUX_COEF clean_generated_parameter_stability_coef=0.0"
      fi
      ;;
    corridor)
      tag="corridor"
      env_config="sc2"
      map_name="corridor"
      memory="$CORRIDOR_MEMORY"
      cpus="$CORRIDOR_CPUS_PER_TASK"
      if [[ "$auxiliary" == "param" ]]; then
        aux_args="clean_condition_gradient_consistency_coef=0.0 clean_generated_parameter_stability_coef=$CORRIDOR_AUX_COEF"
      else
        aux_args="clean_condition_gradient_consistency_coef=$CORRIDOR_AUX_COEF clean_generated_parameter_stability_coef=0.0"
      fi
      ;;
    *)
      echo "ERROR: unsupported scene: $scene" >&2
      return 2
      ;;
  esac

  case "$improvement:$auxiliary" in
    concrete:param)
      model_prefix="dual_branch_binary_concrete_param_stability_hypercond"
      suffix="concrete_param_stability"
      ;;
    concrete:grad)
      model_prefix="dual_branch_binary_concrete_grad_consistency_hypercond"
      suffix="concrete_grad_consistency"
      ;;
    adaptive:param)
      model_prefix="dual_branch_hard_gate_adaptive_param_stability_hypercond"
      suffix="adaptive10_param_stability"
      ;;
    adaptive:grad)
      model_prefix="dual_branch_hard_gate_adaptive_grad_consistency_hypercond"
      suffix="adaptive10_grad_consistency"
      ;;
    *)
      echo "ERROR: unsupported improvement/auxiliary: $improvement/$auxiliary" >&2
      return 2
      ;;
  esac

  if [[ "$scene" == "counter" ]]; then
    model="grf_abs_${model_prefix}"
    aux_args="$aux_args env_args.write_video=False"
  else
    model="rpg_${model_prefix}"
  fi

  run_name="${tag}_${suffix}_10m_s${seed}"
  job_name="${tag}_${suffix}_s${seed}"
  existing=$(active_job "$job_name")
  if [[ -n "$existing" ]]; then
    echo "reused active job=$existing name=$job_name"
    return
  fi

  job_id=$(sbatch --parsable \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$cpus" \
    --mem="$memory" \
    --time="$TIME" \
    --job-name="$job_name" \
    --output=ozstar_logs/%x_%j.out \
    --error=ozstar_logs/%x_%j.err \
    --export=ALL,PYTHON_BIN="$PYTHON_BIN",CONFIG=clean_hyper,ENV_CONFIG="$env_config",MAP_NAME="$map_name",MODEL_TYPE="$model",SEED="$seed",T_MAX="$T_MAX",TEST_INTERVAL="$TEST_INTERVAL",BATCH_SIZE_RUN="$BATCH_SIZE_RUN",EXPECTED_BATCH_SIZE_RUN="$BATCH_SIZE_RUN",BATCH_SIZE="$BATCH_SIZE",BUFFER_SIZE="$BUFFER_SIZE",USE_WANDB="$USE_WANDB",WANDB_MODE="$WANDB_MODE",USE_CUDA="$USE_CUDA",RUN_NAME="$run_name",GROUP_NAME="$run_name",OMP_NUM_THREADS="$cpus",MKL_NUM_THREADS="$cpus",OPENBLAS_NUM_THREADS="$cpus",NUMEXPR_NUM_THREADS="$cpus",EXTRA_ARGS="$(common_args "$cpus") $aux_args" \
    scripts/ozstar_train_offline.sbatch)
  printf 'submitted job=%s scene=%s improvement=%s auxiliary=%s seed=%s cpus=%s memory=%s model=%s\n' \
    "${job_id%%;*}" "$scene" "$improvement" "$auxiliary" "$seed" "$cpus" "$memory" "$model"
}

submitted=0
for scene in counter corridor; do
  for improvement in concrete adaptive; do
    for auxiliary in param grad; do
      for seed in $SEEDS; do
        if (( submitted > 0 )); then
          sleep "$SUBMIT_GAP_SECONDS"
        fi
        submit_one "$scene" "$improvement" "$auxiliary" "$seed"
        submitted=$((submitted + 1))
      done
    done
  done
done

echo "processed total: $submitted"
