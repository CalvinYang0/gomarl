#!/bin/bash
set -euo pipefail

# Keep the two requested Counter jobs, preserve any of the six desired
# follow-ups that are already active, stop every unrelated active job, and
# submit any missing follow-ups.

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl-dual-branch}"
PYTHON_BIN="${PYTHON_BIN:-/home/kyang/.conda/envs/marl_cpu/bin/python}"
SEED="${SEED:-1}"
CANCEL_OTHERS="${CANCEL_OTHERS:-NO}"
TERMINATION_TIMEOUT="${TERMINATION_TIMEOUT:-180}"

cd "$REPO_DIR"

if [[ "$CANCEL_OTHERS" != "YES" ]]; then
  echo "ERROR: set CANCEL_OTHERS=YES to confirm stopping all jobs except the two keep targets" >&2
  exit 2
fi

KEEP_K50="grf_counter_equal1_episode_random_randommask_k50_w10_timestep_s${SEED}"
KEEP_SAMPLED="grf_counter_equal1_episode_random_sampled_gate_s${SEED}"
FOLLOWUP_K30="grf_counter_equal1_episode_random_randommask_k30_w10_timestep_s${SEED}"
FOLLOWUP_K70="grf_counter_equal1_episode_random_randommask_k70_w10_timestep_s${SEED}"
FOLLOWUP_TRANSFORMER_GATE="grf_counter_transformer_only_obs_gate_s${SEED}"
FOLLOWUP_TRANSFORMER_RANDOM="grf_counter_transformer_only_obs_gate_randommask_d50_w10_timestep_s${SEED}"
FOLLOWUP_LINEAR_GATE="grf_counter_linear_only_obs_gate_s${SEED}"
FOLLOWUP_LINEAR_RANDOM="grf_counter_linear_only_obs_gate_randommask_d50_w10_timestep_s${SEED}"

mapfile -t ACTIVE_ROWS < <(squeue -h -u "$USER" -o '%A|%j|%T')
keep_k50_found=0
keep_sampled_found=0
CANCEL_IDS=()

echo "== Active-job selection =="
for row in "${ACTIVE_ROWS[@]}"; do
  IFS='|' read -r job_id job_name job_state <<< "$row"
  case "$job_name" in
    "$KEEP_K50")
      keep_k50_found=1
      echo "KEEP   $job_id $job_state $job_name"
      ;;
    "$KEEP_SAMPLED")
      keep_sampled_found=1
      echo "KEEP   $job_id $job_state $job_name"
      ;;
    "$FOLLOWUP_K30"|"$FOLLOWUP_K70"|\
    "$FOLLOWUP_TRANSFORMER_GATE"|"$FOLLOWUP_TRANSFORMER_RANDOM"|\
    "$FOLLOWUP_LINEAR_GATE"|"$FOLLOWUP_LINEAR_RANDOM")
      echo "REUSE  $job_id $job_state $job_name"
      ;;
    *)
      CANCEL_IDS+=("$job_id")
      echo "STOP   $job_id $job_state $job_name"
      ;;
  esac
done

if (( keep_k50_found == 0 || keep_sampled_found == 0 )); then
  echo "ERROR: both requested keep jobs must be active before any cancellation" >&2
  echo "expected: $KEEP_K50" >&2
  echo "expected: $KEEP_SAMPLED" >&2
  exit 2
fi

echo "== Preflight before any cancellation =="
"$PYTHON_BIN" -m py_compile \
  src/modules/agents/clean_hyper_agent.py \
  src/controllers/clean_controller.py \
  src/learners/clean_learner.py
"$PYTHON_BIN" scripts/smoke_test_dual_branch_dynamic_gate.py
"$PYTHON_BIN" scripts/smoke_test_counter_mask_parameter_relation.py

echo "== Incremental W&B sync before cancellation =="
SYNC_ONLY=YES REPO_DIR="$REPO_DIR" PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/ozstar_replace_active_with_today_8.sh

if (( ${#CANCEL_IDS[@]} > 0 )); then
  echo "== Stopping non-kept jobs =="
  printf '%s\n' "${CANCEL_IDS[@]}"
  scancel "${CANCEL_IDS[@]}"

  deadline=$(( $(date +%s) + TERMINATION_TIMEOUT ))
  while true; do
    remaining=0
    for job_id in "${CANCEL_IDS[@]}"; do
      if squeue -h -j "$job_id" | grep -q .; then
        remaining=$((remaining + 1))
      fi
    done
    (( remaining == 0 )) && break
    if (( $(date +%s) >= deadline )); then
      echo "ERROR: $remaining job(s) remain after ${TERMINATION_TIMEOUT}s" >&2
      exit 1
    fi
    sleep 3
  done
else
  echo "No non-kept active jobs to stop."
fi

echo "== Submitting six Counter experiments =="
REPO_DIR="$REPO_DIR" PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/ozstar_submit_counter_equal1_randommask_k30_k70_2.sh
REPO_DIR="$REPO_DIR" PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/ozstar_submit_counter_transformer_obs_gate_2.sh
REPO_DIR="$REPO_DIR" PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/ozstar_submit_counter_linear_obs_gate_2.sh

echo "complete: kept=2 new_requested=6 expected_active_total=8"
squeue -u "$USER" -o "%.18i %.90j %.10T %.12M %.10m %R"
