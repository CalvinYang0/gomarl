#!/bin/bash
set -euo pipefail

# One-shot W&B update for both stability hard-gate Counter jobs. The job id
# is discovered from its exact Slurm name, then matched to the offline W&B
# directory by start timestamp.

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl-dual-branch}"
PYTHON_BIN="${PYTHON_BIN:-/home/kyang/.conda/envs/marl_cpu/bin/python}"
WANDB_ROOT="${WANDB_ROOT:-wandb}"
WANDB_ENTITY="${WANDB_ENTITY:-hjh331-sjtu}"
WANDB_PROJECT="${WANDB_PROJECT:-gomarl}"
SEED="${SEED:-1}"
LOOKBACK_DAYS="${LOOKBACK_DAYS:-14}"
SYNC_TIMEOUT="${SYNC_TIMEOUT:-600}"
MAX_START_DELTA_SECONDS="${MAX_START_DELTA_SECONDS:-1800}"

cd "$REPO_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -d "$WANDB_ROOT" ]]; then
  echo "ERROR: W&B directory does not exist: $REPO_DIR/$WANDB_ROOT" >&2
  exit 2
fi
for command in awk basename date find sacct sed sort timeout; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "ERROR: required command is unavailable: $command" >&2
    exit 2
  }
done

START_DATE="$(date -d "$LOOKBACK_DAYS days ago" +%F)"
JOB_NAMES=(
  "grf_counter_hard_gate_param_stability_s${SEED}"
  "grf_counter_hard_gate_grad_consistency_s${SEED}"
)

echo "== Dynamic-gate Counter W&B update =="
echo "repo: $REPO_DIR"
echo "destination: $WANDB_ENTITY/$WANDB_PROJECT"
echo "search window: $START_DATE to now"

mapfile -t ALL_RUN_DIRS < <(
  find "$WANDB_ROOT" -maxdepth 1 -type d -name 'offline-run-*' -print \
    | LC_ALL=C sort
)
if (( ${#ALL_RUN_DIRS[@]} == 0 )); then
  echo "ERROR: no offline-run-* directories found below $WANDB_ROOT" >&2
  exit 2
fi

run_dir_epoch() {
  local run_dir="$1"
  local base date_part time_part
  base="$(basename "$run_dir")"
  if [[ ! "$base" =~ ^offline-run-([0-9]{8})_([0-9]{6})- ]]; then
    return 1
  fi
  date_part="${BASH_REMATCH[1]}"
  time_part="${BASH_REMATCH[2]}"
  date -d "${date_part:0:4}-${date_part:4:2}-${date_part:6:2} ${time_part:0:2}:${time_part:2:2}:${time_part:4:2}" +%s
}

latest_job_record() {
  local job_name="$1"
  sacct -X -n -P -S "$START_DATE" \
    --format=JobIDRaw,JobName%60,State,Start \
    | awk -F'|' -v expected="$job_name" \
      '{name=$2; sub(/[[:space:]]+$/, "", name)} \
       name == expected && $4 != "" && $4 != "Unknown" {record=$0} \
       END {print record}'
}

declare -A USED_RUN_DIRS
JOB_IDS=()
JOB_STATES=()
RUN_DIRS=()

for job_name in "${JOB_NAMES[@]}"; do
  record="$(latest_job_record "$job_name")"
  if [[ -z "$record" ]]; then
    echo "ERROR: no recent Slurm job found with exact name $job_name" >&2
    exit 2
  fi
  IFS='|' read -r job_id matched_name job_state start_time <<< "$record"
  start_epoch="$(date -d "$start_time" +%s)"

  best_dir=""
  best_delta=$((MAX_START_DELTA_SECONDS + 1))
  for candidate in "${ALL_RUN_DIRS[@]}"; do
    [[ -z "${USED_RUN_DIRS[$candidate]:-}" ]] || continue
    candidate_epoch="$(run_dir_epoch "$candidate")" || continue
    delta=$((candidate_epoch - start_epoch))
    (( delta >= -120 && delta <= MAX_START_DELTA_SECONDS )) || continue
    absolute_delta="$delta"
    (( absolute_delta >= 0 )) || absolute_delta=$((-absolute_delta))
    if (( absolute_delta < best_delta )); then
      best_dir="$candidate"
      best_delta="$absolute_delta"
    fi
  done

  if [[ -z "$best_dir" ]]; then
    echo "ERROR: no W&B run started near job=$job_id name=$job_name" >&2
    exit 2
  fi
  USED_RUN_DIRS["$best_dir"]=1
  JOB_IDS+=("$job_id")
  JOB_STATES+=("$job_state")
  RUN_DIRS+=("$best_dir")
  echo "matched job=$job_id state=$job_state delta=${best_delta}s run=$best_dir"
done

uploaded=0
failed=0
for index in "${!RUN_DIRS[@]}"; do
  run_dir="${RUN_DIRS[$index]}"
  job_id="${JOB_IDS[$index]}"
  job_name="${JOB_NAMES[$index]}"
  job_state="${JOB_STATES[$index]}"
  run_file="$(find "$run_dir" -maxdepth 1 -type f -name 'run-*.wandb' -print -quit)"
  if [[ -z "$run_file" ]]; then
    echo "ERROR: no run-*.wandb file found for job=$job_id in $run_dir" >&2
    failed=$((failed + 1))
    continue
  fi

  echo
  echo "syncing job=$job_id name=$job_name state=$job_state"
  sacct -j "$job_id" \
    --format=JobID,JobName%40,State,Elapsed,ExitCode,MaxRSS,ReqMem \
    | sed -n '1,6p'
  if timeout "$SYNC_TIMEOUT" "$PYTHON_BIN" -m wandb sync \
    -e "$WANDB_ENTITY" \
    -p "$WANDB_PROJECT" \
    --append \
    --include-offline \
    --include-synced \
    --no-mark-synced \
    --skip-console \
    "$run_dir"; then
    uploaded=$((uploaded + 1))
  else
    echo "ERROR: W&B sync failed for job=$job_id run=$run_dir" >&2
    failed=$((failed + 1))
  fi
done

echo
echo "sync result: uploaded=$uploaded failed=$failed"
(( failed == 0 ))
