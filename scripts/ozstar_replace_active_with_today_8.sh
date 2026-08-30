#!/bin/bash
set -euo pipefail

# Replace every active Slurm job owned by the current user with today's eight
# experiments. Data safety order:
#   1) incrementally append every running offline W&B run;
#   2) cancel the exact captured Slurm job ids;
#   3) wait for writers to exit and append the closed runs one final time;
#   4) submit 4 Counter follow-ups + 2 Counter single-branch controls
#      + 2 Corridor single-Transformer controls.

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl-dual-branch}"
PYTHON_BIN="${PYTHON_BIN:-/home/kyang/.conda/envs/marl_cpu/bin/python}"
WANDB_ROOT="${WANDB_ROOT:-wandb}"
WANDB_ENTITY="${WANDB_ENTITY:-hjh331-sjtu}"
WANDB_PROJECT="${WANDB_PROJECT:-gomarl}"
SYNC_TIMEOUT="${SYNC_TIMEOUT:-600}"
TERMINATION_TIMEOUT="${TERMINATION_TIMEOUT:-180}"
# Slurm's %S timestamp is reported in the controller/login-node timezone,
# while W&B's offline-run directory timestamp can come from a UTC compute
# node.  Allow the largest Australian UTC offset plus a small startup margin.
START_GRACE_SECONDS="${START_GRACE_SECONDS:-50400}"
CANCEL_OLD="${CANCEL_OLD:-NO}"
SYNC_ONLY="${SYNC_ONLY:-NO}"

cd "$REPO_DIR"

if [[ "$SYNC_ONLY" != "YES" && "$SYNC_ONLY" != "NO" ]]; then
  echo "ERROR: SYNC_ONLY must be YES or NO" >&2
  exit 2
fi
if [[ "$SYNC_ONLY" != "YES" && "$CANCEL_OLD" != "YES" ]]; then
  echo "ERROR: set CANCEL_OLD=YES to confirm replacing all active Slurm jobs" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -d "$WANDB_ROOT" ]]; then
  echo "ERROR: W&B directory does not exist: $REPO_DIR/$WANDB_ROOT" >&2
  exit 2
fi
for command in awk basename date find grep scancel squeue stat timeout; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "ERROR: required command is unavailable: $command" >&2
    exit 2
  }
done

mapfile -t ACTIVE_ROWS < <(squeue -h -u "$USER" -o '%A|%j|%T|%S')
if (( ${#ACTIVE_ROWS[@]} == 0 )); then
  echo "No active jobs to replace; proceeding to the new eight."
else
  echo "== Captured active jobs =="
  printf '%s\n' "${ACTIVE_ROWS[@]}"
fi

run_name_from_job_name() {
  local job_name="$1"
  if [[ "$job_name" =~ ^(.+)_s([0-9]+)$ ]]; then
    printf '%s_10m_s%s\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
    return 0
  fi
  return 1
}

run_dir_epoch() {
  local run_dir="$1" base date_part time_part
  base="$(basename "$run_dir")"
  if [[ ! "$base" =~ ^offline-run-([0-9]{8})_([0-9]{6})- ]]; then
    return 1
  fi
  date_part="${BASH_REMATCH[1]}"
  time_part="${BASH_REMATCH[2]}"
  date -d "${date_part:0:4}-${date_part:4:2}-${date_part:6:2} ${time_part:0:2}:${time_part:2:2}:${time_part:4:2}" +%s
}

find_active_run_dir() {
  local run_name="$1" start_time="$2"
  local start_epoch earliest run_dir run_epoch run_file matched best_dir="" best_epoch=0
  start_epoch="$(date -d "$start_time" +%s)" || return 1
  earliest=$((start_epoch - START_GRACE_SECONDS))
  while IFS= read -r -d '' run_dir; do
    run_epoch="$(run_dir_epoch "$run_dir")" || continue
    (( run_epoch >= earliest )) || continue

    matched=0
    if [[ -f "$run_dir/files/config.yaml" ]] && \
       grep -qF -- "$run_name" "$run_dir/files/config.yaml"; then
      matched=1
    else
      run_file="$(find "$run_dir" -maxdepth 1 -type f -name 'run-*.wandb' -print -quit)"
      if [[ -f "$run_file" ]] && grep -aqF -- "$run_name" "$run_file"; then
        matched=1
      fi
    fi
    (( matched == 1 )) || continue

    if (( run_epoch > best_epoch )); then
      best_dir="$run_dir"
      best_epoch="$run_epoch"
    fi
  done < <(
    find "$WANDB_ROOT" -mindepth 1 -maxdepth 1 -type d \
      -name 'offline-run-*' -print0 2>/dev/null
  )
  [[ -n "$best_dir" ]] && printf '%s\n' "$best_dir"
}

sync_run() {
  local phase="$1" job_id="$2" job_name="$3" run_dir="$4" run_file
  run_file="$(find "$run_dir" -maxdepth 1 -type f -name 'run-*.wandb' -print -quit)"
  if [[ ! -f "$run_file" ]]; then
    echo "ERROR: no run-*.wandb for job=$job_id name=$job_name in $run_dir" >&2
    return 1
  fi
  echo "[$phase] syncing job=$job_id name=$job_name bytes=$(stat -c '%s' "$run_file") run=$run_dir"
  timeout "$SYNC_TIMEOUT" "$PYTHON_BIN" -m wandb sync \
    -e "$WANDB_ENTITY" \
    -p "$WANDB_PROJECT" \
    --append \
    --include-offline \
    --include-synced \
    --no-mark-synced \
    --skip-console \
    "$run_dir"
}

JOB_IDS=()
JOB_NAMES=()
RUN_DIRS=()
initial_failed=0

for row in "${ACTIVE_ROWS[@]}"; do
  IFS='|' read -r job_id job_name job_state start_time <<< "$row"
  JOB_IDS+=("$job_id")
  JOB_NAMES+=("$job_name")
  if [[ "$job_state" != "RUNNING" ]]; then
    RUN_DIRS+=("")
    echo "[initial] job=$job_id name=$job_name state=$job_state has no active W&B writer"
    continue
  fi
  run_name="$(run_name_from_job_name "$job_name")" || {
    echo "ERROR: cannot derive W&B run name from active job $job_name" >&2
    RUN_DIRS+=("")
    initial_failed=$((initial_failed + 1))
    continue
  }
  if ! run_dir="$(find_active_run_dir "$run_name" "$start_time")"; then
    run_dir=""
  fi
  if [[ -z "$run_dir" ]]; then
    echo "ERROR: active W&B directory not found for job=$job_id expected_run=$run_name" >&2
    RUN_DIRS+=("")
    initial_failed=$((initial_failed + 1))
    continue
  fi
  RUN_DIRS+=("$run_dir")
  if ! sync_run initial "$job_id" "$job_name" "$run_dir"; then
    initial_failed=$((initial_failed + 1))
  fi
done

if (( initial_failed > 0 )); then
  echo "ERROR: initial W&B update failed for $initial_failed job(s); nothing was cancelled" >&2
  exit 1
fi

if [[ "$SYNC_ONLY" == "YES" ]]; then
  echo "sync-only complete: active_jobs=${#JOB_IDS[@]}; no jobs were cancelled or submitted"
  squeue -u "$USER" -o "%.18i %.90j %.10T %.12M %.10m %R"
  exit 0
fi

if (( ${#JOB_IDS[@]} > 0 )); then
  echo "== Cancelling captured jobs =="
  printf '%s\n' "${JOB_IDS[@]}"
  scancel "${JOB_IDS[@]}"

  deadline=$(( $(date +%s) + TERMINATION_TIMEOUT ))
  while true; do
    remaining=0
    for job_id in "${JOB_IDS[@]}"; do
      if squeue -h -j "$job_id" | grep -q .; then
        remaining=$((remaining + 1))
      fi
    done
    (( remaining > 0 )) || break
    if (( $(date +%s) >= deadline )); then
      echo "ERROR: $remaining cancelled job(s) still present after ${TERMINATION_TIMEOUT}s" >&2
      exit 1
    fi
    sleep 3
  done
fi

final_failed=0
for index in "${!RUN_DIRS[@]}"; do
  run_dir="${RUN_DIRS[$index]}"
  [[ -n "$run_dir" ]] || continue
  if ! sync_run final "${JOB_IDS[$index]}" "${JOB_NAMES[$index]}" "$run_dir"; then
    final_failed=$((final_failed + 1))
  fi
done
if (( final_failed > 0 )); then
  echo "ERROR: final W&B sync failed for $final_failed run(s); new jobs were not submitted" >&2
  exit 1
fi

echo "== Submitting today's eight experiments =="
REPO_DIR="$REPO_DIR" PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/ozstar_submit_counter_episode_random_followups_4.sh
REPO_DIR="$REPO_DIR" PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/ozstar_submit_counter_single_branch_random_drop_2.sh
REPO_DIR="$REPO_DIR" PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/ozstar_submit_corridor_single_transformer_random_drop_2.sh

echo "replacement complete: previous=${#JOB_IDS[@]} new_requested=8"
squeue -u "$USER" -o "%.18i %.90j %.10T %.12M %.10m %R"
