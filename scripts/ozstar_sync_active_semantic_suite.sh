#!/bin/bash
set -u

# Incrementally mirror only the currently running semantic-suite jobs. Resolve
# each local run from the Slurm job start time so historical runs with the same
# display name can never be selected.

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl}"
PYTHON_BIN="${PYTHON_BIN:-/home/kyang/.conda/envs/marl_cpu/bin/python}"
USER_NAME="${USER_NAME:-${USER:-kyang}}"
SYNC_INTERVAL="${SYNC_INTERVAL:-300}"
SYNC_TIMEOUT="${SYNC_TIMEOUT:-240}"
START_GRACE_SECONDS="${START_GRACE_SECONDS:-120}"
DEBUG_LOG_MAX_BYTES="${DEBUG_LOG_MAX_BYTES:-52428800}"

cd "$REPO_DIR"

for command in squeue date find grep stat timeout; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "ERROR: required command is unavailable: $command" >&2
    exit 2
  }
done

declare -A LAST_SIZE

job_run_name() {
  local job_name="$1"
  if [[ "$job_name" =~ ^(grf_(pass|3v1|counter)|corridor)_gshr_s([0-9]+)$ ]]; then
    printf '%s_gimp_shared_s%s\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[3]}"
    return 0
  fi
  if [[ "$job_name" =~ ^(grf_(pass|3v1|counter)|corridor)_gcmp_s([0-9]+)$ ]]; then
    printf '%s_gimp_adaptive_compact_s%s\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[3]}"
    return 0
  fi
  return 1
}

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

find_current_run_dir() {
  local run_name="$1"
  local job_start_epoch="$2"
  local earliest=$((job_start_epoch - START_GRACE_SECONDS))
  local config run_dir run_epoch
  local best_dir="" best_epoch=0

  while IFS= read -r -d '' config; do
    grep -qF -- "$run_name" "$config" || continue
    run_dir="${config%/files/config.yaml}"
    run_epoch="$(run_dir_epoch "$run_dir")" || continue
    if (( run_epoch >= earliest && run_epoch > best_epoch )); then
      best_dir="$run_dir"
      best_epoch="$run_epoch"
    fi
  done < <(find wandb -maxdepth 3 -type f \
    -path 'wandb/offline-run-*/files/config.yaml' -print0 2>/dev/null)

  [[ -n "$best_dir" ]] && printf '%s\n' "$best_dir"
}

sync_once() {
  local found=0
  local job_id job_name start_time run_name start_epoch run_dir run_file size

  while IFS='|' read -r job_id job_name start_time; do
    run_name="$(job_run_name "$job_name")" || continue
    found=$((found + 1))

    start_epoch="$(date -d "$start_time" +%s 2>/dev/null)" || {
      echo "$job_id $job_name: cannot parse start time $start_time"
      continue
    }
    run_dir="$(find_current_run_dir "$run_name" "$start_epoch")"
    if [[ -z "$run_dir" ]]; then
      if [[ "$job_name" == *_gcmp_s* ]]; then
        echo "$job_id $job_name: compact mask-discovery stage; W&B starts in stage 2"
      else
        echo "$job_id $job_name: W&B run not started yet"
      fi
      continue
    fi

    run_file="$(find "$run_dir" -maxdepth 1 -type f -name 'run-*.wandb' -print -quit)"
    if [[ ! -f "$run_file" ]]; then
      echo "$job_id $job_name: W&B data file not found in $run_dir"
      continue
    fi

    size="$(stat -c '%s' "$run_file")"
    if [[ "${LAST_SIZE[$run_dir]:-}" == "$size" ]]; then
      echo "$job_id $job_name: unchanged"
      continue
    fi

    echo "$job_id $job_name: syncing $run_dir"
    timeout "$SYNC_TIMEOUT" "$PYTHON_BIN" -m wandb sync \
      --append \
      --include-offline \
      --include-synced \
      --no-mark-synced \
      --skip-console \
      "$run_dir" || echo "$job_id $job_name: sync failed or timed out"

    LAST_SIZE[$run_dir]="$(stat -c '%s' "$run_file")"
  done < <(squeue -u "$USER_NAME" -h -t R -o '%A|%j|%S')

  if (( found == 0 )); then
    echo "No running semantic-suite jobs found."
  else
    echo "Matched $found running semantic-suite job(s)."
  fi
}

while true; do
  date
  sync_once

  debug_log="wandb/debug-cli.${USER_NAME}.log"
  if [[ -f "$debug_log" ]] && (( $(stat -c '%s' "$debug_log") > DEBUG_LOG_MAX_BYTES )); then
    : > "$debug_log"
  fi

  echo "Next sync in ${SYNC_INTERVAL} seconds."
  sleep "$SYNC_INTERVAL"
done
