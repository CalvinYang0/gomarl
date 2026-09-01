#!/bin/bash
set -u

# One-shot incremental W&B upload for every currently running Counter job in
# this checkout.  Job start times and exact run names prevent an older run with
# the same base name from being selected.

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl-dual-branch}"
PYTHON_BIN="${PYTHON_BIN:-/home/kyang/.conda/envs/marl_cpu/bin/python}"
WANDB_ROOT="${WANDB_ROOT:-wandb}"
WANDB_ENTITY="${WANDB_ENTITY:-hjh331-sjtu}"
WANDB_PROJECT="${WANDB_PROJECT:-gomarl}"
SYNC_TIMEOUT="${SYNC_TIMEOUT:-600}"
START_GRACE_SECONDS="${START_GRACE_SECONDS:-300}"

cd "$REPO_DIR" || exit 2

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 2
fi

job_run_name() {
  local job_name="$1"
  if [[ "$job_name" =~ ^(.+)_s([0-9]+)(.*)$ ]]; then
    printf '%s_10m_s%s%s\n' \
      "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}"
    return 0
  fi
  return 1
}

run_dir_epoch() {
  local base date_part time_part
  base=$(basename "$1")
  [[ "$base" =~ ^offline-run-([0-9]{8})_([0-9]{6})- ]] || return 1
  date_part="${BASH_REMATCH[1]}"
  time_part="${BASH_REMATCH[2]}"
  date -d \
    "${date_part:0:4}-${date_part:4:2}-${date_part:6:2} ${time_part:0:2}:${time_part:2:2}:${time_part:4:2}" \
    +%s
}

find_current_run_dir() {
  local run_name="$1" earliest="$2"
  local config run_dir epoch best_dir="" best_epoch=0
  while IFS= read -r -d '' config; do
    grep -qF -- "$run_name" "$config" || continue
    run_dir="${config%/files/config.yaml}"
    epoch=$(run_dir_epoch "$run_dir") || continue
    if (( epoch >= earliest && epoch > best_epoch )); then
      best_dir="$run_dir"
      best_epoch=$epoch
    fi
  done < <(
    find "$WANDB_ROOT" -maxdepth 3 -type f \
      -path "$WANDB_ROOT/offline-run-*/files/config.yaml" -print0 2>/dev/null
  )
  [[ -n "$best_dir" ]] && printf '%s\n' "$best_dir"
}

uploaded=0
failed=0
matched=0

while IFS='|' read -r job_id job_name start_time; do
  [[ "$job_name" == grf_counter_* ]] || continue
  job_record=$(scontrol show job -o "$job_id" 2>/dev/null || true)
  work_dir=$(sed -n 's/.* WorkDir=\([^ ]*\).*/\1/p' <<< "$job_record")
  [[ "$work_dir" == "$REPO_DIR" ]] || continue
  matched=$((matched + 1))

  run_name=$(job_run_name "$job_name") || {
    echo "ERROR: cannot infer W&B run name for job=$job_id name=$job_name" >&2
    failed=$((failed + 1))
    continue
  }
  start_epoch=$(date -d "$start_time" +%s 2>/dev/null) || {
    echo "ERROR: cannot parse start time for job=$job_id: $start_time" >&2
    failed=$((failed + 1))
    continue
  }
  earliest=$((start_epoch - START_GRACE_SECONDS))
  run_dir=$(find_current_run_dir "$run_name" "$earliest")
  if [[ -z "$run_dir" ]]; then
    echo "ERROR: active offline run not found job=$job_id run=$run_name" >&2
    failed=$((failed + 1))
    continue
  fi
  run_file=$(find "$run_dir" -maxdepth 1 -type f -name 'run-*.wandb' -print -quit)
  if [[ ! -f "$run_file" ]]; then
    echo "ERROR: W&B data file missing job=$job_id dir=$run_dir" >&2
    failed=$((failed + 1))
    continue
  fi

  echo "syncing job=$job_id name=$job_name run=$run_name bytes=$(stat -c '%s' "$run_file")"
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
    failed=$((failed + 1))
  fi
done < <(squeue -u "${USER:-kyang}" -h -t R -o '%A|%j|%S')

echo "running Counter sync result: matched=$matched uploaded=$uploaded failed=$failed"
(( matched > 0 && failed == 0 ))
