#!/bin/bash
set -euo pipefail

# Upload the three dual-branch Counter runs from OzSTAR's local W&B store.
# Match each offline directory to its Slurm job start time; this works even
# when the local W&B version does not write useful fields to config.yaml.
# The append/no-mark-synced combination makes this safe to rerun: completed
# runs keep the same remote run id, while an active run can upload new records
# again after it advances or finishes.

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl-dual-branch}"
PYTHON_BIN="${PYTHON_BIN:-/home/kyang/.conda/envs/marl_cpu/bin/python}"
WANDB_ROOT="${WANDB_ROOT:-wandb}"
WANDB_ENTITY="${WANDB_ENTITY:-hjh331-sjtu}"
WANDB_PROJECT="${WANDB_PROJECT:-gomarl}"
SYNC_TIMEOUT="${SYNC_TIMEOUT:-600}"
JOB_IDS="${JOB_IDS:-15022706 15022707 15022708}"
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
for command in awk basename date find sacct sort timeout; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "ERROR: required command is unavailable: $command" >&2
    exit 2
  }
done

echo "== Dual-branch Counter W&B sync =="
echo "repo: $REPO_DIR"
echo "destination: $WANDB_ENTITY/$WANDB_PROJECT"
echo "jobs: $JOB_IDS"

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

job_start_epoch() {
  local job_id="$1"
  local start
  start="$(sacct -X -n -P -j "$job_id" --format=JobIDRaw,Start \
    | awk -F'|' -v expected="$job_id" '$1 == expected && $2 != "" && $2 != "Unknown" {print $2; exit}')"
  [[ -n "$start" ]] || return 1
  date -d "$start" +%s
}

declare -A USED_RUN_DIRS
RUN_DIRS=()
for job_id in $JOB_IDS; do
  start_epoch="$(job_start_epoch "$job_id")" || {
    echo "ERROR: cannot resolve start time for Slurm job $job_id" >&2
    exit 2
  }

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
    echo "ERROR: no W&B run started near Slurm job $job_id" >&2
    exit 2
  fi
  USED_RUN_DIRS["$best_dir"]=1
  RUN_DIRS+=("$best_dir")
  echo "matched job=$job_id delta=${best_delta}s run=$best_dir"
done

synced=0
failed=0
for run_dir in "${RUN_DIRS[@]}"; do
  run_file="$(find "$run_dir" -maxdepth 1 -type f -name 'run-*.wandb' -print -quit)"
  if [[ -z "$run_file" ]]; then
    echo "ERROR: no run-*.wandb file found in $run_dir" >&2
    failed=$((failed + 1))
    continue
  fi

  echo "syncing: $run_dir"
  if timeout "$SYNC_TIMEOUT" "$PYTHON_BIN" -m wandb sync \
    -e "$WANDB_ENTITY" \
    -p "$WANDB_PROJECT" \
    --append \
    --include-offline \
    --include-synced \
    --no-mark-synced \
    --skip-console \
    "$run_dir"; then
    synced=$((synced + 1))
  else
    echo "ERROR: W&B sync failed for $run_dir" >&2
    failed=$((failed + 1))
  fi
done

echo "sync result: uploaded=$synced failed=$failed"
(( failed == 0 ))
