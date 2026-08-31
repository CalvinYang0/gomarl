#!/bin/bash
set -euo pipefail

# Final-sync the two valuable ~28h Counter runs, then remove all local runtime
# artifacts under wandb/, results/, and ozstar_logs/.  Cleanup is permitted only
# after both uploads succeed and every target resolves to an ordinary directory
# directly below REPO_DIR.

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl-dual-branch}"
PYTHON_BIN="${PYTHON_BIN:-/home/kyang/.conda/envs/marl_cpu/bin/python}"
WANDB_ENTITY="${WANDB_ENTITY:-hjh331-sjtu}"
WANDB_PROJECT="${WANDB_PROJECT:-gomarl}"
SYNC_TIMEOUT="${SYNC_TIMEOUT:-900}"
SEED="${SEED:-1}"
CLEAN_CONFIRM="${CLEAN_CONFIRM:-NO}"

cd "$REPO_DIR"
REPO_REAL="$(realpath "$REPO_DIR")"

if [[ "$CLEAN_CONFIRM" != "YES" ]]; then
  echo "ERROR: set CLEAN_CONFIRM=YES to authorize permanent runtime cleanup" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 2
fi
for command in find grep realpath stat timeout; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "ERROR: required command is unavailable: $command" >&2
    exit 2
  }
done

RUN_NAMES=(
  "grf_counter_equal1_episode_random_sampled_gate_10m_s${SEED}"
  "grf_counter_equal1_episode_random_randommask_k50_w10_timestep_10m_s${SEED}"
)

run_dir_matches() {
  local run_dir="$1" run_name="$2" run_file
  if [[ -f "$run_dir/files/config.yaml" ]] && \
     grep -qF -- "$run_name" "$run_dir/files/config.yaml"; then
    return 0
  fi
  if grep -RaqF -- "$run_name" "$run_dir/files" "$run_dir/logs" 2>/dev/null; then
    return 0
  fi
  run_file="$(find "$run_dir" -maxdepth 1 -type f -name 'run-*.wandb' -print -quit)"
  [[ -f "$run_file" ]] && grep -aqF -- "$run_name" "$run_file"
}

find_latest_run_dir() {
  local run_name="$1" run_dir base latest=""
  while IFS= read -r -d '' run_dir; do
    base="$(basename "$run_dir")"
    # Both protected jobs started on 2026-08-29.  Ignoring older directories
    # makes matching fast and prevents selecting a historical same-name run.
    [[ "$base" > "offline-run-20260828_235959" ]] || continue
    run_dir_matches "$run_dir" "$run_name" || continue
    if [[ -z "$latest" || "$run_dir" > "$latest" ]]; then
      latest="$run_dir"
    fi
  done < <(
    find "$REPO_REAL/wandb" -mindepth 1 -maxdepth 1 -type d \
      -name 'offline-run-*' -print0 2>/dev/null
  )
  [[ -n "$latest" ]] && printf '%s\n' "$latest"
}

RUN_DIRS=()
for run_name in "${RUN_NAMES[@]}"; do
  if ! run_dir="$(find_latest_run_dir "$run_name")"; then
    run_dir=""
  fi
  if [[ -z "$run_dir" ]]; then
    echo "ERROR: protected offline run not found: $run_name" >&2
    echo "Nothing was deleted." >&2
    exit 1
  fi
  run_file="$(find "$run_dir" -maxdepth 1 -type f -name 'run-*.wandb' -print -quit)"
  if [[ ! -f "$run_file" ]]; then
    echo "ERROR: protected run data is missing: $run_dir" >&2
    echo "Nothing was deleted." >&2
    exit 1
  fi
  RUN_DIRS+=("$run_dir")
  echo "protected run=$run_name bytes=$(stat -c '%s' "$run_file") dir=$run_dir"
done

echo "== Final W&B uploads =="
for index in "${!RUN_NAMES[@]}"; do
  run_name="${RUN_NAMES[$index]}"
  run_dir="${RUN_DIRS[$index]}"
  echo "syncing protected run: $run_name"
  if ! timeout "$SYNC_TIMEOUT" "$PYTHON_BIN" -m wandb sync \
      -e "$WANDB_ENTITY" \
      -p "$WANDB_PROJECT" \
      --append \
      --include-offline \
      --include-synced \
      --no-mark-synced \
      --skip-console \
      "$run_dir"; then
    echo "ERROR: protected upload failed: $run_name" >&2
    echo "Nothing was deleted." >&2
    exit 1
  fi
done

echo "== Validating cleanup targets =="
CLEAN_DIRS=(wandb results ozstar_logs)
for relative in "${CLEAN_DIRS[@]}"; do
  target="$REPO_REAL/$relative"
  if [[ ! -d "$target" || -L "$target" ]]; then
    echo "ERROR: cleanup target must be a real directory, not a symlink: $target" >&2
    echo "Nothing was deleted." >&2
    exit 1
  fi
  resolved="$(realpath "$target")"
  if [[ "$resolved" != "$REPO_REAL/$relative" ]]; then
    echo "ERROR: cleanup target escaped the repository: $target -> $resolved" >&2
    echo "Nothing was deleted." >&2
    exit 1
  fi
  echo "validated: $resolved"
done

echo "== Permanently clearing runtime data =="
for relative in "${CLEAN_DIRS[@]}"; do
  target="$REPO_REAL/$relative"
  find "$target" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
  echo "cleared: $target"
done

echo "sync-and-clean complete: uploaded=2 cleared=wandb,results,ozstar_logs"
du -sh "$REPO_REAL" "$REPO_REAL/wandb" "$REPO_REAL/results" "$REPO_REAL/ozstar_logs"
